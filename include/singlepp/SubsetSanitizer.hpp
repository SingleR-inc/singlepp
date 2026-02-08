#ifndef SINGLEPP_SUBSET_SANITIZER_HPP
#define SINGLEPP_SUBSET_SANITIZER_HPP

#include "utils.hpp"
#include "scaled_ranks.hpp"

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include <algorithm>
#include <vector>
#include <type_traits>

namespace singlepp {

/*
 * This class sanitizes any user-provided subsets so that we can provide a
 * sorted and unique subset to the tatami extractor. We then undo the sorting
 * to use the original indices in the rank filler. This entire thing is
 * necessary as the behavior of the subsets isn't something that the user can
 * easily control (e.g., if the reference/test datasets do not use the same
 * feature ordering, in which case the subset is necessarily unsorted).
 */
template<bool sparse_, typename Index_>
class SubsetSanitizer {
private:
    bool my_use_sorted_subset = false;
    const std::vector<Index_>& my_original_subset;
    std::vector<Index_> my_sorted_subset;

    typedef typename std::vector<Index_>::size_type Size; // index type of the input subset vector in the constructor.
    typename std::conditional<sparse_, bool, std::vector<Size> >::type my_original_indices;

    typename std::conditional<sparse_, std::vector<Index_>, bool>::type my_remapping;
    typename std::conditional<sparse_, Index_, bool>::type my_remap_start = 0;

public:
    SubsetSanitizer(const std::vector<Index_>& sub) : my_original_subset(sub) {
        const auto num_subset = sub.size();
        for (I<decltype(num_subset)> i = 1; i < num_subset; ++i) {
            if (sub[i] <= sub[i-1]) {
                my_use_sorted_subset = true;
                break;
            }
        }

        if (my_use_sorted_subset) {
            std::vector<std::pair<Index_, Size> > store;
            sanisizer::reserve(store, num_subset);
            for (I<decltype(num_subset)> i = 0; i < num_subset; ++i) {
                store.emplace_back(sub[i], i);
            }
            std::sort(store.begin(), store.end());

            sanisizer::reserve(my_sorted_subset, num_subset);
            if constexpr(sparse_) {
                my_remap_start = store.front().first;
                sanisizer::resize(my_remapping, store.back().first - my_remap_start + 1);
            } else {
                sanisizer::resize(my_original_indices, num_subset);
            }

            for (const auto& s : store) {
                if (my_sorted_subset.empty() || my_sorted_subset.back() != s.first) {
                    my_sorted_subset.push_back(s.first);
                }
                if constexpr(sparse_) {
                    my_remapping[s.first - my_remap_start] = s.second;
                } else {
                    my_original_indices[s.second] = my_sorted_subset.size() - 1;
                }
            }
        }
    }

public:
    const std::vector<Index_>& extraction_subset() const {
        if (my_use_sorted_subset) {
            return my_sorted_subset;
        } else {
            return my_original_subset;
        }
    }

    template<typename Input_, typename Stat_>
    void fill_ranks(const Input_& input, RankedVector<Stat_, Index_>& vec) const {
        // The indices in the output 'vec' refer to positions on the subset
        // vector, as if the input data was already subsetted. 
        vec.clear();

        if constexpr(sparse_) {
            if (my_use_sorted_subset) {
                for (I<decltype(input.number)> s = 0; s < input.number; ++s) {
                    vec.emplace_back(input.value[s], my_remapping[input.index[s] - my_remap_start]);
                }
            } else {
                for (I<decltype(input.number)> s = 0; s < input.number; ++s) {
                    vec.emplace_back(input.value[s], s);
                }
            }

        } else {
            if (my_use_sorted_subset) {
                const auto num = my_original_indices.size();
                for (I<decltype(num)> s = 0; s < num; ++s) {
                    vec.emplace_back(input[my_original_indices[s]], s);
                }
            } else {
                const auto num = my_original_subset.size();
                for (I<decltype(num)> s = 0; s < num; ++s) {
                    vec.emplace_back(input[s], s);
                }
            }
        }

        // RankedVector's whole schtick is that it's sorted, so we make sure of it here.
        std::sort(vec.begin(), vec.end());
    }
};

}

#endif
