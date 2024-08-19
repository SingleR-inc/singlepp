#ifndef SINGLEPP_SCALED_RANKS_HPP
#define SINGLEPP_SCALED_RANKS_HPP

#include "macros.hpp"

#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <uint8_t>

namespace singlepp {

namespace internal {

template<typename Stat_, typename Index_>
using RankedVector = std::vector<std::pair<Stat_, Index_> >;

// This class sanitizes any user-provided subsets so that we can provide a
// sorted and unique subset to the tatami extractor. We then undo the sorting
// to use the original indices in the rank filler. This entire thing is
// necessary as the behavior of the subsets isn't something that the user can
// easily control (e.g., if the reference/test datasets do not use the same
// feature ordering, in which case the subset is necessarily unsorted).
template<typename Index_>
class SubsetSorter {
private:
    bool my_use_sorted_subset = false;
    const std::vector<Index_>& my_original_subset;
    std::vector<Index_> my_sorted_subset;
    std::vector<size_t> my_original_indices;

public:
    SubsetSorter(const std::vector<Index_>& sub) : my_original_subset(sub) {
        size_t num_subset = sub.size();
        for (size_t i = 1; i < num_subset; ++i) {
            if (sub[i] <= sub[i-1]) {
                my_use_sorted_subset = true;
                break;
            }
        }

        if (my_use_sorted_subset) {
            std::vector<std::pair<Index_, size_t> > store;
            store.reserve(num_subset);
            for (size_t i = 0; i < num_subset; ++i) {
                store.emplace_back(sub[i], i);
            }

            std::sort(store.begin(), store.end());
            my_sorted_subset.reserve(num_subset);
            my_original_indices.resize(num_subset);
            for (const auto& s : store) {
                if (my_sorted_subset.empty() || my_sorted_subset.back() != s.first) {
                    my_sorted_subset.push_back(s.first);
                }
                my_original_indices[s.second] = my_sorted_subset.size() - 1;
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

    template<typename Stat_>
    void fill_ranks(const Stat_* ptr, RankedVector<Stat_, Index_>& vec) const {
        if (my_use_sorted_subset) {
            size_t num = my_original_indices.size();
            for (size_t s = 0; s < num; ++s) {
                vec[s].first = ptr[my_original_indices[s]];
                vec[s].second = s;
            }
        } else {
            size_t num = my_original_subset.size();
            for (size_t s = 0; s < num; ++s, ++ptr) {
                vec[s].first = *ptr;
                vec[s].second = s;
            }
        }
        std::sort(vec.begin(), vec.end());
    }
};

template<typename Stat_, typename Index_, typename Output_>
void scaled_ranks(const RankedVector<Stat_, Index_>& collected, Output_* outgoing) { 
    static_assert(std::is_floating_point<Output_>::value);

    // Computing tied ranks. 
    size_t cur_rank = 0;
    auto cIt = collected.begin();
    auto cEnd = collected.end();

    while (cIt != cEnd) {
        auto copy = cIt;
        ++copy;
        Output_ accumulated_rank = cur_rank;
        ++cur_rank;

        while (copy != collected.end() && copy->first == cIt->first) {
            accumulated_rank += cur_rank;
            ++cur_rank;
            ++copy;
        }

        Output_ mean_rank = accumulated_rank / static_cast<Output_>(copy - cIt);
        while (cIt != copy) {
            outgoing[cIt->second] = mean_rank;
            ++cIt;
        }
    }

    // Mean-adjusting and converting to cosine values.
    Output_ sum_squares = 0;
    size_t N = collected.size();
    const Output_ center_rank = static_cast<Output_>(N - 1)/2; 
    for (size_t i = 0 ; i < N; ++i) {
        auto& o = outgoing[i];
        o -= center_rank;
        sum_squares += o*o;
    }

    // Special behaviour for no-variance cells; these are left as all-zero scaled ranks.
    sum_squares = std::max(sum_squares, 0.00000001);
    Output_ denom = std::sqrt(sum_squares) * 2;
    for (size_t i = 0; i < N; ++i) {
        outgoing[i] /= denom;
    }

    return;
}

template<typename Index_>
class SubsetMapping {
private:
    std::vector<uint8_t> my_present;
    std::vector<Index_> my_position;

public:
    uint8_t& present(size_t i) {
        return my_present[i];
    } 

    uint8_t present(size_t i) const {
        return my_present[i];
    } 

    Index_& position(size_t i) {
        return my_position[i];
    } 

    Index_ position(size_t i) const {
        return my_position[i];
    } 

public:
    void clear() {
        std::fill(mapping.present.begin(), mapping.present.end(), 0);
    }

    void resize(size_t n) {
        if (n > mapping.present.size()) {
            std::fill(mapping.present.begin(), mapping.present.end(), 0);
            mapping.present.resize(n);
        } else {
            mapping.present.resize(n);
            std::fill(mapping.present.begin(), mapping.present.end(), 0);
        }
        mapping.position.resize(n);
    }
};

template<typename Stat_, typename Index_>
void subset_ranks(const RankedVector<Stat_, Index_>& x, RankedVector<Stat_, Index_>& output, const SubsetMapping<Index_>& subset) {
    size_t N = x.size();
    for (size_t i = 0; i < N; ++i) {
        if (subset.present(i)) {
            output.emplace_back(x[i].first, subset.position(i));
        }
    }
    return;
}

template<typename Stat_, typename Index_, typename Simple_>
void simplify_ranks(const RankedVector<Stat_, Index_>& x, RankedVector<Simple_, Index_>& output) {
    if (x.size()) {
        Simple_ counter = 0;
        auto last = x[0].first;
        for (const auto& r : x) {
            if (r.first != last) {
                ++counter;
                last = r.first;
            }
            output.emplace_back(counter, r.second);
        }
    }
    return;
}

}

}

#endif
