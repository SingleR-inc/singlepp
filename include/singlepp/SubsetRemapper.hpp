#ifndef SINGLEPP_SUBSET_REMAPPER_HPP
#define SINGLEPP_SUBSET_REMAPPER_HPP

#include "scaled_ranks.hpp"

#include <vector>
#include <limits>
#include <cstddef>
#include <type_traits>

namespace singlepp {

/*
 * This class remaps the indices in the RankVector to the subset of interest.
 * For example, if our subset of features of interest is:
 *
 *    [a, c, g, e]
 *
 * ... the user should call .add(a), .add(c), .add(e), etc. Then, when we
 * receive a rank vector like:
 *
 * [(A, a), (B, b), (C, c), (D, d), (E, e), (F, f), (G, g)],
 *
 * ... the .remap() method filters out the entries that weren't added by
 * .add(), and then remaps the remaining indices to their position on the
 * subset vector, yielding:
 *
 * [(A, 0), (C, 1), (E, 3), (G, 2)]
 *
 * The idea is to adjust the indices so it appears as if we had been working
 * with the subset of features all along, allowing us to call scaled_ranks() to
 * perform the rest of the analysis on the subsets only. This is primarily
 * intended for use on the reference rank vectors during fine-tuning, given that
 * data extracted from the reference/test matrices is already subsetted when
 * returned by SubsetSanitizer::fill_ranks().
 */
template<typename Index_>
class SubsetRemapper {
private:
    Index_ my_max_markers;

    // This uses a vector instead of an unordered_map for fast remap()
    // inside the inner loop of the fine-tuning iterations.
    std::vector<Index_> my_mapping;
    std::vector<Index_> my_used;
    Index_ my_counter = 0;

public:
    SubsetRemapper(const Index_ max_markers) : my_max_markers(max_markers) {
        sanisizer::resize(my_mapping, max_markers, max_markers);
        sanisizer::resize(my_used, max_markers);
    }

    void add(Index_ i) {
        if (my_mapping[i] == my_max_markers) {
            my_mapping[i] = my_counter;
            my_used.push_back(i);
            ++my_counter;
        }
    }

    void clear() {
        my_counter = 0;
        for (auto u : my_used) {
            my_mapping[u] = my_max_markers;
        }
        my_used.clear();
    }

    Index_ size() const {
        return my_counter;
    }

public:
    template<typename Stat_>
    void remap(typename RankedVector<Stat_, Index_>::const_iterator begin, typename RankedVector<Stat_, Index_>::const_iterator end, RankedVector<Stat_, Index_>& output) const {
        output.clear();
        for (; begin != end; ++begin) {
            const auto& target = my_mapping[begin->second];
            if (target != my_max_markers) {
                output.emplace_back(begin->first, target);
            }
        }
    }

    template<typename Stat_>
    void remap(const RankedVector<Stat_, Index_>& input, RankedVector<Stat_, Index_>& output) const {
        remap(input.begin(), input.end(), output);
    }
};

}

#endif
