#ifndef SINGLEPP_SUBSET_REMAPPER_HPP
#define SINGLEPP_SUBSET_REMAPPER_HPP

#include "scaled_ranks.hpp"

#include <vector>
#include <limits>

namespace singlepp {

namespace internal {

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
    // This uses a vector instead of an unordered_map for fast remap()
    // inside the inner loop of the fine-tuning iterations.
    std::vector<std::pair<bool, Index_> > my_mapping;
    std::vector<size_t> my_used;
    Index_ my_counter = 0;

public:
    void add(size_t i) {
        if (i >= my_mapping.size()) {
            my_mapping.resize(i + 1);
        }
        if (!my_mapping[i].first) {
            my_mapping[i].first = true;
            my_mapping[i].second = my_counter;
            my_used.push_back(i);
            ++my_counter;
        }
    }

    void clear() {
        my_counter = 0;
        for (auto u : my_used) {
            my_mapping[u].first = false;
        }
        my_used.clear();
    }

    void reserve(size_t n) {
        my_mapping.reserve(n);
    }

public:
    template<typename Stat_>
    void remap(const RankedVector<Stat_, Index_>& input, RankedVector<Stat_, Index_>& output) const {
        output.clear();

        if (static_cast<size_t>(std::numeric_limits<Index_>::max()) < my_mapping.size()) {
            // Avoid unnecessary check if the size is already greater than the largest possible index.
            // This also avoids the need to cast to indices size_t for comparison to my_mapping.size().
            for (const auto& x : input) {
                const auto& target = my_mapping[x.second];
                if (target.first) {
                    output.emplace_back(x.first, target.second);
                }
            }

        } else {
            // Otherwise, it is safe to cast the size to Index_ outside the
            // loop so that we don't need to cast x.second to size_t inside the loop.
            Index_ maxed = my_mapping.size();
            for (const auto& x : input) {
                if (maxed > x.second) {
                    const auto& target = my_mapping[x.second];
                    if (target.first) {
                        output.emplace_back(x.first, target.second);
                    }
                }
            }
        }
    }
};

}

}

#endif
