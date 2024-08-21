#ifndef SINGLEPP_INTERSECTION_HPP
#define SINGLEPP_INTERSECTION_HPP

#include "macros.hpp"

#include <vector>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <unordered_map>

/**
 * @file Intersection.hpp
 * @brief Intersection of features.
 */

namespace singlepp {

/**
 * Intersection of features between two datasets (typically test and reference).
 * Each element corresponds to a pair of matching features and contains the row indices of those features in the test (`first`) or reference (`second`) dataset.
 */
template<typename Index_>
using Intersection = std::vector<std::pair<Index_, Index_> >;

/**
 * @cond
 */
namespace internal {

template<typename Index_, typename Id_>
Intersection<Index_> intersect_features(Index_ test_n, const Id_* test_id, Index_ ref_n, const Id_* ref_id) {
    std::unordered_map<Id_, Index_> ref_found;
    for (Index_ i = 0; i < ref_n; ++i) {
        auto current = ref_id[i];
        auto tfIt = ref_found.find(current);
        if (tfIt == ref_found.end()) { // only using the first occurrence of each ID in ref_id.
            ref_found[current] = i;
        }
    }

    Intersection<Index_> output;
    for (Index_ i = 0; i < test_n; ++i) {
        auto current = test_id[i];
        auto tfIt = ref_found.find(current);
        if (tfIt != ref_found.end()) {
            output.emplace_back(i, tfIt->second);
            ref_found.erase(tfIt); // only using the first occurrence of each ID in test_id; the next will not enter this clause.
        }
    }

    // This is implicitly sorted by the test indices... not that it really
    // matters, as subset_to_markers() doesn't care that it's unsorted.
    return output;
}

template<typename Index_>
std::pair<std::vector<Index_>, std::vector<Index_> > unzip(const Intersection<Index_>& intersection) {
    size_t n = intersection.size();
    std::vector<Index_> left(n), right(n);
    for (size_t i = 0; i < n; ++i) {
        left[i] = intersection[i].first;
        right[i] = intersection[i].second;
    }
    return std::make_pair(std::move(left), std::move(right));
}

}
/**
 * @endcond
 */

}

#endif
