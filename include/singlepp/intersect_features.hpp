#ifndef SINGLEPP_INTERSECT_FEATURES_HPP
#define SINGLEPP_INTERSECT_FEATURES_HPP

#include "macros.hpp"

#include <vector>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <unordered_map>

namespace singlepp {

namespace internal {

template<typename Index_>
struct Intersection {
    std::vector<std::pair<Index_, Index_> > pairs; // (index in target, index in reference)
    Index_ test_n, ref_n;
};

template<typename Index_, typename Id_>
Intersection<Index_> intersect_features(Index_ test_n, const Id_* test_id, Index_ ref_n, const Id_* ref_id) {
    std::unordered_map<Id_, Index_> test_found;
    for (Index_ i = 0; i < test_n; ++i) {
        auto current = test_id[i];
        auto tfIt = test_found.find(current);
        if (tfIt == test_found.end()) { // only using the first occurrence of each ID in test_id.
            test_found[current] = i;
        }
    }

    Intersection<Index_> output;
    output.test_n = test_n;
    output.ref_n = ref_n;

    for (Index_ i = 0; i < ref_n; ++i) {
        auto current = ref_id[i];
        auto tfIt = test_found.find(current);
        if (tfIt != test_found.end()) {
            output.pairs.emplace_back(tfIt->second, i);
            test_found.erase(tfIt); // only using the first occurrence of each ID in ref_id; the next will not enter this clause.
        }
    }

    std::sort(output.pairs.begin(), output.pairs.end());
    return output;
}

template<typename Index_>
std::pair<std::vector<Index_>, std::vector<Index_> > unzip(const Intersection<Index_>& intersection) {
    size_t n = intersection.pairs.size();
    std::vector<Index_> left(n), right(n);
    for (size_t i = 0; i < n; ++i) {
        left[i] = intersection.pairs[i].first;
        right[i] = intersection.pairs[i].second;
    }
    return std::make_pair(std::move(left), std::move(right));
}

}

}

#endif
