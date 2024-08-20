#ifndef SINGLEPP_INTERSECT_FEATURES_HPP
#define SINGLEPP_INTERSECT_FEATURES_HPP

#include "macros.hpp"

#include <vector>
#include <algorithm>
#include <cstdint>
#include <numeric>

namespace singlepp {

namespace internal {

template<typename Index_>
struct Intersection {
    std::vector<std::pair<Index_, Index_> > pairs; // (index in target, index in reference)
    Index_ test_n, ref_n;
};

template<typename Index_, typename Id_>
Intersection<Index_> intersect_features(Index_ test_n, const Id_* test_id, Index_ ref_n, const Id_* ref_id) {
    size_t max_num = 0; 
    if (test_n) {
        max_num = std::max(max_num, static_cast<size_t>(*std::max_element(test_id, test_id + test_n)) + 1);
    }
    if (ref_n) {
        max_num = std::max(max_num, static_cast<size_t>(*std::max_element(ref_id, ref_id + ref_n)) + 1);
    }

    std::vector<Index_> test_id_reordered(max_num);
    std::vector<uint8_t> found(max_num);

    for (Index_ i = 0; i < test_n; ++i) {
        auto current = test_id[i];
        auto& curfound = found[current];
        if (!curfound) { // only using the first occurrence of each ID in test_id.
            test_id_reordered[current] = i;
            curfound = 1;
        }
    }

    Intersection<Index_> output;
    output.test_n = test_n;
    output.ref_n = ref_n;

    for (Index_ i = 0; i < ref_n; ++i) {
        auto current = ref_id[i];
        auto& curfound = found[current];
        if (curfound) {
            output.pairs.emplace_back(test_id_reordered[current], i);
            curfound = 0; // only using the first occurrence of each ID in ref_id.
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
