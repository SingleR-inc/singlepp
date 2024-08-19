#ifndef SINGLEPP_PROCESS_FEATURES_HPP
#define SINGLEPP_PROCESS_FEATURES_HPP

#include "macros.hpp"

#include "Markers.hpp"

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
void subset_to_markers(Intersection<Index_>& intersection, Markers<Index_>& markers, int top) {
    std::vector<uint8_t> available(intersection.ref_n);
    for (const auto& in : intersection.pairs) {
        available[in.second] = 1;
    }

    // Figuring out the top markers to retain, that are _also_ in the intersection.
    std::vector<uint8_t> all_markers(intersection.ref_n);
    {
        size_t ngroups = markers.size();
        for (size_t i = 0; i < ngroups; ++i) {
            auto& inner_markers = markers[i];
            size_t inner_ngroups = inner_markers.size(); // should be the same as ngroups, but we'll just do this to be safe.

            for (size_t j = 0; j < inner_ngroups; ++j) {
                auto& current = inner_markers[j];

                std::vector<Index_> replacement;
                size_t upper_bound = static_cast<size_t>(top >= 0 ? top : -1); // in effect, no upper bound if top = -1.
                size_t output_size = std::min(current.size(), upper_bound);

                if (output_size) {
                    replacement.reserve(output_size);
                    for (auto marker : current) {
                        if (available[marker]) {
                            all_markers[marker] = 1;
                            replacement.push_back(marker);
                            if (replacement.size() == output_size) {
                                break;
                            }
                        }
                    }
                }

                current.swap(replacement);
            }
        }
    }

    // Subsetting the intersection down to the chosen set of markers.
    std::vector<Index_> mapping(intersection.ref_n);
    {
        size_t counter = 0;
        size_t npairs = intersection.pairs.size();
        for (size_t i = 0; i < npairs; ++i) {
            const auto& in = intersection.pairs[i];
            if (all_markers[in.second]) {
                mapping[in.second] = counter;
                intersection.pairs[counter] = in;
                ++counter;
            }
        }
        intersection.pairs.resize(counter);
    }

    // Reindexing the markers.
    {
        size_t ngroups = markers.size();
        for (size_t i = 0; i < ngroups; ++i) {
            auto& inner_markers = markers[i];
            size_t inner_ngroups = inner_markers.size();
            for (size_t j = 0; j < inner_ngroups; ++j) {
                for (auto& k : inner_markers[j]) {
                    k = mapping[k];
                }
            }
        }
    }

    return;
}

// Use this method when the feature spaces are already identical.
template<typename Index_>
std::vector<Index_> subset_to_markers(Markers<Index_>& markers, int top) {
    std::vector<uint8_t> available;

    {
        size_t ngroups = markers.size();
        for (size_t i = 0; i < ngroups; ++i) {
            auto& inner_markers = markers[i];
            size_t inner_ngroups = inner_markers.size();

            for (size_t j = 0; j < inner_ngroups; ++j) {
                auto& current = inner_markers[j];
                if (top >= 0) {
                    current.resize(std::min(current.size(), static_cast<size_t>(top)));
                }
                if (current.size()) {
                    size_t biggest = static_cast<size_t>(*std::max_element(current.begin(), current.end()));
                    if (biggest >= available.size()) {
                        available.resize(biggest + 1);
                    }
                    for (auto x : current) {
                        available[x] = 1;
                    }
                }
            }
        }
    }

    std::vector<Index_> subset, mapping;
    {
        size_t nmarkers = std::accumulate(available.begin(), available.end(), static_cast<size_t>(0));
        subset.reserve(nmarkers);
        mapping.resize(available.size());

        for (Index_ i = 0, end = available.size(); i < end; ++i) {
            if (available[i]) {
                mapping[i] = subset.size();
                subset.push_back(i);
            }
        }
    }

    {
        size_t ngroups = markers.size();
        for (size_t i = 0; i < ngroups; ++i) {
            auto& inner_markers = markers[i];
            size_t inner_ngroups = inner_markers.size();
            for (size_t j = 0; j < inner_ngroups; ++j) {
                for (auto& k : inner_markers[j]) {
                    k = mapping[k];
                }
            }
        }
    }

    return subset;
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
