#ifndef SINGLEPP_SUBSET_TO_MARKERS_HPP
#define SINGLEPP_SUBSET_TO_MARKERS_HPP

#include "Markers.hpp"
#include "Intersection.hpp"

#include <vector>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <unordered_set>
#include <unordered_map>

namespace singlepp {

namespace internal {

// Use this method when the feature spaces are already identical.
template<typename Index_>
std::vector<Index_> subset_to_markers(Markers<Index_>& markers, int top) {
    std::unordered_set<Index_> available;

    size_t ngroups = markers.size();
    for (size_t i = 0; i < ngroups; ++i) {
        auto& inner_markers = markers[i];
        size_t inner_ngroups = inner_markers.size();

        for (size_t j = 0; j < inner_ngroups; ++j) {
            auto& current = inner_markers[j];
            if (top >= 0) {
                current.resize(std::min(current.size(), static_cast<size_t>(top)));
            }
            available.insert(current.begin(), current.end());
        }
    }

    std::vector<Index_> subset(available.begin(), available.end());
    std::sort(subset.begin(), subset.end());

    std::unordered_map<Index_, Index_> mapping;
    mapping.reserve(subset.size());
    for (Index_ i = 0, end = subset.size(); i < end; ++i) {
        mapping[subset[i]] = i;
    }

    for (size_t i = 0; i < ngroups; ++i) {
        auto& inner_markers = markers[i];
        size_t inner_ngroups = inner_markers.size();
        for (size_t j = 0; j < inner_ngroups; ++j) {
            for (auto& k : inner_markers[j]) {
                k = mapping.find(k)->second;
            }
        }
    }

    return subset;
}

template<typename Index_>
std::pair<std::vector<Index_>, std::vector<Index_> > subset_to_markers(const Intersection<Index_>& intersection, Markers<Index_>& markers, int top) {
    std::unordered_set<Index_> available;
    available.reserve(intersection.size());
    for (const auto& in : intersection) {
        available.insert(in.second);
    }

    // Figuring out the top markers to retain, that are _also_ in the intersection.
    std::unordered_set<Index_> all_markers;
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
                    if (available.find(marker) != available.end()) {
                        all_markers.insert(marker);
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

    // Subsetting the intersection down to the chosen set of markers.
    std::unordered_map<Index_, Index_> mapping;
    mapping.reserve(all_markers.size());
    std::pair<std::vector<Index_>, std::vector<Index_> > output;
    size_t counter = 0;
    for (const auto& in : intersection) {
        if (all_markers.find(in.second) != all_markers.end()) {
            mapping[in.second] = counter;
            output.first.push_back(in.first);
            output.second.push_back(in.second);
            ++counter;
        }
    }

    // Reindexing the markers.
    for (size_t i = 0; i < ngroups; ++i) {
        auto& inner_markers = markers[i];
        size_t inner_ngroups = inner_markers.size();
        for (size_t j = 0; j < inner_ngroups; ++j) {
            for (auto& k : inner_markers[j]) {
                k = mapping.find(k)->second;
            }
        }
    }

    return output;
}

}

}

#endif
