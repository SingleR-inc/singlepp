#ifndef SINGLEPP_SUBSET_TO_MARKERS_HPP
#define SINGLEPP_SUBSET_TO_MARKERS_HPP

#include "Markers.hpp"
#include "Intersection.hpp"
#include "utils.hpp"

#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <unordered_map>
#include <type_traits>
#include <optional>
#include <cstddef>

namespace singlepp {

namespace internal {

template<typename Size_>
Size_ cap_at_top(Size_ size, const std::optional<std::size_t>& top) {
    if (top.has_value()) {
        return sanisizer::min(size, *top);
    } else {
        return size;
    }
}

// Use this method when the feature spaces are already identical.
template<typename Index_>
std::vector<Index_> subset_to_markers(Markers<Index_>& markers, const std::optional<std::size_t>& top) {
    std::unordered_set<Index_> available;

    const auto ngroups = markers.size();
    for (I<decltype(ngroups)> i = 0; i < ngroups; ++i) {
        auto& inner_markers = markers[i];
        auto inner_ngroups = inner_markers.size();

        for (I<decltype(inner_ngroups)> j = 0; j < inner_ngroups; ++j) {
            auto& current = inner_markers[j];
            current.resize(cap_at_top(current.size(), top));
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

    for (I<decltype(ngroups)> i = 0; i < ngroups; ++i) {
        auto& inner_markers = markers[i];
        const auto inner_ngroups = inner_markers.size();
        for (I<decltype(inner_ngroups)> j = 0; j < inner_ngroups; ++j) {
            for (auto& k : inner_markers[j]) {
                k = mapping.find(k)->second;
            }
        }
    }

    return subset;
}

template<typename Index_>
std::pair<std::vector<Index_>, std::vector<Index_> > subset_to_markers(
    const Intersection<Index_>& intersection,
    Markers<Index_>& markers,
    const std::optional<std::size_t>& top
) {
    std::unordered_set<Index_> available;
    available.reserve(intersection.size());
    for (const auto& in : intersection) {
        available.insert(in.second);
    }

    // Figuring out the top markers to retain, that are _also_ in the intersection.
    std::unordered_set<Index_> all_markers;
    const auto ngroups = markers.size();
    for (I<decltype(ngroups)> i = 0; i < ngroups; ++i) {
        auto& inner_markers = markers[i];
        auto inner_ngroups = inner_markers.size(); // should be the same as ngroups, but we'll just do this to be safe.

        for (I<decltype(inner_ngroups)> j = 0; j < inner_ngroups; ++j) {
            auto& current = inner_markers[j];
            std::vector<Index_> replacement;
            auto output_size = cap_at_top(current.size(), top);

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
    Index_ counter = 0;
    for (const auto& in : intersection) {
        if (all_markers.find(in.second) != all_markers.end()) {
            mapping[in.second] = counter;
            output.first.push_back(in.first);
            output.second.push_back(in.second);
            ++counter;
        }
    }

    // Reindexing the markers.
    for (I<decltype(ngroups)> i = 0; i < ngroups; ++i) {
        auto& inner_markers = markers[i];
        const auto inner_ngroups = inner_markers.size();
        for (I<decltype(inner_ngroups)> j = 0; j < inner_ngroups; ++j) {
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
