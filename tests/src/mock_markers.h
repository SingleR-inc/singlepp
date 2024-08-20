#ifndef MOCK_MARKERS_H
#define MOCK_MARKERS_H

#include "singlepp/subset_to_markers.hpp"
#include <random>
#include <algorithm>

template<typename Index_, class Engine_>
void fill_markers(std::vector<Index_>& source, size_t len, size_t universe, Engine_& rng) {
    source.resize(universe);
    std::iota(source.begin(), source.end(), 0);
    std::shuffle(source.begin(), source.end(), rng);
    if (len < universe) {
        source.resize(len);
    }
}

template<typename Index_>
singlepp::Markers<Index_> mock_markers(size_t nlabels, size_t len, size_t universe, int seed = 42) {
    singlepp::Markers<Index_> output(nlabels);    
    std::mt19937_64 rng(seed);
    for (size_t i = 0; i < nlabels; ++i) {
        output[i].resize(nlabels);
        for (size_t j = 0; j < nlabels; ++j) {
            if (i != j) {
                fill_markers(output[i][j], len, universe, rng);
            }
        }
    }
    return output;
}

template<typename Index_>
singlepp::Markers<Index_> mock_markers_diagonal(size_t nlabels, size_t len, size_t universe, int seed = 42) {
    singlepp::Markers<Index_> output(nlabels);    
    std::mt19937_64 rng(seed);
    for (size_t i = 0; i < nlabels; ++i) {
        output[i].resize(nlabels);
        fill_markers(output[i][i], len, universe, rng);
    }
    return output;
}

template<typename Index_>
singlepp::internal::Intersection<Index_> mock_intersection(size_t n1, size_t n2, size_t shared, int seed = 999) {
    std::mt19937_64 rng(seed);

    auto choose = [&](size_t n, size_t s) -> auto {
        std::vector<int> chosen;
        for (size_t i = 0; i < n; ++i) {
            double prob = (rng() % 100000) / 100000.0;
            if (prob < static_cast<double>(s - chosen.size()) / (n - i)) {
                chosen.push_back(i);
            }
        }
        return chosen;
    };

    auto chosen1 = choose(n1, shared);
    std::sort(chosen1.begin(), chosen1.end());
    auto chosen2 = choose(n2, shared);
    std::shuffle(chosen2.begin(), chosen2.end(), rng);

    singlepp::internal::Intersection<Index_> inter;
    inter.test_n = n1;
    inter.ref_n = n2;
    for (size_t i = 0; i < shared; ++i) {
        inter.pairs.emplace_back(chosen1[i], chosen2[i]);
    }

    return inter;
}

#endif
