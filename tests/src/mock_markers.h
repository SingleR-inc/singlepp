#ifndef MOCK_MARKERS_H
#define MOCK_MARKERS_H

#include "singlepp/subset_to_markers.hpp"

#include <random>
#include <algorithm>
#include <numeric>
#include <cstddef>

template<typename Index_, class Engine_>
void fill_markers(std::vector<Index_>& source, std::size_t len, std::size_t universe, Engine_& rng) {
    // Adjusted version of the sampling algorithm from R's sample.int() function.
    source.resize(universe);
    std::iota(source.begin(), source.end(), 0);
    std::size_t used = 0;
    while (used < len && used < universe) {
        auto& chosen = source[rng() % (universe - used) + used];
        std::swap(source[used], chosen);
        ++used;
    }
    if (len < universe) {
        source.resize(len);
    }
}

template<typename Index_>
singlepp::PairwiseMarkers<Index_> mock_pairwise_markers(std::size_t nlabels, std::size_t len, std::size_t universe, unsigned long long seed) {
    singlepp::PairwiseMarkers<Index_> output(nlabels);    
    std::mt19937_64 rng(seed);
    for (std::size_t i = 0; i < nlabels; ++i) {
        output[i].resize(nlabels);
        for (std::size_t j = 0; j < nlabels; ++j) {
            if (i != j) {
                fill_markers(output[i][j], len, universe, rng);
            }
        }
    }
    return output;
}

template<typename Index_>
singlepp::PairwiseMarkers<Index_> mock_diagonal_markers(std::size_t nlabels, std::size_t len, std::size_t universe, unsigned long long seed) {
    singlepp::PairwiseMarkers<Index_> output(nlabels);    
    std::mt19937_64 rng(seed);
    for (std::size_t i = 0; i < nlabels; ++i) {
        output[i].resize(nlabels);
        fill_markers(output[i][i], len, universe, rng);
    }
    return output;
}

template<typename Index_>
singlepp::Intersection<Index_> mock_intersection(std::size_t n1, std::size_t n2, std::size_t shared, unsigned long long seed) {
    std::mt19937_64 rng(seed);

    auto choose = [&](std::size_t n, std::size_t s) -> auto {
        std::vector<int> chosen;
        for (std::size_t i = 0; i < n; ++i) {
            double prob = (rng() % 100000) / 100000.0;
            if (prob < static_cast<double>(s - chosen.size()) / (n - i)) {
                chosen.push_back(i);
            }
        }
        return chosen;
    };

    auto chosen1 = choose(n1, shared);
    std::shuffle(chosen1.begin(), chosen1.end(), rng);
    auto chosen2 = choose(n2, shared);
    std::shuffle(chosen2.begin(), chosen2.end(), rng);

    singlepp::Intersection<Index_> inter;
    for (std::size_t i = 0; i < shared; ++i) {
        inter.emplace_back(chosen1[i], chosen2[i]);
    }

    return inter;
}

#endif
