#ifndef MOCK_MARKERS_H
#define MOCK_MARKERS_H

#include "singlepp/process_features.hpp"
#include <random>
#include <algorithm>

inline singlepp::Markers mock_markers(size_t nlabels, size_t len, size_t universe, int seed = 42) {
    singlepp::Markers output(nlabels);    
    std::mt19937_64 rng(seed);

    for (size_t i = 0; i < nlabels; ++i) {
        output[i].resize(nlabels);
        for (size_t j = 0; j < nlabels; ++j) {
            if (i != j) {
                auto& source = output[i][j];
                source.resize(universe);
                std::iota(source.begin(), source.end(), 0);
                std::shuffle(source.begin(), source.end(), rng);
                if (len < universe) {
                    source.resize(len);
                }
            }
        }
    }

    return output;
}

#endif
