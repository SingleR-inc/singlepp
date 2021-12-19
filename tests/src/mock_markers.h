#ifndef MOCK_MARKERS_H
#define MOCK_MARKERS_H

#include "singlepp/process_features.hpp"
#include <random>

inline singlepp::Markers mock_markers(size_t nlabels, size_t len, size_t universe, int seed = 42) {
    singlepp::Markers output(nlabels);    
    std::mt19937_64 rng(seed);

    for (size_t i = 0; i < nlabels; ++i) {
        output[i].resize(nlabels);
        for (size_t j = 0; j < nlabels; ++j) {
            if (i != j) {
                for (size_t k = 0; k < len; ++k) {
                    output[i][j].push_back(rng() % universe);
                }
            }
        }
    }

    return output;
}

#endif
