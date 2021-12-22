#ifndef SINGLEPP_SPAWN_MATRIX_H
#define SINGLEPP_SPAWN_MATRIX_H

#include "tatami/tatami.hpp"
#include <memory>
#include <random>
#include <vector>

inline std::shared_ptr<tatami::Matrix<double, int> > spawn_matrix(size_t nr, size_t nc, int seed) {
    std::vector<double> contents(nr*nc);
    std::mt19937_64 rng(seed);
    std::normal_distribution<> dist;
    for (auto& c : contents) {
        c = dist(rng);
    }
    return std::shared_ptr<tatami::Matrix<double, int> >(new tatami::DenseColumnMatrix<double, int>(nr, nc, std::move(contents)));
}

#endif
