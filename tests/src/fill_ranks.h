#ifndef FILL_RANKS_H
#define FILL_RANKS_H

#include <numeric>
#include <vector>
#include <algorithm>
#include "singlepp/scaled_ranks.hpp"

template<typename Stat, typename Index = int>
singlepp::RankedVector<Stat, Index> fill_ranks(size_t n, const Stat* ptr) {
    singlepp::RankedVector<Stat, Index> vec(n);
    for (size_t s = 0; s < n; ++s, ++ptr) {
        vec[s].first = *ptr;
        vec[s].second = s;
    }
    std::sort(vec.begin(), vec.end());
    return vec;
}

inline std::vector<double> quick_scaled_ranks(const std::vector<double>& values) {
    auto vec = fill_ranks(values.size(), values.data());
    std::vector<double> scaled(values.size());
    singlepp::scaled_ranks(vec, scaled.data());
    return scaled;
}

#endif
