#ifndef FILL_RANKS_H
#define FILL_RANKS_H

#include <numeric>
#include <vector>
#include <algorithm>
#include "singlepp/scaled_ranks.hpp"

template<typename Index_, typename Stat_>
singlepp::RankedVector<Stat_, Index_> fill_ranks(Index_ n, const Stat_* ptr) {
    singlepp::RankedVector<Stat_, Index_> vec(n);
    for (Index_ s = 0; s < n; ++s, ++ptr) {
        vec[s].first = *ptr;
        vec[s].second = s;
    }
    std::sort(vec.begin(), vec.end());
    return vec;
}

template<typename Stat_>
std::vector<Stat_> quick_scaled_ranks(const std::vector<Stat_>& values) {
    auto vec = fill_ranks(values.size(), values.data());
    std::vector<Stat_> scaled(values.size());
    singlepp::scaled_ranks(values.size(), vec, scaled.data());
    return scaled;
}

#endif
