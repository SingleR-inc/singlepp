#ifndef FILL_RANKS_H
#define FILL_RANKS_H

#include <numeric>
#include <vector>
#include "singlepp/scaled_ranks.hpp"

// Testing overloads.
template<typename Stat, typename Index = int>
singlepp::RankedVector<Stat, Index> fill_ranks(const std::vector<int>& subset, const Stat* ptr, int offset = 0) {
    singlepp::RankedVector<Stat, Index> vec(subset.size());
    singlepp::fill_ranks(subset, ptr, vec, offset);
    return vec;
}

template<typename Stat, typename Index = int>
singlepp::RankedVector<Stat, Index> fill_ranks(size_t n, const Stat* ptr) {
    std::vector<int> everything(n);
    std::iota(everything.begin(), everything.end(), 0);
    return fill_ranks(everything, ptr);
}

#endif
