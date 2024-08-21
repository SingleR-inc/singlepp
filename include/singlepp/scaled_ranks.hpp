#ifndef SINGLEPP_SCALED_RANKS_HPP
#define SINGLEPP_SCALED_RANKS_HPP

#include "macros.hpp"

#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace singlepp {

namespace internal {

template<typename Stat_, typename Index_>
using RankedVector = std::vector<std::pair<Stat_, Index_> >;

template<typename Stat_, typename Index_, typename Output_>
void scaled_ranks(const RankedVector<Stat_, Index_>& collected, Output_* outgoing) { 
    static_assert(std::is_floating_point<Output_>::value);

    // Computing tied ranks. 
    size_t cur_rank = 0;
    auto cIt = collected.begin();
    auto cEnd = collected.end();

    while (cIt != cEnd) {
        auto copy = cIt;
        ++copy;
        Output_ accumulated_rank = cur_rank;
        ++cur_rank;

        while (copy != collected.end() && copy->first == cIt->first) {
            accumulated_rank += cur_rank;
            ++cur_rank;
            ++copy;
        }

        Output_ mean_rank = accumulated_rank / static_cast<Output_>(copy - cIt);
        while (cIt != copy) {
            outgoing[cIt->second] = mean_rank;
            ++cIt;
        }
    }

    // Mean-adjusting and converting to cosine values.
    Output_ sum_squares = 0;
    size_t N = collected.size();
    const Output_ center_rank = static_cast<Output_>(N - 1)/2; 
    for (size_t i = 0 ; i < N; ++i) {
        auto& o = outgoing[i];
        o -= center_rank;
        sum_squares += o*o;
    }

    // Special behaviour for no-variance cells; these are left as all-zero scaled ranks.
    sum_squares = std::max(sum_squares, 0.00000001);
    Output_ denom = std::sqrt(sum_squares) * 2;
    for (size_t i = 0; i < N; ++i) {
        outgoing[i] /= denom;
    }
}

template<typename Index_>
class RankRemapper {
private:
    std::unordered_map<Index_, Index_> my_mapping;
    Index_ my_counter = 0;

public:
    void add(Index_ i) {
        auto it = my_mapping.find(i);
        if (it == my_mapping.end()) {
            my_mapping[i] = my_counter;
            ++my_counter;
        }
    }

    void clear() {
        my_counter = 0;
        my_mapping.clear();
    }

    void reserve(size_t n) {
        my_mapping.reserve(n);
    }

public:
    template<typename Stat_>
    void remap(const RankedVector<Stat_, Index_>& input, RankedVector<Stat_, Index_>& output) const {
        output.clear();
        for (const auto& x : input) {
            auto it = my_mapping.find(x.second);
            if (it != my_mapping.end()) {
                output.emplace_back(x.first, it->second);
            }
        }
    }
};

template<typename Stat_, typename Index_, typename Simple_>
void simplify_ranks(const RankedVector<Stat_, Index_>& x, RankedVector<Simple_, Index_>& output) {
    if (x.size()) {
        Simple_ counter = 0;
        auto last = x[0].first;
        for (const auto& r : x) {
            if (r.first != last) {
                ++counter;
                last = r.first;
            }
            output.emplace_back(counter, r.second);
        }
    }
}

}

}

#endif
