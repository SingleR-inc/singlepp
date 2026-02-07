#ifndef SINGLEPP_SCALED_RANKS_HPP
#define SINGLEPP_SCALED_RANKS_HPP

#include "utils.hpp"

#include <algorithm>
#include <vector>
#include <cmath>
#include <type_traits>
#include <cassert>

namespace singlepp {

template<typename Stat_, typename Index_>
using RankedVector = std::vector<std::pair<Stat_, Index_> >;

template<typename Stat_, typename Index_, typename Output_>
void scaled_ranks(const Index_ num_markers, const RankedVector<Stat_, Index_>& collected, Output_* outgoing) { 
    static_assert(std::is_floating_point<Output_>::value);
    assert(sanisizer::is_equal(num_markers, collected.size()));

    // Computing tied ranks. 
    Index_ cur_rank = 0;
    auto cIt = collected.begin();
    auto cEnd = collected.end();

    while (cIt != cEnd) {
        auto copy = cIt;
        do {
            ++copy;
        } while (copy != cEnd && copy->first == cIt->first);

        const Output_ jump = copy - cIt;
        const Output_ mean_rank = cur_rank + (jump - 1) / static_cast<Output_>(2);
        while (cIt != copy) {
            outgoing[cIt->second] = mean_rank;
            ++cIt;
        }

        cur_rank += jump;
    }

    // Mean-adjusting and converting to cosine values.
    Output_ sum_squares = 0;
    const Output_ center_rank = static_cast<Output_>(num_markers - 1) / static_cast<Output_>(2); 
    for (Index_ i = 0 ; i < num_markers; ++i) {
        auto& o = outgoing[i];
        o -= center_rank;
        sum_squares += o * o;
    }

    // Special behaviour for no-variance cells; these are left as all-zero scaled ranks.
    if (sum_squares == 0) {
        std::fill_n(outgoing, num_markers, 0);
    } else {
        const Output_ denom = 0.5 / std::sqrt(sum_squares);
        for (Index_ i = 0; i < num_markers; ++i) {
            outgoing[i] *= denom;
        }
    }
}

template<typename Stat_, typename Index_, typename Output_>
void scaled_ranks(const Index_ num_markers, const RankedVector<Stat_, Index_>& collected, std::vector<Output_>& outgoing) { 
    assert(sanisizer::is_equal(num_markers, outgoing.size()));
    scaled_ranks(num_markers, collected, outgoing.data());
}

template<typename Index_, typename Float_>
struct SparseScaled {
    std::vector<std::pair<Index_, Float_> > nonzero;
    Float_ zero = 0;
};

template<typename Stat_, typename Index_, typename Float_>
void scaled_ranks(const Index_ num_markers, const RankedVector<Stat_, Index_>& collected, SparseScaled<Index_, Float_>& outgoing) { 
    static_assert(std::is_floating_point<Float_>::value);
    assert(sanisizer::is_greater_than_or_equal(num_markers, collected.size()));

    // Computing tied ranks: before, at, and after zero.
    const Index_ ncollected = collected.size();
    Index_ cur_rank = 0;
    auto cIt = collected.begin();
    auto cEnd = collected.end();
    outgoing.nonzero.clear();

    while (cIt != cEnd && cIt->first < 0) {
        auto copy = cIt;
        do {
            ++copy;
        } while (copy != cEnd && copy->first == cIt->first);

        const Float_ jump = copy - cIt;
        const Float_ mean_rank = cur_rank + (jump - 1) / static_cast<Float_>(2);
        while (cIt != copy) {
            outgoing.nonzero.emplace_back(cIt->second, mean_rank);
            ++cIt;
        }

        cur_rank += jump;
    }

    const Index_ num_zero = num_markers - ncollected;
    Float_ zero_rank = 0; 
    if (num_zero) {
        zero_rank = cur_rank + static_cast<Float_>(num_zero - 1) / static_cast<Float_>(2);
        cur_rank += num_zero;
    }

    while (cIt != cEnd) {
        auto copy = cIt;
        do {
            ++copy;
        } while (copy != cEnd && copy->first == cIt->first);

        const Float_ jump = copy - cIt;
        const Float_ mean_rank = cur_rank + (jump - 1) / static_cast<Float_>(2);
        while (cIt != copy) {
            outgoing.nonzero.emplace_back(cIt->second, mean_rank);
            ++cIt;
        }

        cur_rank += jump;
    }

    // Mean-adjusting and converting to cosine values.
    Float_ sum_squares = 0;
    const Float_ center_rank = static_cast<Float_>(ncollected - 1) / static_cast<Float_>(2); 
    for (auto& nz : outgoing.nonzero) {
        auto& o = nz.second;
        o -= center_rank;
        sum_squares += o * o;
    }

    // Special behaviour for no-variance cells; these are left as all-zero scaled ranks.
    if (sum_squares == 0) {
        outgoing.nonzero.clear();
        outgoing.zero = 0;
    } else {
        const Float_ denom = 0.5 / std::sqrt(sum_squares);
        for (auto& nz : outgoing.nonzero) {
            nz.second *= denom;
        }
        std::sort(outgoing.nonzero.begin(), outgoing.nonzero.end());
        outgoing.zero = zero_rank * denom;
    }
}

template<typename Stat_, typename Index_, typename Simple_>
void simplify_ranks(const RankedVector<Stat_, Index_>& x, RankedVector<Simple_, Index_>& output) {
    if (x.empty()) {
        return; 
    }

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

#endif
