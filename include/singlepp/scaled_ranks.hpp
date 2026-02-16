#ifndef SINGLEPP_SCALED_RANKS_HPP
#define SINGLEPP_SCALED_RANKS_HPP

#include "utils.hpp"

#include "sanisizer/sanisizer.hpp"

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
    if (num_markers == 0) {
        return;
    }

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
void scaled_ranks(
    const Index_ num_markers,
    const Index_ num_negative,
    const typename RankedVector<Stat_, Index_>::const_iterator negative_start,
    const typename RankedVector<Stat_, Index_>::const_iterator negative_end,
    const Index_ num_positive,
    const typename RankedVector<Stat_, Index_>::const_iterator positive_start,
    const typename RankedVector<Stat_, Index_>::const_iterator positive_end,
    SparseScaled<Index_, Float_>& outgoing
) {
    static_assert(std::is_floating_point<Float_>::value);

    assert(sanisizer::is_equal(negative_end - negative_start, num_negative));
    assert(sanisizer::is_equal(positive_end - positive_start, num_positive));
    assert(sanisizer::is_greater_than_or_equal(num_markers, sanisizer::sum<Index_>(num_positive, num_negative)));

    outgoing.nonzero.clear();
    outgoing.zero = 0;
    if (num_markers == 0) {
        return;
    }

    // Computing tied ranks: before, at, and after zero.
    Index_ cur_rank = 0;
    auto nIt = negative_start;
    while (nIt != negative_end) {
        auto copy = nIt;
        do {
            ++copy;
        } while (copy != negative_end && copy->first == nIt->first);

        const Float_ jump = copy - nIt;
        const Float_ mean_rank = cur_rank + (jump - 1) / static_cast<Float_>(2);
        while (nIt != copy) {
            outgoing.nonzero.emplace_back(nIt->second, mean_rank);
            ++nIt;
        }

        cur_rank += jump;
    }

    Index_ num_zero = num_markers - num_negative - num_positive;
    Float_ zero_rank = 0; 
    if (num_zero) {
        zero_rank = cur_rank + static_cast<Float_>(num_zero - 1) / static_cast<Float_>(2);
        cur_rank += num_zero;
    }

    auto pIt = positive_start;
    while (pIt != positive_end) {
        auto copy = pIt;
        do {
            ++copy;
        } while (copy != positive_end && copy->first == pIt->first);

        const Float_ jump = copy - pIt;
        const Float_ mean_rank = cur_rank + (jump - 1) / static_cast<Float_>(2);
        while (pIt != copy) {
            outgoing.nonzero.emplace_back(pIt->second, mean_rank);
            ++pIt;
        }

        cur_rank += jump;
    }

    // Mean-adjusting and converting to cosine values.
    Float_ sum_squares = 0;
    const Float_ center_rank = static_cast<Float_>(num_markers - 1) / static_cast<Float_>(2); 
    for (auto& nz : outgoing.nonzero) {
        auto& o = nz.second;
        o -= center_rank;
        sum_squares += o * o;
    }
    if (num_zero) {
        zero_rank -= center_rank;
        sum_squares += num_zero * zero_rank * zero_rank;
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
        outgoing.zero = zero_rank * denom;
    }
}

template<typename Stat_, typename Index_, typename Float_>
void scaled_ranks(
    const Index_ num_markers,
    const typename RankedVector<Stat_, Index_>::const_iterator negative_start,
    const typename RankedVector<Stat_, Index_>::const_iterator negative_end,
    const typename RankedVector<Stat_, Index_>::const_iterator positive_start,
    const typename RankedVector<Stat_, Index_>::const_iterator positive_end,
    SparseScaled<Index_, Float_>& outgoing
) { 
    scaled_ranks<Stat_, Index_, Float_>(
        num_markers,
        negative_end - negative_start,
        negative_start,
        negative_end,
        positive_end - positive_start,
        positive_start,
        positive_end,
        outgoing
    );
}

template<typename Stat_, typename Index_, typename Float_>
void scaled_ranks(
    const Index_ num_markers,
    const RankedVector<Stat_, Index_>& negative,
    const RankedVector<Stat_, Index_>& positive,
    SparseScaled<Index_, Float_>& outgoing
) { 
    scaled_ranks<Stat_, Index_, Float_>(
        num_markers,
        negative.size(),
        negative.begin(),
        negative.end(),
        positive.size(),
        positive.begin(),
        positive.end(),
        outgoing
    );
}

template<typename Stat_, typename Index_, typename Float_>
void scaled_ranks(
    const Index_ num_markers,
    const std::pair<const RankedVector<Stat_, Index_>*, const RankedVector<Stat_, Index_>*>& info,
    SparseScaled<Index_, Float_>& outgoing
) { 
    scaled_ranks<Stat_, Index_, Float_>(
        num_markers,
        *(info.first),
        *(info.second),
        outgoing
    );
}

template<typename Stat_, typename Index_, typename Simple_>
void simplify_ranks(
    const Index_ size,
    const typename RankedVector<Stat_, Index_>::const_iterator start,
    const typename RankedVector<Stat_, Index_>::const_iterator end,
    RankedVector<Simple_, Index_>& output
) {
    assert(sanisizer::is_equal(size, end - start));
    if (size == 0) {
        return;
    }

    output.reserve(size);

    // The general idea is that Simple_ is smaller than Stat_, to save space.
    // As long as we respect ties, everything should be fine.
    Simple_ counter = 0;
    auto last = start->first;
    for (auto it = start; it < end; ++it) {
        if (it->first != last) {
            ++counter;
            last = it->first;
        }
        output.emplace_back(counter, it->second);
    }
}

template<typename Stat_, typename Index_, typename Simple_>
void simplify_ranks(
    const typename RankedVector<Stat_, Index_>::const_iterator start,
    const typename RankedVector<Stat_, Index_>::const_iterator end,
    RankedVector<Simple_, Index_>& output
) {
    simplify_ranks<Stat_, Index_, Simple_>(end - start, start, end, output);
}

template<typename Stat_, typename Index_, typename Simple_>
void simplify_ranks(const RankedVector<Stat_, Index_>& x, RankedVector<Simple_, Index_>& output) {
    simplify_ranks<Stat_, Index_, Simple_>(x.size(), x.begin(), x.end(), output);
}

}

#endif
