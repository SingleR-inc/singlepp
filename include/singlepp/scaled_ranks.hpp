#ifndef SINGLEPP_SCALED_RANKS_HPP
#define SINGLEPP_SCALED_RANKS_HPP

#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>

namespace singlepp {

template<typename Stat, typename Index>
using RankVector = std::vector<std::pair<Stat, Index> >;

template<bool na_aware = false, typename Stat, typename Index>
void scaled_ranks(size_t slen, const RankVector<Stat, Index>& collected, double* outgoing) { 
    // Computing tied ranks. 
    size_t cur_rank=0;
    auto cIt=collected.begin();
    if constexpr(na_aware) {
        std::fill(outgoing, outgoing + slen, std::numeric_limits<double>::quiet_NaN());
    }

    while (cIt!=collected.end()) {
        auto copy=cIt;
        ++copy;
        double accumulated_rank=cur_rank;
        ++cur_rank;

        while (copy!=collected.end() && copy->first==cIt->first) {
            accumulated_rank+=cur_rank;
            ++cur_rank;
            ++copy;
        }

        double mean_rank=accumulated_rank/(copy-cIt);
        while (cIt!=copy) {
            outgoing[cIt->second]=mean_rank;
            ++cIt;
        }
    }

    // Mean-adjusting and converting to cosine values.
    double sum_squares=0;
    const double center_rank=static_cast<double>(collected.size() - 1)/2; // use collected.size(), not slen, to account for NA's.
    for (size_t i = 0 ; i < slen; ++i) {
        auto& o = outgoing[i];
        if constexpr(na_aware) {
            if (std::isnan(o)) {
                continue;
            }
        }
        o-=center_rank;
        sum_squares+=o*o;
    }

    // Special behaviour for no-variance cells; these are left as all-zero scaled ranks.
    sum_squares = std::max(sum_squares, 0.00000001);
    sum_squares = std::sqrt(sum_squares)*2;
    for (size_t i = 0; i < slen; ++i) {
        auto& o = outgoing[i];
        if constexpr(na_aware) {
            if (std::isnan(o)) {
                continue;
            }
        }
        o /= sum_squares;
    }

    return;
}

template<bool na_aware = false, typename Stat, typename Index>
void scaled_ranks(RankVector<Stat, Index>& collected, double* outgoing) {
    scaled_ranks(collected.size(), collected, outgoing);
    return;
}

//template<bool na_aware = false, class Start, typename Stat, typename Index>
//void scaled_ranks(size_t slen, Start start, RankedVector<Stat, Index>& collected, double* outgoing) {
//    collected.clear();
//    collected.reserve(slen);
//
//    for (size_t s = 0; s < slen; ++s) {
//        const double curval=*(start + s);
//        if (std::isnan(curval)) { 
//            if constexpr(!na_aware) {
//                throw std::runtime_error("missing values not supported in SingleR");
//            }
//        } else {
//            collected.emplace_back(curval, s);
//        }
//    }
//
//    scaled_ranks<na_aware>(slen, collected, outgoing);
//    return;
//}
//
//template<bool na_aware = false, class Start, typename Stat, typename Index>
//void scaled_ranks(Start start, const std::vector<int>& chosen, RankedVector<Stat, Index>& collected, double* outgoing) {
//    size_t slen=chosen.size();
//    collected.clear();
//    collected.reserve(slen);
//
//    size_t s=0;
//    for (auto i : chosen) {
//        const double curval=*(start + i);
//        if (std::isnan(curval)) { 
//            if constexpr(!na_aware) {
//                throw std::runtime_error("missing values not supported in SingleR");
//            }
//        } else {
//            collected.emplace_back(curval, s);
//        }
//        ++s;
//    }
//
//    scaled_ranks<na_aware>(slen, collected, outgoing);
//    return;
//}

}

#endif
