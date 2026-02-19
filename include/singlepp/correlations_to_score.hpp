#ifndef SINGLEPP_CORRELATIONS_TO_SCORE_HPP
#define SINGLEPP_CORRELATIONS_TO_SCORE_HPP

#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>

#include "utils.hpp"

namespace singlepp {

template<typename Float_>
Float_ l2_to_correlation(const Float_ l2) {
    return 1 - 2 * l2;
}

template<typename Float_>
Float_ correlations_to_score(std::vector<Float_>& correlations, Float_ quantile) {
    static_assert(std::is_floating_point<Float_>::value);

    const auto num = correlations.size();
    if (num == 0) {
        return std::numeric_limits<Float_>::quiet_NaN();
    }

    if (quantile == 1 || num == 1) {
        return *std::max_element(correlations.begin(), correlations.end());
    }

    const Float_ denom = num - 1; 
    const Float_ prod = denom * quantile;
    const I<decltype(num)> lower_index = std::floor(prod);
    const I<decltype(num)> upper_index = std::ceil(prod);

    std::nth_element(correlations.begin(), correlations.begin() + upper_index, correlations.end());
    const Float_ upper_val = correlations[upper_index];
    if (upper_index == lower_index) {
        return upper_val;
    }

    // After nth_element(), all elements before 'upper_index' are now less than or equal to the value at 'upper_index'.
    // So if we want to get the value at 'lower_index', we can just find the maximum value rather than sorting again.
    const Float_ lower_val = *std::max_element(correlations.begin(), correlations.begin() + upper_index);

    // Here we compute the type 7 quantile, as done by default in R's stats::quantile() function.
    // This basically interpolates between the two observations flanking the quantile.
    const Float_ upper_prop = prod - lower_index;

    // a.k.a. upper_prop * upper_val + (1 - upper_prop) * lower_val 
    return lower_val + (upper_val - lower_val) * upper_prop;
}

}

#endif
