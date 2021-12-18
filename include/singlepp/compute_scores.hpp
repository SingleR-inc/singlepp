#ifndef SINGLEPP_COMPUTE_SCORES_HPP
#define SINGLEPP_COMPUTE_SCORES_HPP

#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>

namespace singlepp {

inline double correlations_to_scores (std::vector<double>& correlations, double quantile) {
    const size_t ncells=correlations.size();
    if (ncells==0) {
        return std::numeric_limits<double>::quiet_NaN();
    } else if (quantile==0 || ncells==1) {
        return *std::max_element(correlations.begin(), correlations.end());
    } else {
        auto rquantile = 1 - quantile; 
        const double denom=ncells-1;
        const size_t qn=std::floor(denom * rquantile) + 1;

        // Technically, I should do (qn-1)+1, with the first -1 being to get zero-indexed values
        // and the second +1 to obtain the ceiling. But they cancel out, so I won't.
        std::nth_element(correlations.begin(), correlations.begin()+qn, correlations.end());
        const double rightval=correlations[qn];

        // Do NOT be tempted to do the second nth_element with the end at begin()+qn;
        // this does not handle ties properly.
        std::nth_element(correlations.begin(), correlations.begin()+qn-1, correlations.end());
        const double leftval=correlations[qn-1];

        const double rightweight=rquantile - ((qn-1)/denom);
        const double leftweight=(qn/denom) - rquantile;
        return (rightval * rightweight + leftval * leftweight)/(rightweight + leftweight);
    }
}

template<class Vector>
double distance_to_correlation(size_t n, const Vector& p1, const Vector& p2) {
    double d2 = 0;
    for (size_t i = 0; i < n; ++i) {
        double tmp = p1[i] - p2[i];
        d2 += tmp * tmp;
    }
    return 1 - 2 * d2;
}

}

#endif
