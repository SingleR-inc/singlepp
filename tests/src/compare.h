#ifndef COMPARE_H
#define COMPARE_H

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

inline void check_almost_equal_assignment(int expected_best, double expected_delta, int observed_best, double observed_delta) {
    constexpr double tol = 1e-8;
    EXPECT_LT(std::abs(expected_delta - observed_delta), tol);

    // Due to differences in numerical precision between dense/sparse calculations, comparisons may not be exact.
    // This results in different 'best' labels in the presence of near-ties, so if there's a mismatch,
    // we check that the delta is indeed near-zero, i.e., there is a near-tie. 
    EXPECT_GT((expected_best == observed_best) + (expected_delta < tol), 0);

    // That said, it is still possible for this check to fail in very unfortunate circumstances.
    // Specifically, when the delta with another label is very close with the fine-tuning threshold, resulting in differences in the labels chosen for the next fine-tuning iteration.
    // We don't have an easy way to check for this, as it could happen in the middle of fine-tuning where we can't see any of the internal scores.
    // So... we'll just hope it doesn't happen; it seems pretty rare, and we'll just fiddle with the seeds to get out of it if it does occur.
}

inline void check_almost_equal_vectors(const std::vector<double>& expected, const std::vector<double>& observed) {
    constexpr double tol = 1e-8;
    const auto n = expected.size(); 
    ASSERT_EQ(n, observed.size());
    for (std::size_t t = 0; t < n; ++t) {
        EXPECT_LT(std::abs(expected[t] - observed[t]), tol); 
    }
}

#endif
