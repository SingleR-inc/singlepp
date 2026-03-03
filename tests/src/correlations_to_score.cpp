#include <gtest/gtest.h>

#include "singlepp/correlations_to_score.hpp"
#include "singlepp/build_reference.hpp"

#include <algorithm>
#include <vector>

#include "fill_ranks.h"

TEST(L2ToCorrelation, Basic) {
    std::vector<double> values { -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, 0 };
    auto scaled = quick_scaled_ranks(values);
    EXPECT_FLOAT_EQ(singlepp::l2_to_correlation(singlepp::dense_l2(scaled.size(), scaled.data(), scaled.data())), 1);

    auto neg = scaled;
    for (auto& x : neg) {
        x *= -1;
    }
    EXPECT_FLOAT_EQ(singlepp::l2_to_correlation(singlepp::dense_l2(scaled.size(), scaled.data(), neg.data())), -1);

    // Compare to R code:
    // > cor(c(-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, 0), 1:7, method="spearman")
    std::vector<double> values2 { 1, 2, 3, 4, 5, 6, 7 };
    auto scaled2 = quick_scaled_ranks(values2);
    EXPECT_FLOAT_EQ(singlepp::l2_to_correlation(singlepp::dense_l2(scaled.size(), scaled.data(), scaled2.data())), 0.2142857);

    // Checking the clamp.
    EXPECT_EQ(singlepp::l2_to_correlation(-0.1), 1);
    EXPECT_EQ(singlepp::l2_to_correlation(1.1), -1);
}

// Deliberately creating a copy to avoid modifying the input vector so much
// that it eventually becomes sorted.
static double l2_to_score(std::vector<double> l2, double quantile) {
    return singlepp::l2_to_score(l2, singlepp::precompute_quantile_details(l2.size(), quantile));
}

TEST(CorrelationsToScore, Basic) {
    std::vector<double> correlations { -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, 0 };
    std::vector<double> l2 = correlations;
    for (auto& l : l2) {
        l = 0.5 - l / 2;
    }

    EXPECT_FLOAT_EQ(l2_to_score(l2, 1.0), 0.6);
    EXPECT_FLOAT_EQ(l2_to_score(l2, 0.0), -0.5);

    EXPECT_FLOAT_EQ(l2_to_score(l2, 0.5), 0);
    EXPECT_FLOAT_EQ(l2_to_score(l2, 5.0/6), 0.4);
    EXPECT_FLOAT_EQ(l2_to_score(l2, 1.0/6), -0.3);

    EXPECT_FLOAT_EQ(l2_to_score(l2, 0.9), 0.48);
    EXPECT_FLOAT_EQ(l2_to_score(l2, 0.75), 0.3);
    EXPECT_FLOAT_EQ(l2_to_score(l2, 0.65), 0.18);

    // Behaves with just a single sample.
    std::vector<double> solo{ l2[0] };
    EXPECT_FLOAT_EQ(l2_to_score(solo, 0.5), correlations[0]);
}

TEST(CorrelationsToScore, Ties) {
    std::vector<double> correlations { 0.1, 0.2, 0.3, 0.1, 0.2, 0.1 };
    std::vector<double> l2 = correlations;
    for (auto& l : l2) {
        l = 0.5 - l / 2;
    }

    EXPECT_FLOAT_EQ(l2_to_score(l2, 0.0), 0.1);
    EXPECT_FLOAT_EQ(l2_to_score(l2, 0.1), 0.1);
    EXPECT_FLOAT_EQ(l2_to_score(l2, 0.3), 0.1);
    EXPECT_FLOAT_EQ(l2_to_score(l2, 0.5), 0.15);
    EXPECT_FLOAT_EQ(l2_to_score(l2, 0.6), 0.2);
    EXPECT_FLOAT_EQ(l2_to_score(l2, 0.7), 0.2);
    EXPECT_FLOAT_EQ(l2_to_score(l2, 0.9), 0.25);
    EXPECT_FLOAT_EQ(l2_to_score(l2, 1.0), 0.3);
}

TEST(CorrelationsToScore, FloatConversion) {
    // Check that we behave correctly if the conversion of 'len - 1' to a single-precision float is successful but inexact,
    // such that it rounds up, and then when we convert back, the final value exceeds 'len - 1'.
    int inaccurate = 0;
    std::vector<float> non_zero_values(1000, 123);

    for (int i = 0; i < 100; ++i) {
        std::int32_t x = i + static_cast<std::int32_t>(1e8);

        // Check that it's actually inexact.
        auto x_m1 = x - 1;
        float y = x_m1;
        std::int32_t x_back = y;
        const bool current_inexact = (x_m1 < x_back);
        inaccurate += current_inexact;

        auto qdeets = singlepp::precompute_quantile_details<std::int32_t, float>(x, 0);
        if (current_inexact) {
            EXPECT_EQ(qdeets.right_index, x_m1);
            EXPECT_FALSE(qdeets.find_left);
        }
    }

    EXPECT_GT(inaccurate, 0);
}
