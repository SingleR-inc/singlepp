#include <gtest/gtest.h>

#include "singlepp/correlations_to_score.hpp"
#include "singlepp/build_reference.hpp"

#include <algorithm>
#include <vector>

#include "fill_ranks.h"

// Deliberately creating a copy to avoid modifying the input vector so much
// that it eventually becomes sorted.
static double correlations_to_score(std::vector<double> correlations, double quantile) {
    return singlepp::correlations_to_score(correlations, quantile);
}

TEST(CorrelationsToScore, Basic) {
    std::vector<double> correlations { -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, 0 };

    EXPECT_EQ(correlations_to_score(correlations, 1.0), 0.6);
    EXPECT_EQ(correlations_to_score(correlations, 0.0), -0.5);

    EXPECT_EQ(correlations_to_score(correlations, 0.5), 0);
    EXPECT_FLOAT_EQ(correlations_to_score(correlations, 5.0/6), 0.4);
    EXPECT_FLOAT_EQ(correlations_to_score(correlations, 1.0/6), -0.3);

    EXPECT_FLOAT_EQ(correlations_to_score(correlations, 0.9), 0.48);
    EXPECT_FLOAT_EQ(correlations_to_score(correlations, 0.75), 0.3);
    EXPECT_FLOAT_EQ(correlations_to_score(correlations, 0.65), 0.18);

    std::vector<double> empty;
    EXPECT_TRUE(std::isnan(correlations_to_score(empty, 0.0)));

    std::vector<double> solo{10};
    EXPECT_EQ(correlations_to_score(solo, 0.5), 10);
}

TEST(CorrelationsToScore, Ties) {
    std::vector<double> correlations { 0.1, 0.2, 0.3, 0.1, 0.2, 0.1 };
    EXPECT_FLOAT_EQ(correlations_to_score(correlations, 0.0), 0.1);
    EXPECT_FLOAT_EQ(correlations_to_score(correlations, 0.1), 0.1);
    EXPECT_FLOAT_EQ(correlations_to_score(correlations, 0.3), 0.1);
    EXPECT_FLOAT_EQ(correlations_to_score(correlations, 0.5), 0.15);
    EXPECT_FLOAT_EQ(correlations_to_score(correlations, 0.6), 0.2);
    EXPECT_FLOAT_EQ(correlations_to_score(correlations, 0.7), 0.2);
    EXPECT_FLOAT_EQ(correlations_to_score(correlations, 0.9), 0.25);
    EXPECT_FLOAT_EQ(correlations_to_score(correlations, 1.0), 0.3);
}

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
}
