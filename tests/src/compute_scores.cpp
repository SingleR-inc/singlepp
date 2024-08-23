#include <gtest/gtest.h>

#include "singlepp/compute_scores.hpp"

#include <algorithm>
#include <vector>

#include "fill_ranks.h"

// Deliberately creating a copy to avoid modifying the input vector so much
// that it eventually becomes sorted.
static double correlations_to_scores (std::vector<double> correlations, double quantile) {
    return singlepp::internal::correlations_to_scores(correlations, quantile);
}

TEST(CorrelationsToScores, Basic) {
    std::vector<double> correlations { -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, 0 };

    EXPECT_EQ(correlations_to_scores(correlations, 1.0), 0.6);
    EXPECT_EQ(correlations_to_scores(correlations, 0.0), -0.5);

    EXPECT_EQ(correlations_to_scores(correlations, 0.5), 0);
    EXPECT_FLOAT_EQ(correlations_to_scores(correlations, 5.0/6), 0.4);
    EXPECT_FLOAT_EQ(correlations_to_scores(correlations, 1.0/6), -0.3);

    EXPECT_FLOAT_EQ(correlations_to_scores(correlations, 0.9), 0.48);
    EXPECT_FLOAT_EQ(correlations_to_scores(correlations, 0.75), 0.3);
    EXPECT_FLOAT_EQ(correlations_to_scores(correlations, 0.65), 0.18);

    std::vector<double> empty;
    EXPECT_TRUE(std::isnan(correlations_to_scores(empty, 0.0)));
}

TEST(CorrelationsToScores, Ties) {
    std::vector<double> correlations { 0.1, 0.2, 0.3, 0.1, 0.2, 0.1 };
    EXPECT_FLOAT_EQ(correlations_to_scores(correlations, 0.0), 0.1);
    EXPECT_FLOAT_EQ(correlations_to_scores(correlations, 0.1), 0.1);
    EXPECT_FLOAT_EQ(correlations_to_scores(correlations, 0.3), 0.1);
    EXPECT_FLOAT_EQ(correlations_to_scores(correlations, 0.5), 0.15);
    EXPECT_FLOAT_EQ(correlations_to_scores(correlations, 0.6), 0.2);
    EXPECT_FLOAT_EQ(correlations_to_scores(correlations, 0.7), 0.2);
    EXPECT_FLOAT_EQ(correlations_to_scores(correlations, 0.9), 0.25);
    EXPECT_FLOAT_EQ(correlations_to_scores(correlations, 1.0), 0.3);
}

TEST(DistanceToCorrelation, Basic) {
    std::vector<double> values { -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, 0 };
    auto scaled = quick_scaled_ranks(values);
    EXPECT_FLOAT_EQ(singlepp::internal::distance_to_correlation<double>(scaled, scaled), 1);

    auto neg = scaled;
    for (auto& x : neg) {
        x *= -1;
    }
    EXPECT_FLOAT_EQ(singlepp::internal::distance_to_correlation<double>(scaled, neg), -1);

    // Compare to R code:
    // > cor(c(-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, 0), 1:7, method="spearman")
    std::vector<double> values2 { 1, 2, 3, 4, 5, 6, 7 };
    auto scaled2 = quick_scaled_ranks(values2);
    EXPECT_FLOAT_EQ(singlepp::internal::distance_to_correlation<double>(scaled, scaled2), 0.2142857);
}
