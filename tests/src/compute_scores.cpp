#include <gtest/gtest.h>

#include "singlepp/compute_scores.hpp"

#include <algorithm>
#include <vector>

TEST(CorrelationsToScores, Basic) {
    std::vector<double> correlations { -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, 0 };
    EXPECT_EQ(singlepp::correlations_to_scores(correlations, 1), 0.6);
    EXPECT_EQ(singlepp::correlations_to_scores(correlations, 0), -0.5);

    EXPECT_EQ(singlepp::correlations_to_scores(correlations, 0.5), 0);
    EXPECT_FLOAT_EQ(singlepp::correlations_to_scores(correlations, 5.0/6), 0.4);
    EXPECT_FLOAT_EQ(singlepp::correlations_to_scores(correlations, 1.0/6), -0.3);

    EXPECT_FLOAT_EQ(singlepp::correlations_to_scores(correlations, 0.9), 0.48);
    EXPECT_FLOAT_EQ(singlepp::correlations_to_scores(correlations, 0.75), 0.3);
    EXPECT_FLOAT_EQ(singlepp::correlations_to_scores(correlations, 0.65), 0.18);

    std::vector<double> empty;
    EXPECT_TRUE(std::isnan(singlepp::correlations_to_scores(empty, 0)));
}
