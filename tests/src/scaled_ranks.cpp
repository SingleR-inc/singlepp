#include <gtest/gtest.h>

#include "singlepp/scaled_ranks.hpp"
#include "tatami/tatami.hpp"
#include "singlepp/compute_scores.hpp"

#include <algorithm>
#include <vector>

#include "fill_ranks.h"

double expected_variance(double n) {
    return 1 / (4.0 * (n - 1));
}

TEST(FillRanks, Basic) {
    std::vector<double> stuff { 0.34817868, 0.24918308, 0.75879770, 0.71893282, 0.78199329, 0.09039928 };
    auto ranks = fill_ranks(stuff.size(), stuff.data());
    double prev = 0;
    for (size_t i = 0; i < ranks.size(); ++i) {
        EXPECT_TRUE(ranks[i].first > prev);
        EXPECT_EQ(ranks[i].first, stuff[ranks[i].second]);
        prev = ranks[i].first;
    }
}

TEST(FillRanks, Subsetted) {
    std::vector<double> stuff { 2.505, 0.933, -0.109, -0.954, -1.314, 0.050, 1.297 };

    {
        std::vector<int> odds { 1, 3, 5 };
        auto ranks = fill_ranks(odds, stuff.data());
        double prev = -1000;
        for (size_t i = 0; i < ranks.size(); ++i) {
            EXPECT_TRUE(ranks[i].first > prev);
            EXPECT_EQ(ranks[i].first, stuff[odds[ranks[i].second]]);
            prev = ranks[i].first;
        }

        prev = -1000;
        ranks = fill_ranks(odds, stuff.data(), 1);
        for (size_t i = 0; i < ranks.size(); ++i) {
            EXPECT_TRUE(ranks[i].first > prev);
            EXPECT_EQ(ranks[i].first, stuff[odds[ranks[i].second] - 1]);
            prev = ranks[i].first;
        }
    }

    {
        std::vector<int> evens { 6, 4, 2, 0 };
        auto ranks = fill_ranks(evens, stuff.data());
        double prev = -1000;
        for (size_t i = 0; i < ranks.size(); ++i) {
            EXPECT_TRUE(ranks[i].first > prev);
            EXPECT_EQ(ranks[i].first, stuff[evens[ranks[i].second]]);
            prev = ranks[i].first;
        }
    }
}

TEST(ScaledRanks, Basic) {
    std::vector<double> stuff { 0.4234, -0.12, 2.784, 0.232, 5.32, 1.1129 };
    auto ranks = fill_ranks(stuff.size(), stuff.data());
    std::vector<double> out(stuff.size());
    singlepp::scaled_ranks(ranks, out.data());

    // Mean should be zero, variance should be... something.
    auto stats = tatami::stats::variances::compute_direct(out.data(), out.size());
    EXPECT_TRUE(std::abs(stats.first) < 1e-8);
    EXPECT_FLOAT_EQ(stats.second, expected_variance(stuff.size()));

    // Ranking should be preserved.
    EXPECT_EQ(
        std::min_element(stuff.begin(), stuff.end()) - stuff.begin(),
        std::min_element(out.begin(), out.end()) - out.begin()
    );
    EXPECT_EQ(
        std::max_element(stuff.begin(), stuff.end()) - stuff.begin(),
        std::max_element(out.begin(), out.end()) - out.begin()
    );
}

TEST(ScaledRanks, AllZero) {
    std::vector<double> all_zeroes(12);
    auto ranks = fill_ranks(all_zeroes.size(), all_zeroes.data());
    std::vector<double> out (all_zeroes.size());
    singlepp::scaled_ranks(ranks, out.data());
    EXPECT_EQ(out, all_zeroes);
}

TEST(ScaledRanks, Ties) {
    std::vector<double> stuff { -0.038, -0.410, 0.501, -0.174, 0.899, 0.422 };
    size_t original_size = stuff.size();

    auto ranks = fill_ranks(original_size, stuff.data());
    std::vector<double> ref(original_size);
    singlepp::scaled_ranks(ranks, ref.data());

    // Checking values aren't NA or infinite.
    auto stats = tatami::stats::variances::compute_direct(ref.data(), ref.size());
    EXPECT_TRUE(std::abs(stats.first) < 1e-8);
    EXPECT_FLOAT_EQ(stats.second, expected_variance(original_size));

    // Slapping a duplicate onto the end.
    stuff.push_back(stuff[0]);
    ranks = fill_ranks(stuff.size(), stuff.data());
    std::vector<double> tied(stuff.size());
    singlepp::scaled_ranks(ranks, tied.data());
  
    EXPECT_EQ(tied[0], tied.back()); // same rank
    EXPECT_NE(tied[0], ref[0]); // changes the ranks; note that this doesn't work if the first element is right in the middle.

    auto stats2 = tatami::stats::variances::compute_direct(tied.data(), tied.size()); // these properties still hold.
    EXPECT_TRUE(std::abs(stats2.first) < 1e-8);
    EXPECT_FLOAT_EQ(stats2.second, expected_variance(tied.size()));

    // Full duplication.
    for (size_t s = 1; s < original_size; ++s) {
        stuff.push_back(stuff[s]);
    }
    ASSERT_EQ(stuff.size(), original_size * 2);
    ranks = fill_ranks(stuff.size(), stuff.data());
    std::vector<double> dupped(stuff.size());
    singlepp::scaled_ranks(ranks, dupped.data());

    auto stats3 = tatami::stats::variances::compute_direct(dupped.data(), dupped.size()); 
    EXPECT_TRUE(std::abs(stats3.first) < 1e-8);
    EXPECT_FLOAT_EQ(stats3.second, expected_variance(original_size * 2));

    std::vector<double> first_half(dupped.begin(), dupped.begin() + original_size);
    std::vector<double> second_half(dupped.begin() + original_size, dupped.end());
    EXPECT_EQ(first_half, second_half);

    for (size_t s = 0; s < original_size; ++s) {
        EXPECT_FLOAT_EQ(first_half[s] * std::sqrt(2.0), ref[s]);
    }
}

TEST(ScaledRanks, MissingValues) {
    std::vector<double> stuff { 0.6434, -0.211, 9.251, -3.352, 8.372, 1.644 };

    auto ranks = fill_ranks(stuff.size(), stuff.data());
    std::vector<double> ref(stuff.size());
    singlepp::scaled_ranks(ranks, ref.data());

    auto missing = stuff;
    missing.insert(missing.begin(), std::numeric_limits<double>::quiet_NaN());
    missing.insert(missing.begin() + 3, std::numeric_limits<double>::quiet_NaN());
    missing.push_back(std::numeric_limits<double>::quiet_NaN());

    ranks = fill_ranks<true>(missing.size(), missing.data());
    std::vector<double> out(missing.size());
    singlepp::scaled_ranks<true>(missing.size(), ranks, out.data());

    // NaN's propagate through.
    EXPECT_TRUE(std::isnan(out[0]));
    EXPECT_TRUE(std::isnan(out[3]));
    EXPECT_TRUE(std::isnan(out.back()));

    // Everything else is not NaN.
    auto cleansed = out;
    cleansed.erase(cleansed.begin() + cleansed.size() - 1);
    cleansed.erase(cleansed.begin() + 3);
    cleansed.erase(cleansed.begin());

    EXPECT_EQ(cleansed, ref);
}

TEST(ScaledRanks, Subset) {
    std::vector<double> stuff { 0.358, 0.496, 0.125, 0.408, 0.618, 0.264, 0.905, 0.895, 0.264, 0.865, 0.069, 0.581 };
    std::vector<int> sub { 2, 7, 0, 3, 5, 10 };

    auto ranks = fill_ranks(sub, stuff.data());
    std::vector<double> out(sub.size());
    singlepp::scaled_ranks(ranks, out.data());

    // Reference comparison.
    std::vector<double> stuff2;
    for (auto s : sub) {
        stuff2.push_back(stuff[s]);
    }

    ranks = fill_ranks(stuff2.size(), stuff2.data());
    std::vector<double> out2(sub.size());
    singlepp::scaled_ranks(ranks, out2.data());
    EXPECT_EQ(out, out2);

    // Works with missing values in there.
    stuff[sub[0]] = std::numeric_limits<double>::quiet_NaN();
    ranks = fill_ranks<true>(sub, stuff.data());
    EXPECT_EQ(ranks.size(), sub.size() - 1);
    singlepp::scaled_ranks<true>(sub.size(), ranks, out.data());

    stuff2[0] = std::numeric_limits<double>::quiet_NaN();
    ranks = fill_ranks<true>(stuff2.size(), stuff2.data());
    EXPECT_EQ(ranks.size(), stuff2.size() - 1);
    singlepp::scaled_ranks<true>(stuff2.size(), ranks, out2.data());

    EXPECT_TRUE(std::isnan(out[0]));
    EXPECT_TRUE(std::isnan(out2[0]));
    out.erase(out.begin());
    out2.erase(out2.begin());
    EXPECT_EQ(out, out2);
}

TEST(ScaledRanks, CorrelationCheck) {
    std::vector<double> left { 0.5581, 0.1208, 0.1635, 0.8309, 0.3698, 0.7121, 0.3960, 0.7862, 0.8256, 0.1057 };
    std::vector<double> right { -0.4698, -1.0779, -0.2542,  0.1184, -2.0408,  1.4954,  1.1195, -1.0523,  0.4349,  1.6694 };
    ASSERT_EQ(left.size(), right.size());

    auto ranks = fill_ranks(left.size(), left.data());
    std::vector<double> out1(left.size());
    singlepp::scaled_ranks(ranks, out1.data());

    ranks = fill_ranks(right.size(), right.data());
    std::vector<double> out2(right.size());
    singlepp::scaled_ranks(ranks, out2.data());

    double obs = singlepp::distance_to_correlation(out1.size(), out1.data(), out2.data());
    
    // Manual calculation.
    {
        singlepp::RankedVector<double, int> ranks1(left.size());
        singlepp::RankedVector<double, int> ranks2(right.size());

        for (int it = 0; it < 2; ++it) {
            const auto& src = (it == 0 ? left : right);
            auto& ranked = (it == 0 ? ranks1 : ranks2);

            for (size_t s = 0; s < src.size(); ++s) {
                ranked[s].first = src[s];
                ranked[s].second = s;
            }
            std::sort(ranked.begin(), ranked.end());

            for (size_t s = 0; s < src.size(); ++s) {
                ranked[s].first = s + 1;
            }
            std::sort(ranked.begin(), ranked.end(), [](const auto& left, const auto& right) -> bool { return left.second < right.second; });
        }

        double delta_rank = 0;
        for (size_t l = 0; l < left.size(); ++l) {
            double tmp = ranks1[l].first - ranks2[l].first;
            delta_rank += tmp*tmp;
        }

        double spearman = 1 - 6 * delta_rank / (left.size() * (left.size() * left.size() - 1));
        EXPECT_FLOAT_EQ(spearman, obs);
    }
}

