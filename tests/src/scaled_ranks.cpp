#include <gtest/gtest.h>

#include "singlepp/scaled_ranks.hpp"
#include "tatami_stats/tatami_stats.hpp"
#include "singlepp/compute_scores.hpp"

#include <algorithm>
#include <vector>

#include "fill_ranks.h"

double expected_variance(double n) {
    return 1 / (4.0 * (n - 1));
}

TEST(SubsetSorter, Basic) {
    std::vector<double> stuff { 0.34817868, 0.24918308, 0.75879770, 0.71893282, 0.78199329, 0.09039928 };

    // No-op.
    {
        std::vector<int> foo{ 2, 6, 18, 23, 53, 99 };
        singlepp::SubsetSorter ss(foo);
        EXPECT_FALSE(ss.use_sorted_subset);
        EXPECT_EQ(ss.extraction_subset(), foo);
        EXPECT_EQ(&ss.extraction_subset(), &foo); // exact some object, in fact.

        singlepp::RankedVector<double, int> vec(foo.size());
        ss.fill_ranks(stuff.data(), vec);

        double prev = -100000;
        for (size_t i = 0; i < vec.size(); ++i) {
            EXPECT_TRUE(vec[i].first > prev);
            EXPECT_EQ(vec[i].first, stuff[vec[i].second]);
            prev = vec[i].first;
        }
    }

    // Resorting.
    {
        std::vector<int> foo{ 5, 2, 29, 12, 23, 0 };
        singlepp::SubsetSorter ss(foo);
        EXPECT_TRUE(ss.use_sorted_subset);

        auto foocopy = foo;
        std::sort(foocopy.begin(), foocopy.end());
        EXPECT_EQ(ss.extraction_subset(), foocopy);
        
        // Here, stuff corresponds to _sorted_ foo, i.e., foocopy,
        // as we're extracting based on the sorted subsets.
        singlepp::RankedVector<double, int> vec(foocopy.size());
        ss.fill_ranks(stuff.data(), vec); 

        // Check that we get the same results as if we had done a full column
        // extraction and then extracted the subset from the array.
        std::vector<double> expanded(*std::max_element(foo.begin(), foo.end()) + 1);
        for (size_t s = 0; s < foocopy.size(); ++s) {
            expanded[foocopy[s]] = stuff[s];
        }

        for (size_t i = 0; i < vec.size(); ++i) {
            auto s = foo[vec[i].second]; // check it was correctly reindexed back to foo.
            EXPECT_EQ(vec[i].first, expanded[s]);
        }
    }

    // Deduplicating.
    {
        std::vector<int> foo{ 1, 2, 1, 5, 2, 9, 9, 4, 1, 0 };
        singlepp::SubsetSorter ss(foo);
        EXPECT_TRUE(ss.use_sorted_subset);

        std::vector<int> foocopy{ 0, 1, 2, 4, 5, 9 };
        EXPECT_EQ(ss.extraction_subset(), foocopy);
        
        // Here, stuff corresponds to _deduplicated_ foo, i.e., foocopy,
        // as we're extracting based on the deduplicated + sorted subsets.
        singlepp::RankedVector<double, int> vec(foo.size());
        ss.fill_ranks(stuff.data(), vec); 

        // Check that we get the same results as if we had done a full column
        // extraction and then extracted the subset from the array.
        std::vector<double> expanded(*std::max_element(foo.begin(), foo.end()) + 1);
        for (size_t s = 0; s < foocopy.size(); ++s) {
            expanded[foocopy[s]] = stuff[s];
        }

        for (size_t i = 0; i < vec.size(); ++i) {
            auto s = foo[vec[i].second]; // check it was correctly reindexed back to foo.
            EXPECT_EQ(vec[i].first, expanded[s]);
        }
    }
}

TEST(ScaledRanks, Basic) {
    std::vector<double> stuff { 0.4234, -0.12, 2.784, 0.232, 5.32, 1.1129 };
    auto ranks = fill_ranks(stuff.size(), stuff.data());
    std::vector<double> out(stuff.size());
    singlepp::scaled_ranks(ranks, out.data());

    // Mean should be zero, variance should be... something.
    auto stats = tatami_stats::variances::direct(out.data(), out.size(), false);
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

TEST(ScaledRanks, NoVariance) {
    std::vector<double> all_zeroes(12);

    {
        auto ranks = fill_ranks(all_zeroes.size(), all_zeroes.data());
        std::vector<double> out (all_zeroes.size());
        singlepp::scaled_ranks(ranks, out.data());
        EXPECT_EQ(out, all_zeroes);
    }

    {
        std::vector<double> all_ones(12, 1);
        auto ranks = fill_ranks(all_ones.size(), all_ones.data());
        std::vector<double> out (all_ones.size());
        singlepp::scaled_ranks(ranks, out.data());
        EXPECT_EQ(out, all_zeroes); // centered to zero, but not scaled.
    }
}

TEST(ScaledRanks, Ties) {
    std::vector<double> stuff { -0.038, -0.410, 0.501, -0.174, 0.899, 0.422 };
    size_t original_size = stuff.size();

    auto ranks = fill_ranks(original_size, stuff.data());
    std::vector<double> ref(original_size);
    singlepp::scaled_ranks(ranks, ref.data());

    // Checking values aren't NA or infinite.
    auto stats = tatami_stats::variances::direct(ref.data(), ref.size(), false);
    EXPECT_TRUE(std::abs(stats.first) < 1e-8);
    EXPECT_FLOAT_EQ(stats.second, expected_variance(original_size));

    // Slapping a duplicate onto the end.
    stuff.push_back(stuff[0]);
    ranks = fill_ranks(stuff.size(), stuff.data());
    std::vector<double> tied(stuff.size());
    singlepp::scaled_ranks(ranks, tied.data());
  
    EXPECT_EQ(tied[0], tied.back()); // same rank
    EXPECT_NE(tied[0], ref[0]); // changes the ranks; note that this doesn't work if the first element is right in the middle.

    auto stats2 = tatami_stats::variances::direct(tied.data(), tied.size(), false); // these properties still hold.
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

    auto stats3 = tatami_stats::variances::direct(dupped.data(), dupped.size(), false); 
    EXPECT_TRUE(std::abs(stats3.first) < 1e-8);
    EXPECT_FLOAT_EQ(stats3.second, expected_variance(original_size * 2));

    std::vector<double> first_half(dupped.begin(), dupped.begin() + original_size);
    std::vector<double> second_half(dupped.begin() + original_size, dupped.end());
    EXPECT_EQ(first_half, second_half);

    for (size_t s = 0; s < original_size; ++s) {
        EXPECT_FLOAT_EQ(first_half[s] * std::sqrt(2.0), ref[s]);
    }
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

TEST(SimplifyRanks, NoTies) {
    std::vector<double> no_ties { 0.72, 0.56, 0.12, 0.55, 0.50, 0.10, 0.43, 0.54, 0.18 };
    auto ranks = fill_ranks(no_ties.size(), no_ties.data());

    singlepp::RankedVector<int, int> compacted;
    singlepp::simplify_ranks(ranks, compacted);

    for (size_t i = 0; i < compacted.size(); ++i) {
        EXPECT_EQ(ranks[i].second, compacted[i].second);
        EXPECT_EQ(i, compacted[i].first);
    }
}

TEST(SimplifyRanks, WithTies) {
    std::vector<double> with_ties { 0.72, 0.56, 0.72, 0.55, 0.55, 0.10, 0.43, 0.10, 0.72 };
    auto ranks2 = fill_ranks(with_ties.size(), with_ties.data());

    singlepp::RankedVector<int, int> compacted2;
    singlepp::simplify_ranks(ranks2, compacted2);
    for (size_t i = 1; i < compacted2.size(); ++i) {
        EXPECT_TRUE(compacted2[i].first >= compacted2[i-1].first);
    }

    // All tie groups should have the same value.
    std::unordered_map<double, int> by_value;
    for (size_t i = 0; i < ranks2.size(); ++i) {
        EXPECT_EQ(ranks2[i].second, compacted2[i].second);
        auto it = by_value.find(ranks2[i].first);
        if (it != by_value.end()) { 
            EXPECT_EQ(it->second, compacted2[i].first);
        } else {
            by_value[ranks2[i].first] = compacted2[i].first;
        }
    }

    EXPECT_EQ(compacted2.front().first, 0);
    EXPECT_EQ(compacted2.back().first, by_value.size() - 1); 
}
