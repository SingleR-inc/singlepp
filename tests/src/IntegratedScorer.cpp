#include <gtest/gtest.h>

#include "singlepp/IntegratedScorer.hpp"
#include "spawn_matrix.h"
#include "mock_markers.h"
#include "naive_method.h"

class IntegratedScorerTest : public ::testing::TestWithParam<std::tuple<int, double> > {};

TEST_P(IntegratedScorerTest, Basic) {
    // Mocking up the test and references.
    size_t ntest = 20;
    size_t ngenes = 2000;
    auto test = spawn_matrix(ngenes, ntest, 123456790);

    size_t nsamples = 50;
    size_t nrefs = 3;
    auto param = GetParam();
    int ntop = std::get<0>(param);
    double quantile = std::get<1>(param);

    singlepp::SinglePP runner;
    runner.set_top(ntop);
    singlepp::IntegratedBuilder builder;

    std::vector<std::shared_ptr<tatami::Matrix<double, int> > > matrices;
    std::vector<std::vector<int> > labels;
    std::vector<singlepp::SinglePP::Prebuilt> prebuilts;

    for (size_t r = 0; r < nrefs; ++r) {
        size_t seed = r * 321;
        size_t nlabels = 3 + r;

        matrices.push_back(spawn_matrix(ngenes, nsamples, seed));
        labels.push_back(spawn_labels(nsamples, nlabels, seed * 2));

        auto markers = mock_markers(nlabels, 50, ngenes, seed * 3);
        auto pre = runner.build(matrices.back().get(), labels.back().data(), std::move(markers));
        prebuilts.push_back(std::move(pre));

        builder.add(matrices.back().get(), labels.back().data(), prebuilts.back());
    }

    auto integrated = builder.finish();

    // Mocking up some of the best choices.
    std::vector<std::vector<int> > chosen(nrefs);
    std::vector<const int*> chosen_ptrs(nrefs);

    std::mt19937_64 rng(13579);
    for (size_t r = 0; r < nrefs; ++r) {
        size_t nlabels = prebuilts[r].markers.size();
        for (size_t t = 0; t < ntest; ++t) {
            chosen[r].push_back(rng() % nlabels);
        }
        chosen_ptrs[r] = chosen[r].data();
    }

    // Comparing the IntegratedScorer to a reference calculation.
    singlepp::IntegratedScorer scorer;
    scorer.set_quantile(quantile);
    auto output = scorer.run(test.get(), chosen_ptrs, integrated);

    std::vector<std::vector<std::vector<int> > > by_labels(nrefs);
    for (size_t r = 0; r < nrefs; ++r) {
        size_t nlabels = prebuilts[r].markers.size();
        by_labels[r] = split_by_label(nlabels, matrices[r]->ncol(), labels[r]);
    }

    for (size_t t = 0; t < ntest; ++t) {
        std::unordered_set<int> my_universe;
        for (size_t r = 0; r < nrefs; ++r) {
            const auto& pre = prebuilts[r];
            const auto& best_markers = pre.markers[chosen[r][t]];
            for (const auto& x : best_markers) {
                for (auto y : x) {
                    my_universe.insert(pre.subset[y]);
                }
            }
        }

        std::vector<int> universe(my_universe.begin(), my_universe.end());
        std::sort(universe.begin(), universe.end());

        auto col = test->column(t);
        auto scaled = quick_scaled_ranks(col, universe);
        std::vector<double> all_scores;
        for (size_t r = 0; r < nrefs; ++r) {
            double score = naive_score(scaled, by_labels[r][chosen[r][t]], matrices[r].get(), universe, quantile);
            EXPECT_FLOAT_EQ(score, output.scores[r][t]);
            all_scores.push_back(score);
        }

        auto best = std::max_element(all_scores.begin(), all_scores.end());
        EXPECT_EQ(output.best[t], best - all_scores.begin());

        double best_score = *best;
        *best = -100000;
        EXPECT_FLOAT_EQ(output.delta[t], best_score - *std::max_element(all_scores.begin(), all_scores.end()));
    }
}

INSTANTIATE_TEST_CASE_P(
    IntegratedScorer,
    IntegratedScorerTest,
    ::testing::Combine(
        ::testing::Values(5, 10, 20), // number of top genes.
        ::testing::Values(0.5, 0.8, 0.9) // number of quantiles.
    )
);
