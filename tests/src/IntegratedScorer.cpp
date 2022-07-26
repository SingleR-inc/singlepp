#include <gtest/gtest.h>
#include "custom_parallel.h"

#include "singlepp/IntegratedBuilder.hpp"
#include "singlepp/IntegratedScorer.hpp"
#include "spawn_matrix.h"
#include "mock_markers.h"
#include "naive_method.h"

class IntegratedScorerTest : public ::testing::TestWithParam<std::tuple<int, double> > {};

template<class Prebuilt>
std::vector<std::vector<int> > mock_best_choices(size_t ntest, const std::vector<Prebuilt>& prebuilts, size_t seed) {
    size_t nrefs = prebuilts.size();
    std::vector<std::vector<int> > chosen(nrefs);

    std::mt19937_64 rng(seed);
    for (size_t r = 0; r < nrefs; ++r) {
        size_t nlabels = prebuilts[r].markers.size();
        for (size_t t = 0; t < ntest; ++t) {
            chosen[r].push_back(rng() % nlabels);
        }
    }

    return chosen;
}

auto split_by_labels(const std::vector<std::vector<int> >& labels) {
    std::vector<std::vector<std::vector<int> > > by_labels(labels.size());
    for (size_t r = 0; r < labels.size(); ++r) {
        const auto& current = labels[r];
        size_t nlabels = *std::max_element(current.begin(), current.end()) + 1;
        by_labels[r] = split_by_label(nlabels, current);
    }
    return by_labels;
}

template<class Prebuilt>
std::unordered_set<int> create_universe(size_t cell, const std::vector<Prebuilt>& prebuilts, const std::vector<std::vector<int> >& chosen) {
    std::unordered_set<int> tmp;
    for (size_t r = 0; r < prebuilts.size(); ++r) {
        const auto& pre = prebuilts[r];
        const auto& best_markers = pre.markers[chosen[r][cell]];
        for (const auto& x : best_markers) {
            for (auto y : x) {
                if constexpr(std::is_same<Prebuilt, singlepp::BasicBuilder::Prebuilt>::value) {
                    tmp.insert(pre.subset[y]);
                } else {
                    tmp.insert(pre.mat_subset[y]);
                }
            }
        }
    }
    return tmp;
}

TEST_P(IntegratedScorerTest, Basic) {
    // Mocking up the test and individual references, and creating the
    // integrated set of references with IntegratedBuilder.
    size_t ntest = 20;
    size_t ngenes = 2000;
    auto test = spawn_matrix(ngenes, ntest, 123456790);

    size_t nsamples = 50;
    size_t nrefs = 3;
    auto param = GetParam();
    int ntop = std::get<0>(param);
    double quantile = std::get<1>(param);

    singlepp::BasicBuilder builder;
    builder.set_top(ntop);
    singlepp::IntegratedBuilder ibuilder;

    std::vector<std::shared_ptr<tatami::Matrix<double, int> > > matrices;
    std::vector<std::vector<int> > labels;
    std::vector<singlepp::BasicBuilder::Prebuilt> prebuilts;

    for (size_t r = 0; r < nrefs; ++r) {
        size_t seed = r * 321;
        size_t nlabels = 3 + r;

        matrices.push_back(spawn_matrix(ngenes, nsamples, seed));
        labels.push_back(spawn_labels(nsamples, nlabels, seed * 2));

        auto markers = mock_markers(nlabels, 50, ngenes, seed * 3);
        auto pre = builder.run(matrices.back().get(), labels.back().data(), std::move(markers));
        prebuilts.push_back(std::move(pre));

        ibuilder.add(matrices.back().get(), labels.back().data(), prebuilts.back());
    }

    auto integrated = ibuilder.finish();

    // Mocking up some of the best choices.
    auto chosen = mock_best_choices(ntest, prebuilts, 13579);
    std::vector<const int*> chosen_ptrs(nrefs);
    for (size_t r = 0; r < nrefs; ++r) {
        chosen_ptrs[r] = chosen[r].data();
    }

    // Comparing the IntegratedScorer output to a reference calculation.
    singlepp::IntegratedScorer scorer;
    scorer.set_quantile(quantile);
    auto output = scorer.run(test.get(), chosen_ptrs, integrated);
    auto by_labels = split_by_labels(labels);

    for (size_t t = 0; t < ntest; ++t) {
        auto my_universe = create_universe(t, prebuilts, chosen);
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

    // Same results in parallel.
    scorer.set_num_threads(3);
    auto poutput = scorer.run(test.get(), chosen_ptrs, integrated);
    EXPECT_EQ(output.best, poutput.best);
    EXPECT_EQ(output.delta, poutput.delta);
    for (size_t r = 0; r < nrefs; ++r) {
        EXPECT_EQ(output.scores[r], poutput.scores[r]);
    }
}

TEST_P(IntegratedScorerTest, Intersected) {
    // Mocking up the test and individual references, and creating the
    // integrated set of references with IntegratedBuilder.
    size_t ntest = 20;
    size_t ngenes = 2000;
    auto test = spawn_matrix(ngenes, ntest, 24680);

    size_t nsamples = 50;
    size_t nrefs = 3;
    auto param = GetParam();
    int ntop = std::get<0>(param);
    double quantile = std::get<1>(param);

    std::vector<std::shared_ptr<tatami::Matrix<double, int> > > matrices;
    std::vector<std::vector<int> > labels;
    std::vector<singlepp::BasicBuilder::PrebuiltIntersection> prebuilts;

    std::vector<int> ids(ngenes);
    for (size_t g = 0; g < ngenes; ++g) {
        ids[g] = g;
    }
    std::vector<std::vector<int> > kept(nrefs);

    singlepp::BasicBuilder builder;
    builder.set_top(ntop);
    singlepp::IntegratedBuilder ibuilder;

    for (size_t r = 0; r < nrefs; ++r) {
        size_t seed = r * 10;
        size_t nlabels = 3 + r;

        std::mt19937_64 rng(seed);
        auto& keep = kept[r];
        for (size_t s = 0; s < ngenes; ++s) {
            if (rng() % 100 < 50) {
                keep.push_back(s);
            } else {
                keep.push_back(-1); // i.e., unique to the reference.
            }
        }

        matrices.push_back(spawn_matrix(keep.size(), nsamples, seed));
        labels.push_back(spawn_labels(nsamples, nlabels, seed * 2));

        // Adding each one to the list.
        auto markers = mock_markers(nlabels, 50, keep.size(), seed * 3);
        auto pre = builder.run(ngenes, ids.data(), matrices.back().get(), keep.data(), labels.back().data(), markers);
        prebuilts.push_back(std::move(pre));

        ibuilder.add(ngenes, ids.data(), matrices.back().get(), keep.data(), labels.back().data(), prebuilts.back());
    }

    auto integrated = ibuilder.finish();

    // Mocking up some of the best choices.
    auto chosen = mock_best_choices(ntest, prebuilts, 2468);
    std::vector<const int*> chosen_ptrs(nrefs);
    for (size_t r = 0; r < nrefs; ++r) {
        chosen_ptrs[r] = chosen[r].data();
    }

    // Comparing the IntegratedScorer to a reference calculation.
    singlepp::IntegratedScorer scorer;
    scorer.set_quantile(quantile);
    auto output = scorer.run(test.get(), chosen_ptrs, integrated);
    auto by_labels = split_by_labels(labels);

    std::vector<std::unordered_map<int, int> > reverser(nrefs);
    for (size_t r = 0; r < nrefs; ++r) {
        const auto& keep = kept[r];
        for (size_t i = 0; i < keep.size(); ++i) {
            reverser[r][keep[i]] = i;
        }
    }

    for (size_t t = 0; t < ntest; ++t) {
        auto my_universe = create_universe(t, prebuilts, chosen);

        std::vector<double> all_scores;
        for (size_t r = 0; r < nrefs; ++r) {
            std::vector<int> universe_test, universe_ref;
            for (auto s : my_universe) {
                auto it = reverser[r].find(s);
                if (it != reverser[r].end()) {
                    universe_test.push_back(s);
                    universe_ref.push_back(it->second);
                }
            }

            auto col = test->column(t);
            auto scaled = quick_scaled_ranks(col, universe_test);
            double score = naive_score(scaled, by_labels[r][chosen[r][t]], matrices[r].get(), universe_ref, quantile);
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
