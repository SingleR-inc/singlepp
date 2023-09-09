#include <gtest/gtest.h>
#include "custom_parallel.h"

#include "singlepp/Classifier.hpp"
#include "tatami/tatami.hpp"

#include "mock_markers.h"
#include "spawn_matrix.h"
#include "naive_method.h"

#include <memory>
#include <vector>
#include <random>

class ClassifierSimpleTest : public ::testing::TestWithParam<std::tuple<int, double> > {};

TEST_P(ClassifierSimpleTest, Simple) {
    auto param = GetParam();
    int top = std::get<0>(param);
    double quantile = std::get<1>(param);

    // Mocking up the test and references.
    size_t ngenes = 200;
    auto mat = spawn_matrix(ngenes, 5, 42);
 
    size_t nlabels = 3;
    size_t nrefs = 50;
    auto refs = spawn_matrix(ngenes, nrefs, 100);
    auto labels = spawn_labels(nrefs, nlabels, 1000);

    auto markers = mock_markers(nlabels, 50, ngenes); 

    // Running the implementation.
    singlepp::Classifier runner;
    runner.set_fine_tune(false).set_top(top).set_quantile(quantile);
    auto output = runner.run(mat.get(), refs.get(), labels.data(), markers);

    // Implementing the reference score calculation.
    auto original_markers = markers;
    auto subset = singlepp::subset_markers(markers, top);
    auto naive = naive_method(nlabels, labels, refs, mat, subset, quantile);

    for (size_t c = 0; c < mat->ncol(); ++c) {
        EXPECT_EQ(naive.best[c], output.best[c]);
        EXPECT_TRUE(std::abs(naive.delta[c] - output.delta[c]) < 1e-6);
        EXPECT_TRUE(output.delta[c] > 0);

        for (size_t r = 0; r < nlabels; ++r) {
            EXPECT_TRUE(std::abs(naive.scores[r][c] - output.scores[r][c]) < 1e-6);
        }
    }

    // Same result with multiple threads.
    runner.set_num_threads(3);
    auto poutput = runner.run(mat.get(), refs.get(), labels.data(), original_markers);
    EXPECT_EQ(output.best, poutput.best);
    EXPECT_EQ(output.delta, poutput.delta);
    for (size_t r = 0; r < nlabels; ++r) {
        EXPECT_EQ(output.scores[r], poutput.scores[r]);
    }
}

TEST_P(ClassifierSimpleTest, AlreadySubset) {
    auto param = GetParam();
    int top = std::get<0>(param);
    double quantile = std::get<1>(param);

    // Mocking up the test and references.
    size_t ngenes = 200;
    auto mat = spawn_matrix(ngenes, 5, 12345);
 
    size_t nlabels = 3;
    size_t nrefs = 50;
    auto refs = spawn_matrix(ngenes, nrefs, 67);
    auto labels = spawn_labels(nrefs, nlabels, 89);

    auto markers = mock_markers(nlabels, 50, ngenes); 

    // Running the default.
    singlepp::Classifier runner;
    runner.set_fine_tune(false).set_top(top).set_quantile(quantile);
    auto output = runner.run(mat.get(), refs.get(), labels.data(), markers);

    // Comparing to running on a prebuilt with a provided subset vector. We
    // make it a little complicated (hence the delayed subset and reversal)
    // just to check that all the indexing is correct.
    singlepp::BasicBuilder builder;
    builder.set_top(top);
    auto built = builder.run(refs.get(), labels.data(), markers);
    EXPECT_EQ(built.num_labels(), nlabels);
    EXPECT_EQ(built.num_profiles(), nrefs);

    auto copy = built.subset;
    std::reverse(copy.begin(), copy.end());
    auto sub = tatami::make_DelayedSubset<0>(mat, copy);

    std::vector<int> index(copy.size());
    std::iota(index.rbegin(), index.rend(), 0);

    singlepp::BasicScorer scorer;
    scorer.set_fine_tune(false).set_quantile(quantile);
    auto output2 = scorer.run(sub.get(), built, index.data());

    // Should get the same result.
    EXPECT_EQ(output.best, output2.best);
    EXPECT_EQ(output.delta, output2.delta);
    for (size_t r = 0; r < nlabels; ++r) {
        EXPECT_EQ(output.scores[r], output2.scores[r]);
    }

    // Checking that this works:
    EXPECT_EQ(built.num_labels(), nlabels);
    EXPECT_EQ(built.num_profiles(), nrefs);
}

INSTANTIATE_TEST_CASE_P(
    Classifier,
    ClassifierSimpleTest,
    ::testing::Combine(
        ::testing::Values(5, 10, 20), // number of top genes.
        ::testing::Values(1, 0.93, 0.8, 0.66) // quantile
    )
);

class ClassifierIntersectTest : public ::testing::TestWithParam<std::tuple<int, double, double> > {};

TEST_P(ClassifierIntersectTest, Intersect) {
    auto param = GetParam();
    int top = std::get<0>(param);
    double quantile = std::get<1>(param);
    double prop = std::get<2>(param);

    // Creating overlapping ID vectors.
    size_t ngenes = 200;
    std::mt19937_64 rng(top * quantile * prop);
    std::vector<int> left, right;
    for (size_t x = 0; x < ngenes; ++x) {
        if (rng() % 100 < 100 * prop) { 
            left.push_back(x);
        }
        if (rng() % 100 < 100 * prop) { 
            right.push_back(x);
        }
    }
    std::shuffle(left.begin(), left.end(), rng);
    std::shuffle(right.begin(), right.end(), rng);

    // Mocking up the test and references.
    auto mat = spawn_matrix(left.size(), 5, 42);

    size_t nlabels = 3;
    size_t nrefs = 50;
    auto refs = spawn_matrix(right.size(), nrefs, 100);
    auto labels = spawn_labels(nrefs, nlabels, 1000);
    auto markers = mock_markers(nlabels, 50, right.size()); 

    // Computing the observed result.
    singlepp::Classifier runner;
    runner.set_fine_tune(false).set_top(top).set_quantile(quantile);
    auto result = runner.run(mat.get(), left.data(), refs.get(), right.data(), labels.data(), markers);

    // Computing the result via the build method.
    singlepp::BasicBuilder builder;
    builder.set_top(top);
    auto build0 = builder.run(mat->nrow(), left.data(), refs.get(), right.data(), labels.data(), markers);
    EXPECT_EQ(build0.num_labels(), nlabels);
    EXPECT_EQ(build0.num_profiles(), nrefs);

    singlepp::BasicScorer scorer;
    scorer.set_fine_tune(false).set_quantile(quantile);
    auto result0 = scorer.run(mat.get(), build0);

    EXPECT_EQ(result0.best, result.best);
    EXPECT_EQ(result0.delta, result.delta);
    EXPECT_EQ(build0.num_labels(), nlabels);

    // Computing the reference result using the other run() method,
    // after effectively subsetting the input matrices and reindexing the markers.
    auto intersection = singlepp::intersect_features(left.size(), left.data(), right.size(), right.data());
    std::sort(intersection.begin(), intersection.end());
    auto pairs = singlepp::unzip(intersection);
    auto submat = tatami::make_DelayedSubset<0>(mat, pairs.first);
    auto subrefs = tatami::make_DelayedSubset<0>(refs, pairs.second);

    std::unordered_map<int, int> locations;
    for (size_t i = 0; i < pairs.second.size(); ++i) {
        locations[pairs.second[i]] = i;
    }

    auto markers2 = markers;
    for (size_t i = 0; i < nlabels; ++i) {
        for (size_t j = 0; j < nlabels; ++j) {
            if (i == j) {
                continue;
            }

            std::vector<int> current;
            for (auto s : markers[i][j]) {
                auto it = locations.find(s);
                if (it != locations.end()) {
                    current.push_back(it->second);
                }
            }
            markers2[i][j] = current;
        }
    }

    auto result2 = runner.run(submat.get(), subrefs.get(), labels.data(), markers2);
    EXPECT_EQ(result2.scores[0], result.scores[0]);
    EXPECT_EQ(result2.best, result.best);
    EXPECT_EQ(result2.delta, result.delta);
}

INSTANTIATE_TEST_CASE_P(
    Classifier,
    ClassifierIntersectTest,
    ::testing::Combine(
        ::testing::Values(5, 10, 20), // nuber of top genes.
        ::testing::Values(1, 0.8, 0.66), // quantile
        ::testing::Values(0.5, 0.9) // proportion subset
    )
);

TEST(ClassifierTest, Simple) {
    // Mocking up the references.
    size_t ngenes = 200;
 
    size_t nlabels = 3;
    size_t nrefs = 50;
    auto refs = spawn_matrix(ngenes, nrefs, 100);
    auto labels = spawn_labels(nrefs, nlabels, 1000);

    auto markers = mock_markers(nlabels, 50, ngenes); 

    // Checking that we get an exact match when we use the references
    // directly for annotation. We set quantile = 1 so that a perfect
    // correlation to any reference profile guarantees a match.
    singlepp::Classifier runner;
    runner.set_quantile(1);

    auto output = runner.run(refs.get(), refs.get(), labels.data(), markers);
    for (size_t r = 0; r < nrefs; ++r) {
        EXPECT_EQ(labels[r], output.best[r]);
        EXPECT_TRUE(output.delta[r] > 0);
    }
}

TEST(ClassifierTest, NoShared) {
    size_t ngenes = 100;
    size_t nlabels = 3;
    size_t nrefs = 50;

    auto mat = spawn_matrix(ngenes, 10, 100);
    auto refs = spawn_matrix(ngenes, nrefs, 100);
    auto labels = spawn_labels(nrefs, nlabels, 1000);
    auto markers = mock_markers(nlabels, 50, ngenes); 

    std::vector<int> left(ngenes), right(ngenes);
    std::iota(left.begin(), left.end(), 0);
    std::iota(right.begin(), right.end(), ngenes);

    singlepp::BasicBuilder builder;
    builder.set_top(20);
    auto built = builder.run(ngenes, left.data(), refs.get(), right.data(), labels.data(), markers);
    EXPECT_EQ(built.mat_subset.size(), 0);
    EXPECT_EQ(built.ref_subset.size(), 0);

    singlepp::BasicScorer scorer;
    auto output = scorer.run(mat.get(), built);
    for (const auto& curscore : output.scores) {
        for (auto s : curscore) {
            EXPECT_EQ(s, 1); // distance of zero when there are no genes ==> correlation of 1.
        }
    }

    EXPECT_EQ(output.delta, std::vector<double>(mat->ncol())); // all-zeros, no differences between first and second.
}

TEST(ClassifierTest, Nulls) {
    // Mocking up the test and references.
    size_t ngenes = 200;
    auto mat = spawn_matrix(ngenes, 5, 42);
 
    size_t nlabels = 3;
    size_t nrefs = 50;
    auto refs = spawn_matrix(ngenes, nrefs, 100);
    auto labels = spawn_labels(nrefs, nlabels, 1000);

    auto markers = mock_markers(nlabels, 50, ngenes); 

    // Checking that nulls are respected.
    singlepp::Classifier runner;

    std::vector<int> best(nrefs);
    std::vector<double*> nulls(nlabels, NULL);
    runner.run(refs.get(), refs.get(), labels.data(), markers, best.data(), nulls, NULL);
    
    auto manual = runner.run(refs.get(), refs.get(), labels.data(), markers);
    EXPECT_EQ(best, manual.best);
}
