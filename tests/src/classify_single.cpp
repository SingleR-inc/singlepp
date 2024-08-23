#include <gtest/gtest.h>
#include "custom_parallel.h"

#include "singlepp/classify_single.hpp"
#include "tatami/tatami.hpp"

#include "mock_markers.h"
#include "spawn_matrix.h"
#include "naive_method.h"

#include <memory>
#include <vector>
#include <random>

class ClassifySingleSimpleTest : public ::testing::TestWithParam<std::tuple<int, double> > {};

TEST_P(ClassifySingleSimpleTest, Simple) {
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

    auto markers = mock_markers<int>(nlabels, 50, ngenes); 

    // Performing classification without fine-tuning for a reference comparison.
    singlepp::TrainSingleOptions<int, double> bopt;
    bopt.top = top;
    auto trained = singlepp::train_single(*refs, labels.data(), markers, bopt);

    singlepp::ClassifySingleOptions<double> copt;
    copt.fine_tune = false;
    copt.quantile = quantile;
    auto output = singlepp::classify_single<int>(*mat, trained, copt);

    // Implementing the reference score calculation.
    auto original_markers = markers;
    auto subset = singlepp::internal::subset_to_markers(markers, top);
    auto naive = naive_method(nlabels, labels, refs, mat, subset, quantile);

    int NC = mat->ncol();
    for (int c = 0; c < NC; ++c) {
        EXPECT_EQ(naive.best[c], output.best[c]);
        EXPECT_TRUE(std::abs(naive.delta[c] - output.delta[c]) < 1e-6);
        EXPECT_TRUE(output.delta[c] > 0);

        for (size_t r = 0; r < nlabels; ++r) {
            EXPECT_TRUE(std::abs(naive.scores[r][c] - output.scores[r][c]) < 1e-6);
        }
    }

    // Same result with multiple threads.
    {
        bopt.num_threads = 3;
        auto ptrained = singlepp::train_single(*refs, labels.data(), original_markers, bopt);
        copt.num_threads = 3;
        auto poutput = singlepp::classify_single<int>(*mat, ptrained, copt);

        EXPECT_EQ(output.best, poutput.best);
        EXPECT_EQ(output.delta, poutput.delta);
        for (size_t r = 0; r < nlabels; ++r) {
            EXPECT_EQ(output.scores[r], poutput.scores[r]);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    ClassifySingle,
    ClassifySingleSimpleTest,
    ::testing::Combine(
        ::testing::Values(5, 10, 20), // number of top genes.
        ::testing::Values(1, 0.93, 0.8, 0.66) // quantile
    )
);

class ClassifySingleIntersectTest : public ::testing::TestWithParam<std::tuple<int, double, double> > {};

TEST_P(ClassifySingleIntersectTest, Intersect) {
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
    auto markers = mock_markers<int>(nlabels, 50, right.size()); 

    // Computing the observed result.
    singlepp::TrainSingleOptions<int, double> bopt;
    bopt.top = top;
    auto trained = singlepp::train_single_intersect<int>(left.size(), left.data(), *refs, right.data(), labels.data(), markers, bopt);

    singlepp::ClassifySingleOptions<double> copt;
    copt.quantile = quantile;
    auto result = singlepp::classify_single_intersect<int>(*mat, trained, copt);

    // Computing the reference result using the other run() method,
    // after effectively subsetting the input matrices and reindexing the markers.
    auto intersection = singlepp::intersect_genes(left.size(), left.data(), right.size(), right.data());
    std::pair<std::vector<int>, std::vector<int> > pairs;
    for (const auto& in : intersection) {
        pairs.first.push_back(in.first);
        pairs.second.push_back(in.second);
    }
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

    auto trained2 = singlepp::train_single(*subrefs, labels.data(), markers2, bopt);
    auto result2 = singlepp::classify_single<int>(*submat, trained2, copt);
    EXPECT_EQ(result2.scores[0], result.scores[0]);
    EXPECT_EQ(result2.best, result.best);
    EXPECT_EQ(result2.delta, result.delta);
}

INSTANTIATE_TEST_SUITE_P(
    ClassifySingle,
    ClassifySingleIntersectTest,
    ::testing::Combine(
        ::testing::Values(5, 10, 20), // nuber of top genes.
        ::testing::Values(1, 0.8, 0.66), // quantile
        ::testing::Values(0.5, 0.9) // proportion subset
    )
);

TEST(ClassifySingleTest, Simple) {
    // Mocking up the references.
    size_t ngenes = 200;
 
    size_t nlabels = 3;
    size_t nrefs = 50;
    auto refs = spawn_matrix(ngenes, nrefs, 100);
    auto labels = spawn_labels(nrefs, nlabels, 1000);

    auto markers = mock_markers<int>(nlabels, 50, ngenes); 

    // Checking that we get an exact match when we use the references
    // directly for annotation. We set quantile = 1 so that a perfect
    // correlation to any reference profile guarantees a match.
    singlepp::TrainSingleOptions<int, double> bopt;
    auto trained = singlepp::train_single(*refs, labels.data(), markers, bopt);
    singlepp::ClassifySingleOptions<double> copt;
    copt.quantile = 1;
    auto output = singlepp::classify_single<int>(*refs, trained, copt);

    for (size_t r = 0; r < nrefs; ++r) {
        EXPECT_EQ(labels[r], output.best[r]);
        EXPECT_TRUE(output.delta[r] > 0);
    }
}

TEST(ClassifySingleTest, NoShared) {
    size_t ngenes = 100;
    size_t nlabels = 3;
    size_t nrefs = 50;

    auto mat = spawn_matrix(ngenes, 10, 100);
    auto refs = spawn_matrix(ngenes, nrefs, 100);
    auto labels = spawn_labels(nrefs, nlabels, 1000);
    auto markers = mock_markers<int>(nlabels, 50, ngenes); 

    std::vector<int> left(ngenes), right(ngenes);
    std::iota(left.begin(), left.end(), 0);
    std::iota(right.begin(), right.end(), ngenes);

    singlepp::TrainSingleOptions<int, double> bopt;
    bopt.top = 20;
    auto trained = singlepp::train_single_intersect<int>(ngenes, left.data(), *refs, right.data(), labels.data(), markers, bopt);
    EXPECT_EQ(trained.get_test_subset().size(), 0);
    EXPECT_EQ(trained.get_ref_subset().size(), 0);

    singlepp::ClassifySingleOptions<double> copt;
    auto output = singlepp::classify_single_intersect<int>(*mat, trained, copt);
    for (const auto& curscore : output.scores) {
        for (auto s : curscore) {
            EXPECT_EQ(s, 1); // distance of zero when there are no genes ==> correlation of 1.
        }
    }

    EXPECT_EQ(output.delta, std::vector<double>(mat->ncol())); // all-zeros, no differences between first and second.
}

TEST(ClassifySingleTest, Nulls) {
    // Mocking up the test and references.
    size_t ngenes = 200;
    auto mat = spawn_matrix(ngenes, 5, 42);
 
    size_t nlabels = 3;
    size_t nrefs = 50;
    auto refs = spawn_matrix(ngenes, nrefs, 100);
    auto labels = spawn_labels(nrefs, nlabels, 1000);

    auto markers = mock_markers<int>(nlabels, 50, ngenes); 

    singlepp::TrainSingleOptions<int, double> bopt;
    auto trained = singlepp::train_single(*refs, labels.data(), markers, bopt);
    singlepp::ClassifySingleOptions<double> copt;
    auto full = singlepp::classify_single<int>(*mat, trained, copt);

    // Checking that nulls are respected.
    std::vector<int> best(mat->ncol());
    singlepp::ClassifySingleBuffers<int, double> buffers;
    buffers.best = best.data();
    buffers.delta = NULL;
    buffers.scores.resize(nlabels, NULL);
    singlepp::classify_single(*mat, trained, buffers, copt);

    EXPECT_EQ(best, full.best);
}
