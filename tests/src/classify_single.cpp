#include <gtest/gtest.h>

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
    singlepp::TrainSingleOptions bopt;
    bopt.top = top;
    auto trained = singlepp::train_single(*refs, labels.data(), markers, bopt);
    EXPECT_EQ(trained.get_test_nrow(), 200);

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

TEST_P(ClassifySingleSimpleTest, Sparse) {
    auto param = GetParam();
    int top = std::get<0>(param);
    double quantile = std::get<1>(param);

    // Mocking up the test and references.
    size_t ngenes = 250;
    int ntest = 7;
    auto mat = spawn_sparse_matrix(ngenes, ntest, /* seed = */ 42 * quantile + top, /* density = */ 0.24);
    auto smat = tatami::convert_to_compressed_sparse<double, int>(*mat, true, {});
 
    size_t nrefs = 31;
    auto refs = spawn_sparse_matrix(ngenes, nrefs, /* seed = */ 100 * quantile + top, /* density = */ 0.26);
    auto srefs = tatami::convert_to_compressed_sparse<double, int>(*refs, true, {});

    size_t nlabels = 4;
    auto labels = spawn_labels(nrefs, nlabels, /* seed = */ 1000 * quantile + top);
    auto markers = mock_markers<int>(nlabels, 50, ngenes, /* seed = */ 69 * quantile + top); 

    // Comparing every combination of sparse and dense. 
    auto trained = singlepp::train_single(*refs, labels.data(), markers, {});
    auto output = singlepp::classify_single<int>(*mat, trained, {});
    auto sparse_output = singlepp::classify_single<int>(*smat, trained, {});

    auto trained2 = singlepp::train_single(*srefs, labels.data(), markers, {});
    auto output2 = singlepp::classify_single<int>(*mat, trained2, {});
    auto sparse_output2 = singlepp::classify_single<int>(*smat, trained2, {});

    EXPECT_EQ(output.best, sparse_output.best);
    EXPECT_EQ(output.best, output2.best);
    EXPECT_EQ(output.best, sparse_output2.best);

    for (int t = 0; t < ntest; ++t) {
        EXPECT_FLOAT_EQ(output.delta[t], sparse_output.delta[t]);
        EXPECT_FLOAT_EQ(output.delta[t], output2.delta[t]);
        EXPECT_FLOAT_EQ(output.delta[t], sparse_output2.delta[t]);
    }

    for (size_t l = 0; l < nlabels; ++l) {
        for (int t = 0; t < ntest; ++t) {
            EXPECT_FLOAT_EQ(output.scores[l][t], sparse_output.scores[l][t]);
            EXPECT_FLOAT_EQ(output.scores[l][t], output2.scores[l][t]);
            EXPECT_FLOAT_EQ(output.scores[l][t], sparse_output2.scores[l][t]);
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

class ClassifySingleIntersectTest : public ::testing::TestWithParam<std::tuple<int, double, double> > {
protected:
    std::pair<std::vector<int>, std::vector<int> > generate_ids(std::size_t ngenes, double prop_keep, int seed) {
        std::mt19937_64 rng(seed);
        std::vector<int> left, right;
        std::uniform_real_distribution<> dist;

        for (size_t x = 0; x < ngenes; ++x) {
            if (dist(rng) < prop_keep) { 
                left.push_back(x);
            }
            if (dist(rng) < prop_keep) { 
                right.push_back(x);
            }
        }

        std::shuffle(left.begin(), left.end(), rng);
        std::shuffle(right.begin(), right.end(), rng);
        return std::make_pair(std::move(left), std::move(right));
    }

};

TEST_P(ClassifySingleIntersectTest, Intersect) {
    auto param = GetParam();
    int top = std::get<0>(param);
    double quantile = std::get<1>(param);
    double prop = std::get<2>(param);

    // Creating overlapping ID vectors.
    size_t ngenes = 200;
    auto ids = generate_ids(ngenes, prop, top + 100 * quantile + 1000 * prop);
    const auto& left = ids.first;
    const auto& right = ids.second;

    // Mocking up the test and references.
    auto mat = spawn_matrix(left.size(), 5, 42);

    size_t nlabels = 3;
    size_t nrefs = 50;
    auto refs = spawn_matrix(right.size(), nrefs, 100);
    auto labels = spawn_labels(nrefs, nlabels, 1000);
    auto markers = mock_markers<int>(nlabels, 50, right.size()); 

    // Computing the observed result.
    singlepp::TrainSingleOptions bopt;
    bopt.top = top;
    auto trained = singlepp::train_single_intersect<double, int>(left.size(), left.data(), *refs, right.data(), labels.data(), markers, bopt);
    EXPECT_EQ(trained.get_test_nrow(), left.size());
    EXPECT_EQ(trained.get_ref_subset().size(), trained.get_test_subset().size());
    EXPECT_GE(trained.get_test_subset().size(), 10); // should be, on average, 'ngenes * prop^2' overlapping genes.

    singlepp::ClassifySingleOptions<double> copt;
    copt.quantile = quantile;
    auto result = singlepp::classify_single_intersect<int>(*mat, trained, copt);

    // Computing the reference result using the classify_single() function,
    // after effectively subsetting the input matrices and reindexing the markers.
    {
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

    // Using the shuffled intersection to check that the order doesn't matter.
    {
        auto intersection = singlepp::intersect_genes<int>(left.size(), left.data(), right.size(), right.data());
        std::mt19937_64 rng(top + quantile * 3123 + prop * 452);
        std::shuffle(intersection.begin(), intersection.end(), rng);
        auto trained2 = singlepp::train_single_intersect<double, int>(left.size(), intersection, *refs, labels.data(), markers, bopt);

        singlepp::ClassifySingleOptions<double> copt;
        copt.quantile = quantile;
        auto result2 = singlepp::classify_single_intersect<int>(*mat, trained, copt);

        EXPECT_EQ(result2.scores[0], result.scores[0]);
        EXPECT_EQ(result2.best, result.best);
        EXPECT_EQ(result2.delta, result.delta);
    }

    // Back-compatibility check for the soft-deprecated intersection method.
    {
        auto intersection = singlepp::intersect_genes<int>(left.size(), left.data(), right.size(), right.data());
        auto trained2 = singlepp::train_single_intersect<double, int>(intersection, *refs, labels.data(), markers, bopt);

        singlepp::ClassifySingleOptions<double> copt;
        copt.quantile = quantile;
        auto result2 = singlepp::classify_single_intersect<int>(*mat, trained, copt);

        EXPECT_EQ(result2.scores[0], result.scores[0]);
        EXPECT_EQ(result2.best, result.best);
        EXPECT_EQ(result2.delta, result.delta);
    }
}

TEST_P(ClassifySingleIntersectTest, Sparse) {
    auto param = GetParam();
    int top = std::get<0>(param);
    double quantile = std::get<1>(param);
    double prop = std::get<2>(param);

    // Creating overlapping ID vectors.
    size_t ngenes = 300;
    auto ids = generate_ids(ngenes, prop, top + 100 * quantile + 1000 * prop);
    const auto& left = ids.first;
    const auto& right = ids.second;

    // Mocking up the test and references.
    int ntest = 11;
    auto mat = spawn_sparse_matrix(left.size(), ntest, /* seed = */ 42 * quantile + top, /* density = */ 0.24);
    auto smat = tatami::convert_to_compressed_sparse<double, int>(*mat, true, {});
 
    size_t nrefs = 23;
    auto refs = spawn_sparse_matrix(right.size(), nrefs, /* seed = */ 100 * quantile + top, /* density = */ 0.26);
    auto srefs = tatami::convert_to_compressed_sparse<double, int>(*refs, true, {});

    size_t nlabels = 5;
    auto labels = spawn_labels(nrefs, nlabels, /* seed = */ 1000 * quantile + top);
    auto markers = mock_markers<int>(nlabels, 20, right.size(), /* seed = */ 69 * quantile + top); 

    // Comparing every combination of sparse and dense. 
    auto trained = singlepp::train_single_intersect<double, int>(left.size(), left.data(), *refs, right.data(), labels.data(), markers, {});
    auto output = singlepp::classify_single_intersect<int>(*mat, trained, {});
    auto sparse_output = singlepp::classify_single_intersect<int>(*smat, trained, {});

    auto trained2 = singlepp::train_single_intersect<double, int>(left.size(), left.data(), *srefs, right.data(), labels.data(), markers, {});
    auto output2 = singlepp::classify_single_intersect<int>(*mat, trained2, {});
    auto sparse_output2 = singlepp::classify_single_intersect<int>(*smat, trained2, {});

    EXPECT_EQ(output.best, sparse_output.best);
    EXPECT_EQ(output.best, output2.best);
    EXPECT_EQ(output.best, sparse_output2.best);

    for (int t = 0; t < ntest; ++t) {
        EXPECT_FLOAT_EQ(output.delta[t], sparse_output.delta[t]);
        EXPECT_FLOAT_EQ(output.delta[t], output2.delta[t]);
        EXPECT_FLOAT_EQ(output.delta[t], sparse_output2.delta[t]);
    }

    for (size_t l = 0; l < nlabels; ++l) {
        for (int t = 0; t < ntest; ++t) {
            EXPECT_FLOAT_EQ(output.scores[l][t], sparse_output.scores[l][t]);
            EXPECT_FLOAT_EQ(output.scores[l][t], output2.scores[l][t]);
            EXPECT_FLOAT_EQ(output.scores[l][t], sparse_output2.scores[l][t]);
        }
    }
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
    singlepp::TrainSingleOptions bopt;
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

    singlepp::TrainSingleOptions bopt;
    bopt.top = 20;
    auto trained = singlepp::train_single_intersect<double, int>(ngenes, left.data(), *refs, right.data(), labels.data(), markers, bopt);
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

    singlepp::TrainSingleOptions bopt;
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

TEST(ClassifySingleTest, SimpleMismatch) {
    size_t ngenes = 200;
    size_t nlabels = 3;
    size_t nrefs = 50;

    auto refs = spawn_matrix(ngenes, nrefs, 100);
    auto labels = spawn_labels(nrefs, nlabels, 1000);
    auto markers = mock_markers<int>(nlabels, 50, ngenes); 

    singlepp::TrainSingleOptions bopt;
    auto trained = singlepp::train_single(*refs, labels.data(), markers, bopt);

    auto test = spawn_matrix(ngenes + 10, nrefs, 100);
    singlepp::ClassifySingleOptions<double> copt;
    copt.quantile = 1;

    bool failed = false;
    try {
        singlepp::classify_single<int>(*test, trained, copt);
    } catch (std::exception& e) {
        failed = true;
        EXPECT_TRUE(std::string(e.what()).find("number of rows") != std::string::npos);
    }
    EXPECT_TRUE(failed);
}

TEST(ClassifySingleTest, IntersectMismatch) {
    size_t ngenes = 200;
    size_t nlabels = 3;
    size_t nrefs = 50;

    auto refs = spawn_matrix(ngenes, nrefs, 100);
    auto labels = spawn_labels(nrefs, nlabels, 1000);
    auto markers = mock_markers<int>(nlabels, 50, ngenes); 

    std::vector<int> ids(ngenes);
    std::iota(ids.begin(), ids.end(), 0);
    singlepp::TrainSingleOptions bopt;
    auto trained = singlepp::train_single_intersect<double, int>(ngenes, ids.data(), *refs, ids.data(), labels.data(), markers, bopt);

    auto test = spawn_matrix(ngenes + 10, nrefs, 100);
    singlepp::ClassifySingleOptions<double> copt;
    copt.quantile = 1;

    bool failed = false;
    try {
        singlepp::classify_single_intersect<int>(*test, trained, copt);
    } catch (std::exception& e) {
        failed = true;
        EXPECT_TRUE(std::string(e.what()).find("number of rows") != std::string::npos);
    }
    EXPECT_TRUE(failed);
}
