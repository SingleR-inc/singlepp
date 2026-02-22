#include <gtest/gtest.h>

#include "singlepp/classify_single.hpp"
#include "tatami/tatami.hpp"

#include "mock_markers.h"
#include "spawn_matrix.h"
#include "naive_method.h"
#include "compare.h"

#include <memory>
#include <vector>
#include <random>

class ClassifySingleSimpleTest : public ::testing::TestWithParam<std::tuple<int, double> > {};

TEST_P(ClassifySingleSimpleTest, Simple) {
    auto param = GetParam();
    int top = std::get<0>(param);
    double quantile = std::get<1>(param);
    unsigned long long base_seed = top + quantile * 987;

    // Mocking up the test and references.
    size_t ngenes = 200;
    auto mat = spawn_matrix(ngenes, 5, /* seed = */ base_seed + 42);
 
    size_t nlabels = 3;
    size_t nrefs = 50;
    auto refs = spawn_matrix(ngenes, nrefs, /* seed= */ base_seed + 100);
    auto labels = spawn_labels(nrefs, nlabels, /* seed = */ base_seed + 1000);

    auto markers = mock_markers<int>(nlabels, 50, ngenes, /* seed = */ base_seed + 789); 

    // Performing classification without fine-tuning for a reference comparison.
    singlepp::TrainSingleOptions bopt;
    bopt.top = top;
    auto trained = singlepp::train_single(*refs, labels.data(), markers, bopt);
    EXPECT_EQ(trained.test_nrow(), 200);

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

static void check_almost_equal_sparse_results(
    const std::size_t ntest,
    const std::size_t nlabels,
    const singlepp::ClassifySingleResults<int, double>& expected,
    const singlepp::ClassifySingleResults<int, double>& results
) {
    ASSERT_EQ(expected.best.size(), ntest);
    ASSERT_EQ(results.best.size(), ntest);
    ASSERT_EQ(expected.delta.size(), ntest);
    ASSERT_EQ(results.delta.size(), ntest);
    for (std::size_t t = 0; t < ntest; ++t) {
        check_almost_equal_assignment(expected.best[t], expected.delta[t], results.best[t], results.delta[t]);
    }

    ASSERT_EQ(expected.scores.size(), nlabels);
    ASSERT_EQ(results.scores.size(), nlabels);
    for (std::size_t l = 0; l < nlabels; ++l) {
        ASSERT_EQ(expected.scores[l].size(), ntest);
        ASSERT_EQ(results.scores[l].size(), ntest);
        check_almost_equal_vectors(expected.scores[l], results.scores[l]);
    }
}

TEST_P(ClassifySingleSimpleTest, Sparse) {
    auto param = GetParam();
    int top = std::get<0>(param);
    double quantile = std::get<1>(param);
    unsigned long long base_seed = top + quantile * 1000;

    size_t ngenes = 250;
    size_t nlabels = 4;
    auto markers = mock_markers<int>(nlabels, 50, ngenes, /* seed = */ base_seed + 69); 

    int ntest = 11;
    auto test = spawn_sparse_matrix(ngenes, ntest, /* seed = */ base_seed + 42, /* density = */ 0.24);
    auto stest = tatami::convert_to_compressed_sparse<double, int>(*test, true, {});

    size_t nrefs = 51;
    auto labels = spawn_labels(nrefs, nlabels, /* seed = */ base_seed + 1000);
    auto refs = spawn_sparse_matrix(ngenes, nrefs, /* seed = */ base_seed + 100, /* density = */ 0.26);
    auto srefs = tatami::convert_to_compressed_sparse<double, int>(*refs, true, {});

    auto trained = singlepp::train_single(*refs, labels.data(), markers, {});
    auto expected = singlepp::classify_single<int>(*test, trained, {});

    auto sparse_to_dense = singlepp::classify_single<int>(*stest, trained, {});
    check_almost_equal_sparse_results(ntest, nlabels, expected, sparse_to_dense);

    auto sparse_trained = singlepp::train_single(*srefs, labels.data(), markers, {});
    auto dense_to_sparse = singlepp::classify_single<int>(*test, sparse_trained, {});
    check_almost_equal_sparse_results(ntest, nlabels, expected, dense_to_sparse);

    auto sparse_to_sparse = singlepp::classify_single<int>(*test, sparse_trained, {});
    check_almost_equal_sparse_results(ntest, nlabels, expected, sparse_to_sparse);
}

INSTANTIATE_TEST_SUITE_P(
    ClassifySingle,
    ClassifySingleSimpleTest,
    ::testing::Combine(
        ::testing::Values(5, 10, 20), // number of top genes.
        ::testing::Values(1, 0.93, 0.8, 0.66) // quantile
    )
);

/********************************************/

class ClassifySingleIntersectTest : public ::testing::TestWithParam<std::tuple<int, double, double> > {
protected:
    static std::pair<std::vector<int>, std::vector<int> > generate_ids(std::size_t ngenes, double prop_keep, unsigned long long seed) {
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
    unsigned long long base_seed = top + 123 * quantile + 456 * prop; 

    // Creating overlapping ID vectors.
    size_t ngenes = 200;
    auto ids = generate_ids(ngenes, prop, /* seed = */ base_seed);
    const auto& left = ids.first;
    const auto& right = ids.second;

    // Mocking up the test and references.
    auto mat = spawn_matrix(left.size(), 5, /* seed = */ base_seed + 42);

    size_t nlabels = 3;
    size_t nrefs = 50;
    auto refs = spawn_matrix(right.size(), nrefs, /* seed = */ base_seed + 888);
    auto labels = spawn_labels(nrefs, nlabels, /* seed = */ base_seed + 999);
    auto markers = mock_markers<int>(nlabels, 50, right.size(), /* seed = */ base_seed + 69); 

    // Computing the observed result.
    singlepp::TrainSingleOptions bopt;
    bopt.top = top;
    std::vector<int> ref_subset;
    auto trained = singlepp::train_single<double, int>(left.size(), left.data(), *refs, right.data(), labels.data(), markers, &ref_subset, bopt);
    EXPECT_EQ(trained.test_nrow(), left.size());
    EXPECT_EQ(ref_subset.size(), trained.subset().size());
    EXPECT_GE(trained.subset().size(), 10); // should be, on average, 'ngenes * prop^2' overlapping genes.

    singlepp::ClassifySingleOptions<double> copt;
    copt.quantile = quantile;
    auto result = singlepp::classify_single<int>(*mat, trained, copt);

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
        std::mt19937_64 rng(base_seed * 3);
        std::shuffle(intersection.begin(), intersection.end(), rng);
        auto trained2 = singlepp::train_single<double, int>(left.size(), intersection, *refs, labels.data(), markers, NULL, bopt);

        singlepp::ClassifySingleOptions<double> copt;
        copt.quantile = quantile;
        auto result2 = singlepp::classify_single<int>(*mat, trained, copt);

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
    unsigned long long base_seed = top + 12 * quantile + 3456 * prop; 

    // Creating overlapping ID vectors.
    size_t ngenes = 300;
    const auto ids = generate_ids(ngenes, prop, /* seed = */ base_seed);
    const auto& left = ids.first;
    const auto& right = ids.second;

    size_t nlabels = 5;
    auto markers = mock_markers<int>(nlabels, 20, right.size(), /* seed = */ base_seed + 69);

    // Sparse-dense and dense-dense compute the exact same L2, so we can do this comparison without fear of discrepancies due to numerical differences.
    int ntest = 11;
    auto test = spawn_sparse_matrix(left.size(), ntest, /* seed = */ base_seed + 4242, /* density = */ 0.24);
    auto stest = tatami::convert_to_compressed_sparse<double, int>(*test, true, {});
 
    size_t nrefs = 23;
    auto labels = spawn_labels(nrefs, nlabels, /* seed = */ base_seed + 1111);
    auto refs = spawn_sparse_matrix(right.size(), nrefs, /* seed = */ base_seed + 23232, /* density = */ 0.26);
    auto srefs = tatami::convert_to_compressed_sparse<double, int>(*refs, true, {});

    auto trained = singlepp::train_single<double, int>(left.size(), left.data(), *refs, right.data(), labels.data(), markers, NULL, {});
    auto expected = singlepp::classify_single<int>(*test, trained, {});

    auto sparse_to_dense = singlepp::classify_single<int>(*stest, trained, {});
    check_almost_equal_sparse_results(ntest, nlabels, expected, sparse_to_dense);

    auto sparse_trained = singlepp::train_single<double, int>(left.size(), left.data(), *srefs, right.data(), labels.data(), markers, NULL, {});
    auto dense_to_sparse = singlepp::classify_single<int>(*test, sparse_trained, {});
    check_almost_equal_sparse_results(ntest, nlabels, expected, dense_to_sparse);

    auto sparse_to_sparse = singlepp::classify_single<int>(*test, sparse_trained, {});
    check_almost_equal_sparse_results(ntest, nlabels, expected, sparse_to_sparse);
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

/********************************************/

TEST(FineTuneSingle, EdgeCases) {
    size_t ngenes = 200;
    size_t nlabels = 3;
    size_t nprofiles = 50;

    auto markers = mock_markers<int>(nlabels, 10, ngenes, /* seed = */ 20); 
    auto reference = spawn_matrix(ngenes, nprofiles, /* seed = */ 200);
    auto labels = spawn_labels(nprofiles, nlabels, /* seed = */ 2000);

    // Performing classification without fine-tuning for a reference comparison.
    auto trained = singlepp::train_single(*reference, labels.data(), markers, {});
    singlepp::FineTuneSingle<false, false, int, int, double, double> ft(trained);
    singlepp::RankedVector<double, int> placeholder;

    // Check early exit conditions, when there is one clear winner or all of
    // the labels are equal (i.e., no contraction of the feature space).
    {
        std::vector<double> scores { 0.2, 0.5, 0.1 };
        auto output = ft.run(placeholder, trained, 0.8, 0.05, scores);
        EXPECT_EQ(output.first, 1);
        EXPECT_EQ(output.second, 0.3);

        std::fill(scores.begin(), scores.end(), 0.5);
        scores[0] = 0.51;
        output = ft.run(placeholder, trained, 1, 0.05, scores);
        EXPECT_EQ(output.first, 0); // first entry of scores is maxed.
        EXPECT_FLOAT_EQ(output.second, 0.01);
    }

    // Check edge case when there is only a single label, based on the length of 'scores'.
    {
        std::vector<double> scores { 0.5 };
        auto output = ft.run(placeholder, trained, 0.8, 0.05, scores);
        EXPECT_EQ(output.first, 0);
        EXPECT_TRUE(std::isnan(output.second));
    }
}

TEST(FineTuneSingle, ExactRecovery) {
    size_t ngenes = 200;
    size_t nlabels = 3;
    size_t nprofiles = 50;

    auto markers = mock_markers<int>(nlabels, 10, ngenes, /* seed = */ 30); 
    auto reference = spawn_matrix(ngenes, nprofiles, /* seed = */ 300);
    auto labels = spawn_labels(nprofiles, nlabels, /* seed = */ 3000);

    auto trained = singlepp::train_single(*reference, labels.data(), markers, {});
    singlepp::FineTuneSingle<false, false, int, int, double, double> ft(trained);

    // Checking that we eventually pick up the reference, if the input profile
    // is identical to one of the profiles. We set the quantile to 1 to
    // guarantee a score of 1 from a correlation of 1.
    const auto nmarkers = trained.subset().size();
    auto wrk = reference->dense_column(trained.subset());
    std::vector<double> buffer(nmarkers);

    for (size_t r = 0; r < nprofiles; ++r) {
        auto vec = wrk->fetch(r, buffer.data());
        auto ranked = fill_ranks<int>(nmarkers, vec);

        std::vector<double> scores { 0.5, 0.49, 0.48 };
        scores[(labels[r] + 1) % nlabels] = 0; // forcing another label to be zero so that it actually does the fine-tuning.
        auto output = ft.run(ranked, trained, 1, 0.05, scores);
        EXPECT_EQ(output.first, labels[r]);

        scores = std::vector<double>{ 0.5, 0.5, 0.5 };
        scores[labels[r]] = 0; // forcing it to match to some other label. 
        auto output2 = ft.run(ranked, trained, 1, 0.05, scores);
        EXPECT_NE(output2.first, labels[r]);
    }
}

TEST(FineTuneSingle, Diagonal) {
    size_t ngenes = 200;
    size_t nlabels = 3;
    size_t nprofiles = 50;

    // This time there are only markers on the diagonals.
    auto markers = mock_markers_diagonal<int>(nlabels, 10, ngenes, /* seed = */ 40); 
    auto reference = spawn_matrix(ngenes, nprofiles, /* seed = */ 400);
    auto labels = spawn_labels(nprofiles, nlabels, /* seed = */ 4000);

    auto trained = singlepp::train_single(*reference, labels.data(), markers, {});
    singlepp::FineTuneSingle<false, false, int, int, double, double> ft(trained);

    const auto nmarkers = trained.subset().size();
    auto wrk = reference->dense_column(trained.subset());
    std::vector<double> buffer(nmarkers);

    for (size_t r = 0; r < nprofiles; ++r) {
        auto vec = wrk->fetch(r, buffer.data()); 
        auto ranked = fill_ranks<int>(nmarkers, vec);

        std::vector<double> scores { 0.49, 0.5, 0.48 };
        scores[(labels[r] + 1) % nlabels] = 0; // forcing another label to be zero so that it actually does the fine-tuning.
        auto output = ft.run(ranked, trained, 1, 0.05, scores);

        // The key point here is that the diagonals are actually used,
        // so we don't end up with an empty ranking vector and NaN scores.
        EXPECT_EQ(output.first, labels[r]);
        EXPECT_TRUE(output.second > 0);
    }
}

TEST(FineTuneSingle, Sparse) {
    size_t ngenes = 213;
    size_t nlabels = 4;
    size_t nprofiles = 50;

    auto markers = mock_markers<int>(nlabels, 10, ngenes, /* seed = */ 4060); 
    auto labels = spawn_labels(nprofiles, nlabels, /* seed = */ 4070);

    auto new_reference = spawn_sparse_matrix(ngenes, nprofiles, /* seed = */ 4080, /* density = */ 0.3);
    auto new_trained = singlepp::train_single<double>(*new_reference, labels.data(), markers, {});
    singlepp::FineTuneSingle<false, false, int, int, double, double> new_ft(new_trained);
    singlepp::FineTuneSingle<true, false, int, int, double, double> new_ft2(new_trained);

    auto sparse_reference = tatami::convert_to_compressed_sparse<double, int>(*new_reference, true, {});
    auto sparse_trained = singlepp::train_single<double>(*sparse_reference, labels.data(), markers, {});
    singlepp::FineTuneSingle<false, true, int, int, double, double> sparse_ft(sparse_trained);
    singlepp::FineTuneSingle<true, true, int, int, double, double> sparse_ft2(sparse_trained);

    const int ntest = 100; 
    auto new_test = spawn_sparse_matrix(ngenes, ntest, /* seed = */ 5060, /* density = */ 0.2);

    const auto nmarkers = new_trained.subset().size();
    auto wrk = new_test->dense_column(new_trained.subset());
    std::vector<double> buffer(nmarkers);

    for (int t = 0; t < ntest; ++t) {
        auto vec = wrk->fetch(t, buffer.data()); 
        auto ranked = fill_ranks<int>(nmarkers, vec);

        std::vector<double> scores(nlabels, 0.5);
        const auto empty = t % nlabels;
        scores[empty] = 0; // forcing one of the labels to be zero so that it actually does the fine-tuning.

        auto score_copy = scores;
        auto expected = new_ft.run(ranked, new_trained, 0.8, 0.05, score_copy);
        EXPECT_NE(expected.first, empty);

        // Due to differences in numerical precision between dense/sparse calculations, comparisons may not be exact.
        // This results in different 'best' labels in the presence of near-ties, so if there's a mismatch,
        // we check that the delta is indeed near-zero, i.e., there is a near-tie. 
        score_copy = scores;
        auto dense_to_sparse = sparse_ft.run(ranked, sparse_trained, 0.8, 0.05, score_copy);
        check_almost_equal_assignment(expected.first, expected.second, dense_to_sparse.first, dense_to_sparse.second);

        singlepp::RankedVector<double, int> sparse_ranked;
        for (auto r : ranked) {
            if (r.first) {
                sparse_ranked.push_back(r);
            }
        }

        score_copy = scores;
        auto sparse_to_dense = new_ft2.run(sparse_ranked, new_trained, 0.8, 0.05, score_copy);
        check_almost_equal_assignment(expected.first, expected.second, sparse_to_dense.first, sparse_to_dense.second);

        score_copy = scores;
        auto sparse_to_sparse = sparse_ft2.run(sparse_ranked, sparse_trained, 0.8, 0.05, score_copy);
        check_almost_equal_assignment(expected.first, expected.second, sparse_to_sparse.first, sparse_to_sparse.second);
    }
}

/********************************************/

TEST(ClassifySingle, Simple) {
    // Mocking up the references.
    size_t ngenes = 200;
 
    size_t nlabels = 3;
    size_t nrefs = 50;
    auto refs = spawn_matrix(ngenes, nrefs, /* seed = */ 11);
    auto labels = spawn_labels(nrefs, nlabels, /* seed = */ 111);

    auto markers = mock_markers<int>(nlabels, 50, ngenes, /* seed = */ 1111); 

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

TEST(ClassifySingle, NoShared) {
    size_t ngenes = 100;
    size_t nlabels = 3;
    size_t nrefs = 50;

    auto mat = spawn_matrix(ngenes, 10, /* seed = */ 100);
    auto refs = spawn_matrix(ngenes, nrefs, /* seed = */ 101);
    auto labels = spawn_labels(nrefs, nlabels, /* seed = */ 102);
    auto markers = mock_markers<int>(nlabels, 50, ngenes, /* seed = */ 103); 

    std::vector<int> left(ngenes), right(ngenes);
    std::iota(left.begin(), left.end(), 0);
    std::iota(right.begin(), right.end(), ngenes);

    singlepp::TrainSingleOptions bopt;
    bopt.top = 20;
    std::vector<int> ref_subset;
    auto trained = singlepp::train_single<double, int>(ngenes, left.data(), *refs, right.data(), labels.data(), markers, &ref_subset, bopt);
    EXPECT_EQ(trained.subset().size(), 0);
    EXPECT_EQ(ref_subset.size(), 0);

    singlepp::ClassifySingleOptions<double> copt;
    auto output = singlepp::classify_single<int>(*mat, trained, copt);
    for (const auto& curscore : output.scores) {
        for (auto s : curscore) {
            EXPECT_EQ(s, 1); // distance of zero when there are no genes ==> correlation of 1.
        }
    }

    EXPECT_EQ(output.delta, std::vector<double>(mat->ncol())); // all-zeros, no differences between first and second.
}

TEST(ClassifySingle, Nulls) {
    // Mocking up the test and references.
    size_t ngenes = 200;
    auto mat = spawn_matrix(ngenes, 5, /* seed = */ 42);
 
    size_t nlabels = 3;
    size_t nrefs = 50;
    auto refs = spawn_matrix(ngenes, nrefs, /* seed = */ 43);
    auto labels = spawn_labels(nrefs, nlabels, /* seed = */ 44);

    auto markers = mock_markers<int>(nlabels, 50, ngenes,  /* seed = */ 45); 

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

TEST(ClassifySingle, Mismatch) {
    size_t ngenes = 200;
    size_t nlabels = 3;
    size_t nrefs = 50;

    auto refs = spawn_matrix(ngenes, nrefs, /* seed = */ 22);
    auto labels = spawn_labels(nrefs, nlabels, /* seed = */ 23);
    auto markers = mock_markers<int>(nlabels, 50, ngenes, /* seed = */ 24); 

    singlepp::TrainSingleOptions bopt;
    auto trained = singlepp::train_single(*refs, labels.data(), markers, bopt);

    auto test = spawn_matrix(ngenes + 10, nrefs, /* seed = */ 25);
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
