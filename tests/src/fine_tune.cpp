#include <gtest/gtest.h>
#include "singlepp/fine_tune.hpp"
#include "mock_markers.h"
#include "spawn_matrix.h"
#include "fill_ranks.h"
#include "naive_method.h"

TEST(FillLabelsInUseTest, Basic) {
    std::vector<double> scores { 0.5, 0.2, 0.46 };
    std::vector<int> in_use;

    {
        auto output = singlepp::fill_labels_in_use(scores, 0.05, in_use);
        std::vector<int> expected { 0, 2 };
        EXPECT_EQ(in_use, expected);
        EXPECT_EQ(output.first, 0);
        EXPECT_FLOAT_EQ(output.second, 0.04);
    }

    {
        auto output = singlepp::fill_labels_in_use(scores, 0.01, in_use);
        std::vector<int> expected { 0 };
        EXPECT_EQ(in_use, expected);
        EXPECT_EQ(output.first, 0);
        EXPECT_FLOAT_EQ(output.second, 0.04);
    }

    scores = std::vector<double>{ 0.48, 0.5, 0.2, 0.46 };
    in_use = std::vector<int>{ 5, 10, 100 }; // checking that these are cleared out.
    {
        auto output = singlepp::fill_labels_in_use(scores, 0.05, in_use);
        std::vector<int> expected { 0, 1, 3 };
        EXPECT_EQ(in_use, expected);
        EXPECT_EQ(output.first, 1);
        EXPECT_FLOAT_EQ(output.second, 0.02);
    }
}

TEST(ReplaceLabelsInUseTest, Basic) {
    {
        std::vector<double> scores { 0.48, 0.2, 0.5 };
        std::vector<int> in_use { 4, 5, 6 };

        auto output = singlepp::replace_labels_in_use(scores, 0.05, in_use);
        std::vector<int> expected { 4, 6 };
        EXPECT_EQ(in_use, expected);
        EXPECT_EQ(output.first, 6);
        EXPECT_FLOAT_EQ(output.second, 0.02);
    }

    {
        std::vector<double> scores { 0.2, 0.48, 0.51, 0.5 };
        std::vector<int> in_use { 0, 7, 3, 8 };

        auto output = singlepp::replace_labels_in_use(scores, 0.05, in_use);
        std::vector<int> expected { 7, 3, 8 };
        EXPECT_EQ(in_use, expected);
        EXPECT_EQ(output.first, 3);
        EXPECT_FLOAT_EQ(output.second, 0.01);
    }
}

TEST(FineTuneTest, Basic) {
    // Mocking up the test and references.
    size_t ngenes = 200;
 
    size_t nlabels = 3;
    size_t nrefs = 50;
    auto refs = spawn_matrix(ngenes, nrefs, 200);
    auto labels = spawn_labels(nrefs, nlabels, 2000);

    auto markers = mock_markers(nlabels, 10, ngenes); 

    // Mocking up the reference indices.
    std::vector<int> subset(ngenes);
    std::iota(subset.begin(), subset.end(), 0);
    auto references = singlepp::build_indices(refs.get(), labels.data(), subset,
        [](size_t nr, size_t nc, const double* ptr) { 
            return std::shared_ptr<knncolle::Base<int, double> >(new knncolle::KmknnEuclidean<int, double>(nr, nc, ptr)); 
        },
        1
    );

    // Running the fine-tuning edge cases.
    singlepp::FineTuner ft;

    // Check early exit conditions.
    {
        std::vector<double> buffer(refs->ncol());
        auto vec = refs->dense_column()->fetch(0, buffer.data()); // doesn't really matter what we pick here.
        auto ranked = fill_ranks(refs->ncol(), vec);

        std::vector<double> scores { 0.2, 0.5, 0.1 };
        auto output = ft.run(ranked, references, markers, scores, 0.8, 0.05);
        EXPECT_EQ(output.first, 1);
        EXPECT_EQ(output.second, 0.3);

        std::fill(scores.begin(), scores.end(), 0.5);
        scores[0] = 0.51;
        output = ft.run(ranked, references, markers, scores, 1, 0.05);
        EXPECT_EQ(output.first, 0); // first entry of scores is maxed.
        EXPECT_FLOAT_EQ(output.second, 0.01);
    }

    // Check edge case when there is only a single label, 
    // based on the length of 'scores'.
    {
        auto vec = refs->dense_column()->fetch(1); // doesn't really matter
        auto ranked = fill_ranks(vec.size(), vec.data());

        std::vector<double> scores { 0.5 };
        auto output = ft.run(ranked, references, markers, scores, 0.8, 0.05);
        EXPECT_EQ(output.first, 0);
        EXPECT_TRUE(std::isnan(output.second));
    }

    // Checking that we eventually pick up the reference, if the input profile
    // is identical to one of the references. We set the quantile to 1 to
    // guarantee a score of 1 from a correlation of 1.
    auto wrk = refs->dense_column();
    for (size_t r = 0; r < nrefs; ++r) {
        auto vec = wrk->fetch(r);
        auto ranked = fill_ranks(vec.size(), vec.data());

        // Setting the template parameter test = true to force it to do
        // calculations; otherwise it would exit early with the top score.
        std::vector<double> scores { 0.5, 0.49, 0.48 };
        auto output = ft.run<true>(ranked, references, markers, scores, 1, 0.05);
        EXPECT_EQ(output.first, labels[r]);

        // Forcing it to match to some other label. 
        scores = std::vector<double>{ 0.5, 0.5, 0.5 };
        scores[labels[r]] = 0;
        auto output2 = ft.run<true>(ranked, references, markers, scores, 1, 0.05);
        EXPECT_NE(output2.first, labels[r]);
    }
}

TEST(FineTuneTest, Comparison) {
    // Mocking up the test and references.
    size_t ngenes = 200;
 
    size_t nlabels = 3;
    size_t nrefs = 50;
    auto refs = spawn_matrix(ngenes, nrefs, 200);
    auto labels = spawn_labels(nrefs, nlabels, 2000);

    auto markers = mock_markers(nlabels, 10, ngenes); 
    size_t ncells = 11;
    size_t seed = 12345;
    auto mat = spawn_matrix(ngenes, ncells, seed);
    
    // Naive calculation.
    size_t top = 5;
    auto subset = singlepp::subset_markers(markers, top);
    double quantile = 0.75;
    auto naive = naive_method(nlabels, labels, refs, mat, subset, quantile);

    // Recalculation inside the fine-tuner should give the same conclusion.
    singlepp::FineTuner ft;
    auto references = singlepp::build_indices(refs.get(), labels.data(), subset,
        [](size_t nr, size_t nc, const double* ptr) { 
            return std::shared_ptr<knncolle::Base<int, double> >(new knncolle::KmknnEuclidean<int, double>(nr, nc, ptr)); 
        },
        1
    );

    auto wrk = mat->dense_column(subset);
    for (size_t c = 0; c < mat->ncol(); ++c) {
        auto vec = wrk->fetch(c); 
        auto ranked = fill_ranks(vec.size(), vec.data());

        std::vector<double> scores;
        for (size_t l = 0; l < nlabels; ++l) {
            scores.push_back(naive.scores[l][c]);
        }

        // We use a huge threhold to ensure that everyone is in range. 
        // In this case, fine-tuning quits early and 'scores' is not mutated.
        auto output2 = ft.run(ranked, references, markers, scores, quantile, 100); 
        EXPECT_EQ(output2.first, naive.best[c]);
        EXPECT_EQ(output2.second, naive.delta[c]);

        // Everyone is still in range, but this time we set test = true to
        // force the calculations to be performed.  As all markers are still
        // used, we can cross-check the fine-tuning score calculations against
        // the naive method.
        auto output = ft.run<true>(ranked, references, markers, scores, quantile, 100); 
        EXPECT_EQ(output.first, naive.best[c]);
        EXPECT_TRUE(std::abs(naive.delta[c] - output.second) < 1e-6);

    }
}

TEST(FineTuneTest, Diagonal) {
    // Mocking up the test and references. This time, we make
    // sure that there are only markers on the diagonals.
    size_t ngenes = 200;
 
    size_t nlabels = 3;
    size_t nrefs = 50;
    auto refs = spawn_matrix(ngenes, nrefs, 200);
    auto labels = spawn_labels(nrefs, nlabels, 2000);

    auto markers = mock_markers_diagonal(nlabels, 10, ngenes); 

    // Mocking up the reference indices.
    std::vector<int> subset(ngenes);
    std::iota(subset.begin(), subset.end(), 0);
    auto references = singlepp::build_indices(refs.get(), labels.data(), subset,
        [](size_t nr, size_t nc, const double* ptr) { 
            return std::shared_ptr<knncolle::Base<int, double> >(new knncolle::KmknnEuclidean<int, double>(nr, nc, ptr)); 
        },
        1
    );

    // Running the fine-tuner, making sure we pick up the reference label.
    // To do so, we set the quantile to 1 to guarantee a score of 1 from a
    // correlation of 1. Again, this requires setting test = true to force
    // it to do calculations, otherwise it just quits early.
    singlepp::FineTuner ft;

    auto wrk = refs->dense_column();
    for (size_t r = 0; r < nrefs; ++r) {
        auto vec = wrk->fetch(r); 
        auto ranked = fill_ranks(vec.size(), vec.data());
        std::vector<double> scores { 0.49, 0.5, 0.48 };
        auto output = ft.run<true>(ranked, references, markers, scores, 1, 0.05);

        // The key point here is that the diagonals are actually used,
        // so we don't end up with an empty ranking vector and NaN scores.
        EXPECT_EQ(output.first, labels[r]);
        EXPECT_TRUE(output.second > 0);
    }
}
