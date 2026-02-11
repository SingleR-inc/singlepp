#include <gtest/gtest.h>
#include "singlepp/annotate_cells_single.hpp"
#include "mock_markers.h"
#include "spawn_matrix.h"
#include "fill_ranks.h"
#include "naive_method.h"

class FineTuneSingleTest : public ::testing::Test {
protected:
    inline static size_t ngenes = 200;
    inline static size_t nlabels = 3;
    inline static size_t nprofiles = 50;
    inline static size_t nmarkers = 100;

    inline static std::shared_ptr<tatami::Matrix<double, int> > reference;
    inline static std::vector<int> labels;
    inline static singlepp::BuiltReference<int, double> built;
    inline static std::vector<int> subset;

    static void SetUpTestSuite() {
        reference = spawn_matrix(ngenes, nprofiles, /* seed = */ 200);
        labels = spawn_labels(nprofiles, nlabels, /* seed = */ 2000);

        // Mocking up the reference indices.
        subset.reserve(nmarkers);
        for (size_t i = 0; i < nmarkers; ++i) {
            subset.push_back(i * 2);
        }

        built = singlepp::build_reference<double>(*reference, labels.data(), subset, 1);
    }
};

TEST_F(FineTuneSingleTest, EdgeCases) {
    auto markers = mock_markers<int>(nlabels, 10, ngenes); 
    singlepp::FineTuneSingle<false, false, int, int, double, double> ft(built.num_markers, *(built.dense));
    singlepp::RankedVector<double, int> placeholder;

    // Check early exit conditions, when there is one clear winner or all of
    // the labels are equal (i.e., no contraction of the feature space).
    {
        std::vector<double> scores { 0.2, 0.5, 0.1 };
        auto output = ft.run(placeholder, *(built.dense), markers, scores, 0.8, 0.05);
        EXPECT_EQ(output.first, 1);
        EXPECT_EQ(output.second, 0.3);

        std::fill(scores.begin(), scores.end(), 0.5);
        scores[0] = 0.51;
        output = ft.run(placeholder, *(built.dense), markers, scores, 1, 0.05);
        EXPECT_EQ(output.first, 0); // first entry of scores is maxed.
        EXPECT_FLOAT_EQ(output.second, 0.01);
    }

    // Check edge case when there is only a single label, based on the length of 'scores'.
    {
        std::vector<double> scores { 0.5 };
        auto output = ft.run(placeholder, *(built.dense), markers, scores, 0.8, 0.05);
        EXPECT_EQ(output.first, 0);
        EXPECT_TRUE(std::isnan(output.second));
    }
}

TEST_F(FineTuneSingleTest, ExactRecovery) {
    auto markers = mock_markers<int>(nlabels, 10, nmarkers); 
    singlepp::FineTuneSingle<false, false, int, int, double, double> ft(nmarkers, *(built.dense));

    // Checking that we eventually pick up the reference, if the input profile
    // is identical to one of the profiles. We set the quantile to 1 to
    // guarantee a score of 1 from a correlation of 1.
    auto wrk = reference->dense_column(subset);
    std::vector<double> buffer(nmarkers);
    for (size_t r = 0; r < nprofiles; ++r) {
        auto vec = wrk->fetch(r, buffer.data());
        auto ranked = fill_ranks<int>(nmarkers, vec);

        std::vector<double> scores { 0.5, 0.49, 0.48 };
        scores[(labels[r] + 1) % nlabels] = 0; // forcing another label to be zero so that it actually does the fine-tuning.
        auto output = ft.run(ranked, *(built.dense), markers, scores, 1, 0.05);
        EXPECT_EQ(output.first, labels[r]);

        scores = std::vector<double>{ 0.5, 0.5, 0.5 };
        scores[labels[r]] = 0; // forcing it to match to some other label. 
        auto output2 = ft.run(ranked, *(built.dense), markers, scores, 1, 0.05);
        EXPECT_NE(output2.first, labels[r]);
    }
}

TEST_F(FineTuneSingleTest, Diagonal) {
    // This time there are only markers on the diagonals.
    auto markers = mock_markers_diagonal<int>(nlabels, 10, nmarkers); 
    singlepp::FineTuneSingle<false, false, int, int, double, double> ft(nmarkers, *(built.dense));

    auto wrk = reference->dense_column(subset);
    std::vector<double> buffer(nmarkers);
    for (size_t r = 0; r < nprofiles; ++r) {
        auto vec = wrk->fetch(r, buffer.data()); 
        auto ranked = fill_ranks<int>(nmarkers, vec);

        std::vector<double> scores { 0.49, 0.5, 0.48 };
        scores[(labels[r] + 1) % nlabels] = 0; // forcing another label to be zero so that it actually does the fine-tuning.
        auto output = ft.run(ranked, *(built.dense), markers, scores, 1, 0.05);

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
    size_t nmarkers = 105;

    auto markers = mock_markers<int>(nlabels, 10, nmarkers); 
    auto labels = spawn_labels(nprofiles, nlabels, /* seed = */ 2000);
    std::vector<int> subset;
    subset.reserve(nmarkers);
    for (size_t i = 0; i < nmarkers; ++i) {
        subset.push_back(i * 2);
    }

    auto new_reference = spawn_sparse_matrix(ngenes, nprofiles, /* seed = */ 300, /* density = */ 0.2);
    auto new_built = singlepp::build_reference<double>(*new_reference, labels.data(), subset, 1);
    singlepp::FineTuneSingle<false, false, int, int, double, double> new_ft(nmarkers, *(new_built.dense));
    singlepp::FineTuneSingle<true, false, int, int, double, double> new_ft2(nmarkers, *(new_built.dense));

    auto sparse_reference = tatami::convert_to_compressed_sparse<double, int>(*new_reference, true, {});
    auto sparse_built = singlepp::build_reference<double>(*sparse_reference, labels.data(), subset, 1);
    singlepp::FineTuneSingle<false, true, int, int, double, double> sparse_ft(nmarkers, *(sparse_built.sparse));
    singlepp::FineTuneSingle<true, true, int, int, double, double> sparse_ft2(nmarkers, *(sparse_built.sparse));

    const int ntest = 100; 
    auto new_test = spawn_sparse_matrix(ngenes, ntest, /* seed = */ 302, /* density = */ 0.2);

    auto wrk = new_test->dense_column(subset);
    std::vector<double> buffer(nmarkers);
    for (int t = 0; t < ntest; ++t) {
        auto vec = wrk->fetch(t, buffer.data()); 
        auto ranked = fill_ranks<int>(nmarkers, vec);

        std::vector<double> scores(nlabels, 0.5);
        scores[t % nlabels] = 0; // forcing one of the labels to be zero so that it actually does the fine-tuning.

        auto score_copy = scores;
        auto output = new_ft.run(ranked, *(new_built.dense), markers, score_copy, 1, 0.05);

        score_copy = scores;
        auto sparse_output = sparse_ft.run(ranked, *(sparse_built.sparse), markers, score_copy, 1, 0.05);
        EXPECT_EQ(output.first, sparse_output.first);
        EXPECT_FLOAT_EQ(output.second, sparse_output.second);

        singlepp::RankedVector<double, int> sparse_ranked;
        for (auto r : ranked) {
            if (r.first) {
                sparse_ranked.push_back(r);
            }
        }

        score_copy = scores;
        auto output2 = new_ft2.run(sparse_ranked, *(new_built.dense), markers, score_copy, 1, 0.05);
        EXPECT_EQ(output.first, output2.first);
        EXPECT_FLOAT_EQ(output.second, output2.second);

        score_copy = scores;
        auto sparse_output2 = sparse_ft2.run(sparse_ranked, *(sparse_built.sparse), markers, score_copy, 1, 0.05);
        EXPECT_EQ(output.first, sparse_output2.first);
        EXPECT_FLOAT_EQ(output.second, sparse_output2.second);
    }
}
