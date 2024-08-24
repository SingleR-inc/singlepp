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

    inline static std::shared_ptr<tatami::Matrix<double, int> > reference;
    inline static std::vector<int> labels;
    inline static std::vector<singlepp::internal::PerLabelReference<int, double> > indices;

    static void SetUpTestSuite() {
        reference = spawn_matrix(ngenes, nprofiles, /* seed = */ 200);
        labels = spawn_labels(nprofiles, nlabels, /* seed = */ 2000);

        // Mocking up the reference indices.
        std::vector<int> subset(ngenes);
        std::iota(subset.begin(), subset.end(), 0);
        indices = singlepp::internal::build_indices(*reference, labels.data(), subset, knncolle::VptreeBuilder(), 1);
    }
};

TEST_F(FineTuneSingleTest, EdgeCases) {
    auto markers = mock_markers<int>(nlabels, 10, ngenes); 
    singlepp::internal::FineTuneSingle<int, int, double, double> ft;
    singlepp::internal::RankedVector<double, int> placeholder;

    // Check early exit conditions, when there is one clear winner or all of
    // the labels are equal (i.e., no contraction of the feature space).
    {
        std::vector<double> scores { 0.2, 0.5, 0.1 };
        auto output = ft.run(placeholder, indices, markers, scores, 0.8, 0.05);
        EXPECT_EQ(output.first, 1);
        EXPECT_EQ(output.second, 0.3);

        std::fill(scores.begin(), scores.end(), 0.5);
        scores[0] = 0.51;
        output = ft.run(placeholder, indices, markers, scores, 1, 0.05);
        EXPECT_EQ(output.first, 0); // first entry of scores is maxed.
        EXPECT_FLOAT_EQ(output.second, 0.01);
    }

    // Check edge case when there is only a single label, based on the length of 'scores'.
    {
        std::vector<double> scores { 0.5 };
        auto output = ft.run(placeholder, indices, markers, scores, 0.8, 0.05);
        EXPECT_EQ(output.first, 0);
        EXPECT_TRUE(std::isnan(output.second));
    }
}

TEST_F(FineTuneSingleTest, ExactRecovery) {
    auto markers = mock_markers<int>(nlabels, 10, ngenes); 
    singlepp::internal::FineTuneSingle<int, int, double, double> ft;

    // Checking that we eventually pick up the reference, if the input profile
    // is identical to one of the profiles. We set the quantile to 1 to
    // guarantee a score of 1 from a correlation of 1.
    auto wrk = reference->dense_column();
    std::vector<double> buffer(ngenes);
    for (size_t r = 0; r < nprofiles; ++r) {
        auto vec = wrk->fetch(r, buffer.data());
        auto ranked = fill_ranks<int>(ngenes, vec);

        std::vector<double> scores { 0.5, 0.49, 0.48 };
        scores[(labels[r] + 1) % nlabels] = 0; // forcing another label to be zero so that it actually does the fine-tuning.
        auto output = ft.run(ranked, indices, markers, scores, 1, 0.05);
        EXPECT_EQ(output.first, labels[r]);

        scores = std::vector<double>{ 0.5, 0.5, 0.5 };
        scores[labels[r]] = 0; // forcing it to match to some other label. 
        auto output2 = ft.run(ranked, indices, markers, scores, 1, 0.05);
        EXPECT_NE(output2.first, labels[r]);
    }
}

TEST_F(FineTuneSingleTest, Diagonal) {
    // This time there are only markers on the diagonals.
    auto markers = mock_markers_diagonal<int>(nlabels, 10, ngenes); 
    singlepp::internal::FineTuneSingle<int, int, double, double> ft;

    auto wrk = reference->dense_column();
    std::vector<double> buffer(ngenes);
    for (size_t r = 0; r < nprofiles; ++r) {
        auto vec = wrk->fetch(r, buffer.data()); 
        auto ranked = fill_ranks<int>(ngenes, vec);

        std::vector<double> scores { 0.49, 0.5, 0.48 };
        scores[(labels[r] + 1) % nlabels] = 0; // forcing another label to be zero so that it actually does the fine-tuning.
        auto output = ft.run(ranked, indices, markers, scores, 1, 0.05);

        // The key point here is that the diagonals are actually used,
        // so we don't end up with an empty ranking vector and NaN scores.
        EXPECT_EQ(output.first, labels[r]);
        EXPECT_TRUE(output.second > 0);
    }
}
