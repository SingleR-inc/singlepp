#include <gtest/gtest.h>
#include "singlepp/fine_tune.hpp"
#include "mock_markers.h"
#include "spawn_matrix.h"

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
    auto mat = spawn_matrix(ngenes, 5, 42);
 
    size_t nlabels = 3;
    std::vector<std::shared_ptr<tatami::Matrix<double, int> > > refs;
    for (size_t r = 0; r < nlabels; ++r) {
        refs.push_back(spawn_matrix(ngenes, (r + 1) * 5, r * 100));
    }

    auto markers = mock_markers(nlabels, 10, ngenes); 

    // Mocking up the reference indices.
    std::vector<int> subset(ngenes);
    std::iota(subset.begin(), subset.end(), 0);
    auto references = singlepp::build_indices(subset, refs,
        [](size_t nr, size_t nc, const double* ptr) { 
            return std::shared_ptr<knncolle::Base<int, double> >(new knncolle::AnnoyEuclidean<int, double>(nr, nc, ptr)); 
        }
    );

    // Running the fine-tuning edge cases.
    singlepp::FineTuner ft;

    {
        std::vector<double> scores { 0.2, 0.5, 0.1 };
        auto vec = mat->column(0);
        auto output = ft.run(vec.data(), references, markers, scores, 0.8, 0.05);
        EXPECT_EQ(output.first, 1);
        EXPECT_EQ(output.second, 0.3);
    }

    {
        std::vector<double> scores { 0.5 };
        auto vec = mat->column(1);
        auto output = ft.run(vec.data(), references, markers, scores, 0.8, 0.05);
        EXPECT_EQ(output.first, 0);
        EXPECT_TRUE(std::isnan(output.second));
    }

    // Checking that we eventually pick up the reference, if the input profile
    // is identical to one of the references. We set the quantile to 1 to
    // guarantee a score of 1 from a correlation of 1; we can assume that the
    // delta will be pretty large in this case.
    for (size_t l = 0; l < nlabels; ++l) {
        std::vector<double> scores { 0.5, 0.49, 0.48 };
        auto vec = refs[l]->column(0);
        auto output = ft.run(vec.data(), references, markers, scores, 1, 0.05);
        EXPECT_EQ(output.first, l);
        EXPECT_TRUE(output.second > 0.5);

        // Forcing it to match to some other label. In this case, the delta
        // is probably going to be pretty small because the matches are all crap.
        scores = std::vector<double>{ 0.5, 0.5, 0.5 };
        scores[l] = 0;
        auto output2 = ft.run(vec.data(), references, markers, scores, 1, 0.05);
        EXPECT_NE(output2.first, l);
        EXPECT_TRUE(output2.second < 0.5);
    }
}
