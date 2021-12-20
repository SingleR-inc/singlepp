#include <gtest/gtest.h>

#include "singlepp/SinglePP.hpp"
#include "tatami/tatami.hpp"
#include "mock_markers.h"

#include <memory>
#include <vector>
#include <random>

class SinglePPTest : public ::testing::TestWithParam<std::tuple<int, double> > {
protected:    
    std::shared_ptr<tatami::Matrix<double, int> > spawn_matrix(size_t nr, size_t nc, int seed) {
        std::vector<double> contents(nr*nc);
        std::mt19937_64 rng(seed);
        std::normal_distribution<> dist;
        for (auto& c : contents) {
            c = dist(rng);
        }
        return std::shared_ptr<tatami::Matrix<double, int> >(new tatami::DenseColumnMatrix<double, int>(nr, nc, std::move(contents)));
    }
};

TEST_P(SinglePPTest, Simple) {
    auto param = GetParam();
    int top = std::get<0>(param);
    double quantile = std::get<1>(param);

    // Mocking up the test and references.
    size_t ngenes = 200;
    auto mat = spawn_matrix(ngenes, 5, 42);
 
    size_t nlabels = 3;
    std::vector<std::shared_ptr<tatami::Matrix<double, int> > > refs;
    for (size_t r = 0; r < nlabels; ++r) {
        refs.push_back(spawn_matrix(ngenes, (r + 1) * 5, r * 100));
    }

    std::vector<const tatami::Matrix<double, int>*> ref_ptrs; // TODO: replace this part.
    for (size_t r = 0; r < nlabels; ++r) {
        ref_ptrs.push_back(refs[r].get());
    }

    auto markers = mock_markers(nlabels, 50, ngenes); 

    // Running the implementation.
    singlepp::SinglePP runner;
    runner.set_fine_tune(false). set_top(top).set_quantile(quantile);
    auto output = runner.run(mat.get(), ref_ptrs, markers);

    // Implementing the reference score calculation.
    auto subset = singlepp::subset_markers(markers, top);

    for (size_t c = 0; c < mat->ncol(); ++c) {
        auto col = mat->column(c);
        singlepp::RankedVector vec;
        std::vector<double> scaled(subset.size());
        singlepp::scaled_ranks(col.data(), subset, vec, scaled.data());

        for (size_t r = 0; r < nlabels; ++r) {
            const auto& curref = refs[r];
            std::vector<double> correlations;

            for (size_t l = 0; l < curref->ncol(); ++l) {
                auto col2 = curref->column(l);
                std::vector<double> scaled2(subset.size());
                singlepp::scaled_ranks(col2.data(), subset, vec, scaled2.data());
                correlations.push_back(singlepp::distance_to_correlation(scaled.size(), scaled.data(), scaled2.data()));
            }

            double score = singlepp::correlations_to_scores(correlations, quantile);
            EXPECT_TRUE(std::abs(score - output.scores[r][c]) < 1e-6);
        }
    }
}

INSTANTIATE_TEST_CASE_P(
    SinglePP,
    SinglePPTest,
    ::testing::Combine(
        ::testing::Values(5, 10, 20), // nuber of top genes.
        ::testing::Values(0, 0.07, 0.2, 0.33) // quantile
    )
);
