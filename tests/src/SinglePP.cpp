#include <gtest/gtest.h>

#include "singlepp/SinglePP.hpp"
#include "tatami/tatami.hpp"
#include "mock_markers.h"

#include <memory>
#include <vector>
#include <random>

std::shared_ptr<tatami::Matrix<double, int> > spawn_matrix(size_t nr, size_t nc, int seed) {
    std::vector<double> contents(nr*nc);
    std::mt19937_64 rng(seed);
    std::normal_distribution<> dist;
    for (auto& c : contents) {
        c = dist(rng);
    }
    return std::shared_ptr<tatami::Matrix<double, int> >(new tatami::DenseColumnMatrix<double, int>(nr, nc, std::move(contents)));
}

class SinglePPSimpleTest : public ::testing::TestWithParam<std::tuple<int, double> > {};

TEST_P(SinglePPSimpleTest, Simple) {
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
    runner.set_fine_tune(false).set_top(top).set_quantile(quantile);
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
    SinglePPSimpleTest,
    ::testing::Combine(
        ::testing::Values(5, 10, 20), // nuber of top genes.
        ::testing::Values(0, 0.07, 0.2, 0.33) // quantile
    )
);

class SinglePPIntersectTest : public ::testing::TestWithParam<std::tuple<int, double, double> > {};

TEST_P(SinglePPIntersectTest, Intersect) {
    auto param = GetParam();
    int top = std::get<0>(param);
    double quantile = std::get<1>(param);
    double prop = std::get<2>(param);

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

    auto markers = mock_markers(nlabels, ngenes, ngenes); 

    // Creating overlapping ID vectors.
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

    // Computing the observed result.
    singlepp::SinglePP runner;
    runner.set_fine_tune(false).set_top(top).set_quantile(quantile);
    auto result = runner.run(mat.get(), left.data(), ref_ptrs, right.data(), markers);

    // Computing the reference result. We try not to use subset_markers
    auto intersection = singlepp::intersect_features(left.size(), left.data(), right.size(), right.data());
    auto markers2 = markers;
    singlepp::subset_markers(intersection, markers2, top);
    auto pairs = singlepp::unzip(intersection);
    auto submat = tatami::make_DelayedSubset<0>(mat, pairs.first);
    
    std::cout << "Subsetted: " << pairs.second.size() << "\t";
    for (int i = 0; i < 10; ++i) {
        std::cout << pairs.second[i] << ", ";
    }
    std::cout << std::endl;

    std::vector<std::shared_ptr<tatami::Matrix<double, int> > > subrefs(nlabels);
    std::vector<const tatami::Matrix<double, int>*> subref_ptrs(nlabels);
    for (size_t s = 0; s < nlabels; ++s) {
        subrefs[s] = tatami::make_DelayedSubset<0>(refs[s], pairs.second);
        subref_ptrs[s] = subrefs[s].get();
    }

//    std::unordered_map<int, int> locations;
//    for (size_t i = 0; i < pairs.second.size(); ++i) {
//        locations[pairs.second[i]] = i;
//    }
//
//    for (size_t i = 0; i < nlabels; ++i) {
//        for (size_t j = 0; j < nlabels; ++j) {
//            if (i == j) {
//                continue;
//            }
//
//            std::vector<int> current;
//            for (auto s : markers[i][j]) {
//                auto it = locations.find(s);
//                if (it != locations.end()) {
//                    current.push_back(it->second);
//                }
//            }
//            markers2[i][j] = current;
//        }
//    }

    auto result2 = runner.run(submat.get(), subref_ptrs, markers2);
    EXPECT_EQ(result2.scores[0], result.scores[0]);
    EXPECT_EQ(result2.best, result.best);
    EXPECT_EQ(result2.delta, result.delta);
}

INSTANTIATE_TEST_CASE_P(
    SinglePP,
    SinglePPIntersectTest,
    ::testing::Combine(
        ::testing::Values(1000), // nuber of top genes.
        ::testing::Values(0), // quantile
        ::testing::Values(1) // proportion subset
//        ::testing::Values(5, 10, 20), // nuber of top genes.
//        ::testing::Values(0, 0.2, 0.33), // quantile
//        ::testing::Values(0.5, 0.9) // proportion subset
    )
);


