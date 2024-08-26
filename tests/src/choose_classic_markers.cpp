#include <gtest/gtest.h>

#include "singlepp/choose_classic_markers.hpp"
#include "tatami/tatami.hpp"
#include "spawn_matrix.h"

class ChooseClassicMarkersTest : public ::testing::TestWithParam<std::tuple<int, bool> > {};

TEST_P(ChooseClassicMarkersTest, Simple) { 
    size_t ngenes = 100;
    size_t nlabels = 10;
    auto params = GetParam();
    int requested = std::get<0>(params);
    bool reverse = std::get<1>(params);

    auto mat = spawn_matrix(ngenes, nlabels, 1234 * requested);
    std::vector<int> groupings(nlabels);
    std::iota(groupings.begin(), groupings.end(), 0);
    if (reverse) { // reversing the label order to test the indexing.
        std::reverse(groupings.begin(), groupings.end());
    }

    singlepp::ChooseClassicMarkersOptions mopt;
    mopt.number = requested;
    auto output = singlepp::choose_classic_markers(*mat, groupings.data(), mopt);
    auto lwrk = mat->dense_column();
    auto rwrk = mat->dense_column();
    std::vector<double> lbuffer(mat->nrow()), rbuffer(mat->nrow());

    // Comparing against a naive implementation.
    EXPECT_EQ(output.size(), nlabels);
    for (size_t l = 0; l < nlabels; ++l) {
        const auto& current = output[l];
        EXPECT_EQ(current.size(), nlabels);
        auto left = lwrk->fetch(reverse ? nlabels - l - 1 : l, lbuffer.data());

        for (size_t l2 = 0; l2 < nlabels; ++l2) {
            const auto& markers = current[l2];
            int nmarkers = markers.size();

            if (l == l2) {
                EXPECT_TRUE(markers.empty());
                continue;
            } else if (requested >= 0) {
                EXPECT_TRUE(nmarkers <= requested);
            }

            auto right = rwrk->fetch(reverse ? nlabels - l2 - 1 : l2, rbuffer.data());
            std::vector<std::pair<double, int> > sorted;
            for (size_t i = 0; i < ngenes; ++i) {
                sorted.emplace_back(right[i] - left[i], i);
            }

            std::sort(sorted.begin(), sorted.end());
            for (int g = 0; g < nmarkers; ++g) {
                EXPECT_TRUE(sorted[g].first < 0);
                EXPECT_EQ(sorted[g].second, markers[g]);
            }

            if (nmarkers < requested && nmarkers + 1 < static_cast<int>(ngenes)) {
                EXPECT_TRUE(sorted[nmarkers].first >= 0); // early rejection.
            }
        }
    }

    // Same result when parallelized.
    mopt.num_threads = 3;
    auto outputp = singlepp::choose_classic_markers(*mat, groupings.data(), mopt);
    for (size_t l = 0; l < nlabels; ++l) {
        const auto& serial = output[l];
        const auto& parallel = outputp[l];
        for (size_t l2 = 0; l2 < nlabels; ++l2) {
            EXPECT_EQ(serial[l2], parallel[l2]);
        }
    }
}

TEST_P(ChooseClassicMarkersTest, Blocked) { 
    size_t ngenes = 100;
    size_t nlabels = 10;
    auto params = GetParam();
    int requested = std::get<0>(params);
    bool reverse = std::get<1>(params);

    auto mat1 = spawn_matrix(ngenes, nlabels, 1234 * requested);
    auto mat2 = spawn_matrix(ngenes, nlabels, 5678 * requested);

    std::vector<int> groupings(nlabels);
    std::iota(groupings.begin(), groupings.end(), 0);
    if (reverse) {
        std::reverse(groupings.begin(), groupings.end());
    }

    singlepp::ChooseClassicMarkersOptions mopt;
    mopt.number = requested;
    auto output = singlepp::choose_classic_markers(
        std::vector<const tatami::Matrix<double, int>*>{ mat1.get(), mat2.get() },
        std::vector<const int*>{ groupings.data(), groupings.data() },
        mopt
    );

    // Summing the two matrices together.
    std::vector<double> combined(ngenes * nlabels);
    auto wrk1 = mat1->dense_column();
    auto wrk2 = mat2->dense_column();
    auto cIt = combined.begin();
    std::vector<double> buffer1(mat1->nrow()), buffer2(mat2->nrow());

    for (size_t l = 0; l < nlabels; ++l) {
        auto col1 = wrk1->fetch(l, buffer1.data());
        auto col2 = wrk2->fetch(l, buffer2.data());
        for (size_t g = 0; g < ngenes; ++g, ++cIt) {
            *cIt = col1[g] + col2[g];
        }
    }
    tatami::DenseColumnMatrix<double, int> comb_mat(ngenes, nlabels, std::move(combined));
    auto ref = choose_classic_markers(comb_mat, groupings.data(), mopt);

    ASSERT_EQ(output.size(), ref.size());
    ASSERT_EQ(output.size(), nlabels);
    for (size_t l = 0; l < nlabels; ++l) {
        ASSERT_EQ(output[l].size(), ref[l].size());
        ASSERT_EQ(output[l].size(), nlabels);

        for (size_t l2 = 0; l2 < nlabels; ++l2) {
            EXPECT_EQ(output[l][l2], ref[l][l2]);
        }
    }
}

TEST_P(ChooseClassicMarkersTest, BlockedMissing) { 
    size_t ngenes = 100;
    size_t nlabels = 10;
    auto params = GetParam();
    int requested = std::get<0>(params);
    bool reverse = std::get<1>(params);

    singlepp::ChooseClassicMarkersOptions mopt;
    mopt.number = requested;

    // Label 0 is missing from the first matrix.
    auto mat1 = spawn_matrix(ngenes, nlabels - 1, 1234 * requested);
    auto mat2 = spawn_matrix(ngenes, nlabels, 5678 * requested);

    std::vector<int> groupings1(nlabels - 1);
    std::iota(groupings1.begin(), groupings1.end(), 1);
    if (reverse) {
        std::reverse(groupings1.begin(), groupings1.end());
    }

    std::vector<int> groupings2(nlabels);
    std::iota(groupings2.begin(), groupings2.end(), 0);

    // Creating the two references. We remove label 0 (i.e., the first column) from mat2 to 
    // match the groups available in mat1; we also just compute markers from mat2 alone.
    auto sub = tatami::make_DelayedSubsetBlock(mat2, 1, static_cast<int>(nlabels) - 1, false);
    auto ref_comb = singlepp::choose_classic_markers(
        std::vector<tatami::Matrix<double, int>*>{ mat1.get(), sub.get() }, // non-const, to test the overload.
        std::vector<const int*>{ groupings1.data(), groupings2.data() + 1 },
        mopt
    );               
    
    auto ref_solo = singlepp::choose_classic_markers(*mat2, groupings2.data(), mopt);

    // Comparing them.
    auto output = singlepp::choose_classic_markers(
        std::vector<const tatami::Matrix<double, int>*>{ mat1.get(), mat2.get() },
        std::vector<const int*>{ groupings1.data(), groupings2.data() },
        mopt
    );

    ASSERT_EQ(ref_comb.size(), output.size());
    ASSERT_EQ(ref_solo.size(), output.size()); 
    ASSERT_EQ(ref_comb.size(), nlabels);

    for (size_t l = 0; l < nlabels; ++l) {
        ASSERT_EQ(ref_comb[l].size(), output[l].size());
        ASSERT_EQ(ref_solo[l].size(), output[l].size());
        ASSERT_EQ(output[l].size(), nlabels);

        for (size_t l2 = 0; l2 < nlabels; ++l2) {
            auto& ref = ((l2 == 0 || l == 0) ? ref_solo : ref_comb);
            EXPECT_EQ(output[l][l2], ref[l][l2]);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    ChooseClassicMarkers,
    ChooseClassicMarkersTest,
    ::testing::Combine(
        ::testing::Values(-1, 5, 10, 20, 100, 1000), // number of top genes.
        ::testing::Values(false, true) // whether to reverse the label order
    )
);
