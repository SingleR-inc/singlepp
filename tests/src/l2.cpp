#include <gtest/gtest.h>

#include "singlepp/l2.hpp"
#include "singlepp/build_reference.hpp"

TEST(ComputeL2, Dense) {
    std::vector<double> a{ 1.2, -0.5, 2.3, 5.6, -4.4 };
    std::vector<double> b{ -.2, -5.1, 4.4, 1.6,  0.2 };

    const int N = b.size();
    singlepp::RankedVector<double, int> paired_a, paired_b;
    for (int i = 0; i < N; ++i) {
        paired_a.emplace_back(a[i], i);
        paired_b.emplace_back(b[i], i);
    }
    std::sort(paired_a.begin(), paired_a.end());
    std::sort(paired_b.begin(), paired_b.end());

    std::vector<double> scaled_a(N), scaled_b(N);
    singlepp::scaled_ranks_dense(N, paired_a, scaled_a.data());
    singlepp::scaled_ranks_dense(N, paired_b, scaled_b.data());
    auto expected = singlepp::dense_l2(N, scaled_a.data(), scaled_b.data());

    std::vector<double> buffer_b(N);
    auto scaled = singlepp::scaled_ranks_dense_l2(static_cast<int>(a.size()), scaled_a.data(), paired_b, buffer_b.data());

    // Small differences may be present due to changes in the summation order.
    EXPECT_FLOAT_EQ(scaled, expected);
}

TEST(ComputeL2, Densify) {
    std::vector<double> x{ -1, -.2, -1, -1, -5.1, -1, 4.4, -1, 1.6 };

    std::vector<int> indices;
    std::vector<double> values;
    singlepp::CompressedSparseVector<int, double> sparse_x;
    sparse_x.zero = -1;

    for (std::size_t i = 0; i < x.size(); ++i) {
        if (x[i] != sparse_x.zero) {
            indices.push_back(i);
            values.push_back(x[i]);
        }
    }

    sparse_x.number = indices.size();
    sparse_x.index = indices.data();
    sparse_x.value = values.data();

    std::vector<double> xcopy(x.size());
    singlepp::densify_sparse_vector<int>(x.size(), sparse_x, xcopy);
    EXPECT_EQ(x, xcopy);
}

TEST(ComputeL2, SparseEasy) {
    std::vector<double> a{ 3.2, 1.4,  0, -.5,    0, 2.3,   0,  0, -2.4 };
    std::vector<double> b{   0, -.2,  0,   0, -5.1,   0, 4.4,  0,  1.6 };

    const int N = a.size();
    singlepp::RankedVector<double, int> paired_a, paired_b;
    for (int i = 0; i < N; ++i) {
        paired_a.emplace_back(a[i], i);
        paired_b.emplace_back(b[i], i);
    }
    std::sort(paired_a.begin(), paired_a.end());
    std::sort(paired_b.begin(), paired_b.end());

    std::vector<double> scaled_a(N), scaled_b(N);
    singlepp::scaled_ranks_dense(N, paired_a, scaled_a.data());
    singlepp::scaled_ranks_dense(N, paired_b, scaled_b.data());
    const auto expected = singlepp::dense_l2(N, scaled_a.data(), scaled_b.data());

    // Checking that it works with SparseScaled.
    singlepp::RankedVector<double, int> negative_b, positive_b;
    for (const auto& bp : paired_b) {
        if (bp.first < 0) {
            negative_b.push_back(bp);
        } else if (bp.first > 0) {
            positive_b.push_back(bp);
        }
    }

    singlepp::SparseScaled<int, double> sparse_scaled_b;
    singlepp::scaled_ranks_sparse(N, negative_b, positive_b, sparse_scaled_b);
    auto sparse_l2 = singlepp::sparse_l2(N, scaled_a.data(), true, sparse_scaled_b);
    EXPECT_FLOAT_EQ(sparse_l2, expected);

    // Checking that it works with compressed sparse vectors.
    std::vector<int> b_indices;
    std::vector<double> b_values;
    singlepp::CompressedSparseVector<int, double> csv_b;
    csv_b.zero = sparse_scaled_b.zero;

    std::sort(sparse_scaled_b.nonzero.begin(), sparse_scaled_b.nonzero.end());
    for (const auto& nz : sparse_scaled_b.nonzero) {
        b_indices.push_back(nz.first);
        b_values.push_back(nz.second);
    }

    csv_b.number = sparse_scaled_b.nonzero.size();
    csv_b.index = b_indices.data();
    csv_b.value = b_values.data();

    auto sparse_l2_again = singlepp::sparse_l2(N, scaled_a.data(), true, csv_b);
    EXPECT_FLOAT_EQ(sparse_l2_again, expected);

    // Checking direct calculation of distances.
    std::vector<std::pair<int, double> > buffer_b;
    auto direct = singlepp::scaled_ranks_sparse_l2(N, scaled_a.data(), true, negative_b, positive_b, buffer_b);
    EXPECT_FLOAT_EQ(direct, expected);
}

TEST(ComputeL2, SparseEmpty) {
    std::vector<double> a{ 3.2, 1.4,  0, -.5,    0, 2.3,   0,  0, -2.4 };
    const int N = a.size();
    std::vector<double> empty(N);

    singlepp::RankedVector<double, int> paired_a;
    for (int i = 0; i < N; ++i) {
        paired_a.emplace_back(a[i], i);
    }
    std::sort(paired_a.begin(), paired_a.end());

    std::vector<double> scaled_a(N);
    singlepp::scaled_ranks_dense(N, paired_a, scaled_a.data());
    const auto expected = singlepp::dense_l2(a.size(), scaled_a.data(), empty.data());

    singlepp::SparseScaled<int, double> sparse_empty;
    auto sparse_l2 = singlepp::sparse_l2(N, scaled_a.data(), true, sparse_empty);
    EXPECT_FLOAT_EQ(sparse_l2, expected);

    // Now flipping it around so that the query is empty.
    singlepp::RankedVector<double, int> negative_a, positive_a;
    for (const auto& ap : paired_a) {
        if (ap.first < 0) {
            negative_a.push_back(ap);
        } else if (ap.first > 0) {
            positive_a.push_back(ap);
        }
    }

    singlepp::SparseScaled<int, double> sparse_scaled_a;
    singlepp::scaled_ranks_sparse(N, negative_a, positive_a, sparse_scaled_a);
    auto sparse_l2_again = singlepp::sparse_l2(N, empty.data(), false, sparse_scaled_a);
    EXPECT_FLOAT_EQ(sparse_l2_again, expected);

    // Checking direct calculation of distances.
    std::vector<std::pair<int, double> > buffer;
    singlepp::RankedVector<double, int> ranked_empty;
    auto direct = singlepp::scaled_ranks_sparse_l2(N, scaled_a.data(), true, ranked_empty, ranked_empty, buffer);
    EXPECT_FLOAT_EQ(direct, expected);

    direct = singlepp::scaled_ranks_sparse_l2(N, empty.data(), false, negative_a, positive_a, buffer);
    EXPECT_FLOAT_EQ(direct, expected);
}

TEST(ComputeL2, SparseSparseHard) {
    std::mt19937_64 rng(12345);
    std::normal_distribution<> ndist;
    std::uniform_real_distribution<> udist;
    const int n = 57;
    std::vector<double> remapping(n);

    singlepp::RankedVector<double, int> paired_a, paired_b, full_paired_b;
    std::vector<double> scaled_a(n), scaled_b(n);
    singlepp::RankedVector<double, int> negative_b, positive_b;
    std::vector<int> b_indices;
    std::vector<double> b_values;
    singlepp::SparseScaled<int, double> sparse_scaled_b;
    std::vector<std::pair<int, double> > buffer_b;

    for (int it = 0; it < 100; ++it) {
        paired_a.clear();
        paired_b.clear();
        full_paired_b.clear();
        for (int i = 0; i < n; ++i) {
            paired_a.emplace_back(ndist(rng), i);
            if (udist(rng) <= 0.25) {
                paired_b.emplace_back(ndist(rng), i);
                full_paired_b.push_back(paired_b.back());
            } else {
                full_paired_b.emplace_back(0, i);
            }
        }
        std::sort(paired_a.begin(), paired_a.end());
        std::sort(paired_b.begin(), paired_b.end());
        std::sort(full_paired_b.begin(), full_paired_b.end());

        bool a_nonempty = singlepp::scaled_ranks_dense(n, paired_a, scaled_a.data());
        singlepp::scaled_ranks_dense(n, full_paired_b, scaled_b.data());
        auto expected = singlepp::dense_l2(n, scaled_a.data(), scaled_b.data());

        // Computing with the SparseScaled. 
        negative_b.clear();
        positive_b.clear();
        for (const auto& bp : paired_b) {
            if (bp.first < 0) {
                negative_b.push_back(bp);
            } else if (bp.first > 0) {
                positive_b.push_back(bp);
            }
        }

        singlepp::scaled_ranks_sparse(n, negative_b, positive_b, sparse_scaled_b);
        auto sparse_l2 = singlepp::sparse_l2(n, scaled_a.data(), a_nonempty, sparse_scaled_b);
        EXPECT_FLOAT_EQ(sparse_l2, expected);

        // Computing with the CompressedSparseVector. 
        b_indices.clear();
        b_values.clear();
        singlepp::CompressedSparseVector<int, double> csv_b;
        csv_b.zero = sparse_scaled_b.zero;

        std::sort(sparse_scaled_b.nonzero.begin(), sparse_scaled_b.nonzero.end());
        for (const auto& nz : sparse_scaled_b.nonzero) {
            b_indices.push_back(nz.first);
            b_values.push_back(nz.second);
        }

        csv_b.number = sparse_scaled_b.nonzero.size();
        csv_b.index = b_indices.data();
        csv_b.value = b_values.data();

        auto sparse_l2_again = singlepp::sparse_l2(n, scaled_a.data(), a_nonempty, csv_b);
        EXPECT_FLOAT_EQ(sparse_l2_again, expected);

        // Finally, checking the direct calculation.
        auto direct = singlepp::scaled_ranks_sparse_l2(n, scaled_a.data(), a_nonempty, negative_b, positive_b, buffer_b);
        EXPECT_FLOAT_EQ(direct, expected);
    }
}
