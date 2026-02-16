#include <gtest/gtest.h>

#include "singlepp/l2.hpp"
#include "singlepp/build_reference.hpp"

TEST(ComputeL2, DenseDense) {
    std::vector<double> a{ 1.2, -0.5, 2.3, 5.6 };
    std::vector<double> b{ -.2, -5.1, 4.4, 1.6 };

    double expected = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const auto delta = a[i] - b[i];
        expected += delta * delta;
    }

    EXPECT_EQ(singlepp::dense_l2(a.size(), a.data(), b.data()), expected);
    EXPECT_EQ(singlepp::dense_l2(a.size(), a.data(), b.data()), expected);
}

singlepp::SparseScaled<int, double> dense_to_sparse_scaled(const std::vector<double>& x, double val) {
    singlepp::SparseScaled<int, double> output;
    output.zero = val;
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (x[i] != val) {
            output.nonzero.emplace_back(i, x[i]);
        }
    }
    return output;
}

singlepp::CompressedSparseVector<int, double> dense_to_compressed_sparse(const std::vector<double>& x, double val, std::vector<int>& indices, std::vector<double>& values) {
    singlepp::CompressedSparseVector<int, double> output;
    output.zero = val;
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (x[i] != val) {
            indices.push_back(i);
            values.push_back(x[i]);
        }
    }
    output.number = indices.size();
    output.index = indices.data();
    output.value = values.data();
    return output;
}

TEST(ComputeL2, Densify) {
    std::vector<double> b{  -1, -.2,   -1,  -1, -5.1,  -1, 4.4,   -1,  1.6 };

    // Sparse scaled.
    {
        auto scaled_b = dense_to_sparse_scaled(b, -1);
        std::vector<double> bcopy(b.size());
        singlepp::densify_sparse_vector<int>(b.size(), scaled_b, bcopy);
        EXPECT_EQ(b, bcopy);
    }

    // Compressed sparse.
    {
        std::vector<int> indices;
        std::vector<double> values;
        auto sparse_b = dense_to_compressed_sparse(b, -1, indices, values);
        std::vector<double> bcopy(b.size());
        singlepp::densify_sparse_vector<int>(b.size(), sparse_b, bcopy);
        EXPECT_EQ(b, bcopy);
    }
}

TEST(ComputeL2, SparseSparseEasy) {
    std::vector<double> a{ 3.2, 1.4, -2, -.5,   -2, 2.3,  -2, -2, -2.4 };
    std::vector<double> b{  -1, -.2, -1,  -1, -5.1,  -1, 4.4, -1,  1.6 };
    const auto expected = singlepp::dense_l2(a.size(), a.data(), b.data());
    std::vector<double> remapping(a.size());

    {
        auto scaled_a = dense_to_sparse_scaled(a, -2);
        singlepp::setup_sparse_l2_remapping<int>(a.size(), scaled_a, remapping);
        auto remap_copy = remapping;

        auto scaled_b = dense_to_sparse_scaled(b, -1);
        EXPECT_FLOAT_EQ(singlepp::sparse_l2(static_cast<int>(a.size()), scaled_a, remapping, scaled_b), expected);
        EXPECT_EQ(remap_copy, remapping);
    }

    // Checking two compressed sparse vector.
    {
        std::vector<int> a_indices;
        std::vector<double> a_values;
        auto sparse_a = dense_to_compressed_sparse(a, -2, a_indices, a_values);
        singlepp::setup_sparse_l2_remapping<int>(a.size(), sparse_a, remapping);
        auto remap_copy = remapping;

        std::vector<int> b_indices;
        std::vector<double> b_values;
        auto sparse_b = dense_to_compressed_sparse(b, -1, b_indices, b_values);

        EXPECT_FLOAT_EQ(singlepp::sparse_l2(static_cast<int>(a.size()), sparse_a, remapping, sparse_b), expected);
        EXPECT_EQ(remap_copy, remapping);
    }

    // Checking what happens with empty inputs.
    {
        std::vector<double> empty(a.size());
        const auto expected = singlepp::dense_l2(a.size(), a.data(), empty.data());

        singlepp::SparseScaled<int, double> sparse_empty;
        singlepp::setup_sparse_l2_remapping<int>(a.size(), sparse_empty, remapping);
        auto scaled_a = dense_to_sparse_scaled(a, -2);
        EXPECT_FLOAT_EQ(singlepp::sparse_l2(static_cast<int>(a.size()), sparse_empty, remapping, scaled_a), expected);

        EXPECT_EQ(singlepp::sparse_l2(static_cast<int>(a.size()), sparse_empty, remapping, sparse_empty), 0);
    }
}

TEST(ComputeL2, SparseSparseHard) {
    std::mt19937_64 rng(12345);
    std::normal_distribution<> ndist;
    std::uniform_real_distribution<> udist;
    std::size_t n = 57;
    std::vector<double> remapping(n);

    for (int it = 0; it < 100; ++it) {
        const double default_val_a = it / 250.0;
        const double default_val_b = it / 150.0;
        std::vector<double> aa(n, default_val_a), bb(n, default_val_b);

        for (std::size_t i = 0; i < n; ++i) {
            aa[i] = ndist(rng);
            if (udist(rng) <= 0.25) {
                bb[i] = ndist(rng);
            }
        }

        for (std::size_t i = 0; i < n; ++i) {
            aa[i] = ndist(rng);
            if (udist(rng) <= 0.25) {
                bb[i] = ndist(rng);
            }
        }
        auto expected = singlepp::dense_l2(n, aa.data(), bb.data());

        auto scaled_aa = dense_to_sparse_scaled(aa, default_val_a);
        singlepp::setup_sparse_l2_remapping<int>(n, scaled_aa, remapping);
        auto scaled_bb = dense_to_sparse_scaled(bb, default_val_b);
        EXPECT_FLOAT_EQ(expected, singlepp::sparse_l2<int>(n, scaled_aa, remapping, scaled_bb));

        std::vector<int> ibuffer_a, ibuffer_b;
        std::vector<double> vbuffer_a, vbuffer_b;
        auto cs_a = dense_to_compressed_sparse(aa, default_val_a, ibuffer_a, vbuffer_a);
        auto cs_b = dense_to_compressed_sparse(bb, default_val_b, ibuffer_b, vbuffer_b);
        EXPECT_FLOAT_EQ(expected, singlepp::sparse_l2<int>(n, cs_a, remapping, cs_b));
    }
}
