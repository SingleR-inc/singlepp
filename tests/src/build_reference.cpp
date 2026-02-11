#include <gtest/gtest.h>

#include "singlepp/build_reference.hpp"

TEST(ComputeL2, DenseDense) {
    std::vector<double> a{1.2, -0.5, 2.3, 5.6 };
    std::vector<double> b{-.2, -5.1, 4.4, 1.6 };

    double expected = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const auto delta = a[i] - b[i];
        expected += delta * delta;
    }

    EXPECT_EQ(singlepp::compute_l2(a.size(), a, b), expected);
    EXPECT_EQ(singlepp::compute_l2(a.size(), a, b.data()), expected);
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

TEST(ComputeL2, DenseSparse) {
    std::vector<double> a{3.2, 1.2, -2.7, -.5,   .6, 2.3, 5.6, -1.3, -2.4 };
    std::vector<double> b{ -1, -.2,   -1,  -1, -5.1,  -1, 4.4,   -1,  1.6 };
    const auto expected = singlepp::compute_l2(a.size(), a, b);

    // Sparse scaled.
    {
        auto scaled_b = dense_to_sparse_scaled(b, -1);
        EXPECT_EQ(singlepp::compute_l2(static_cast<int>(a.size()), a, scaled_b), expected);
        EXPECT_EQ(singlepp::compute_l2(static_cast<int>(a.size()), scaled_b, a), expected);
    }

    // Compressed sparse.
    {
        std::vector<int> indices;
        std::vector<double> values;
        auto sparse_b = dense_to_compressed_sparse(b, -1, indices, values);
        EXPECT_EQ(singlepp::compute_l2(static_cast<int>(a.size()), a, sparse_b), expected);
    }

    // Checking what happens if the last non-zero element of the other vector is not at the end. 
    {
        auto a2 = a;
        a2.push_back(0.2);
        a2.push_back(-2.5);
        a2.push_back(-1.7);
        auto b2 = b;
        b2.resize(a2.size(), -1);
        const auto expected2 = singlepp::compute_l2(a2.size(), a2, b2);

        auto scaled_b = dense_to_sparse_scaled(b2, -1);
        EXPECT_EQ(singlepp::compute_l2(static_cast<int>(a2.size()), a2, scaled_b), expected2);
    }
}

TEST(ComputeL2, SparseSparse) {
    std::vector<double> a{3.2, 1.4, -2, -.5,   -2, 2.3,  -2, -2, -2.4};
    std::vector<double> b{ -1, -.2, -1,  -1, -5.1,  -1, 4.4, -1,  1.6};
    const auto expected = singlepp::compute_l2(a.size(), a, b);

    {
        auto scaled_a = dense_to_sparse_scaled(a, -2);
        auto scaled_b = dense_to_sparse_scaled(b, -1);
        EXPECT_FLOAT_EQ(singlepp::compute_l2(static_cast<int>(a.size()), scaled_a, scaled_b), expected);
    }

    // Checking two compressed sparse vector.
    {
        std::vector<int> a_indices;
        std::vector<double> a_values;
        auto sparse_a = dense_to_compressed_sparse(a, -2, a_indices, a_values);

        std::vector<int> b_indices;
        std::vector<double> b_values;
        auto sparse_b = dense_to_compressed_sparse(b, -1, b_indices, b_values);

        EXPECT_FLOAT_EQ(singlepp::compute_l2(static_cast<int>(a.size()), sparse_a, sparse_b), expected);
    }

    // Checking what happens if there are non-zero elements in one vector after the last non-zero element of the other vector. 
    {
        auto a2 = a;
        a2.push_back(3.2);
        a2.push_back(-2);
        a2.push_back(-.7);
        auto b2 = b;
        b2.resize(a2.size(), -1);
        const auto expected2 = singlepp::compute_l2(a2.size(), a2, b2);

        auto scaled_a = dense_to_sparse_scaled(a2, -2);
        auto scaled_b = dense_to_sparse_scaled(b2, -1);
        EXPECT_FLOAT_EQ(singlepp::compute_l2(static_cast<int>(a2.size()), scaled_a, scaled_b), expected2);
        EXPECT_FLOAT_EQ(singlepp::compute_l2(static_cast<int>(a2.size()), scaled_b, scaled_a), expected2);
    }

    // Checking what happens with empty inputs.
    {
        std::vector<double> empty(a.size());
        const auto expected = singlepp::compute_l2(a.size(), a, empty);

        singlepp::SparseScaled<int, double> sparse_empty;
        EXPECT_FLOAT_EQ(singlepp::compute_l2(static_cast<int>(a.size()), a, sparse_empty), expected);
        EXPECT_FLOAT_EQ(singlepp::compute_l2(static_cast<int>(a.size()), sparse_empty, a), expected);

        EXPECT_FLOAT_EQ(singlepp::compute_l2(static_cast<int>(a.size()), sparse_empty, sparse_empty), 0);
    }
}
