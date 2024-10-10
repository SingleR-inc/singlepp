#include <gtest/gtest.h>

#include "singlepp/scaled_ranks.hpp"
#include "singlepp/SubsetRemapper.hpp"

TEST(SubsetRemapper, Subsets) {
    singlepp::internal::SubsetRemapper<int> remapper;
    remapper.reserve(10);
    remapper.add(1);
    remapper.add(6); 
    remapper.add(1); // duplicates are ignored.
    remapper.add(8);

    // All indices are retained.
    {
        singlepp::internal::RankedVector<double, int> input;
        for (size_t i = 0; i < 10; ++i) {
            input.emplace_back(static_cast<double>(i) / 10, i);
        }

        singlepp::internal::RankedVector<double, int> output;
        remapper.remap(input, output);

        EXPECT_EQ(output.size(), 3);
        EXPECT_EQ(output[0].first, 0.1);
        EXPECT_EQ(output[0].second, 0);
        EXPECT_EQ(output[1].first, 0.6);
        EXPECT_EQ(output[1].second, 1);
        EXPECT_EQ(output[2].first, 0.8);
        EXPECT_EQ(output[2].second, 2);

        // Checking that the clear() method works as expected.
        auto copy = remapper;
        copy.clear();
        copy.remap(input, output);
        EXPECT_TRUE(output.empty());
    }

    // Only even indices are retained.
    {
        singlepp::internal::RankedVector<double, int> input;
        for (size_t i = 0; i < 10; i += 2) {
            input.emplace_back(static_cast<double>(i) / 10, i);
        }

        singlepp::internal::RankedVector<double, int> output;
        remapper.remap(input, output);

        EXPECT_EQ(output.size(), 2);
        EXPECT_EQ(output[0].first, 0.6);
        EXPECT_EQ(output[0].second, 1);
        EXPECT_EQ(output[1].first, 0.8);
        EXPECT_EQ(output[1].second, 2);

        // Checking that the clear() method works as expected.
        auto copy = remapper;
        copy.clear();
        copy.remap(input, output);
        EXPECT_TRUE(output.empty());

        copy.add(4);
        copy.add(1);
        copy.remap(input, output);
        EXPECT_EQ(output.size(), 1);
        EXPECT_EQ(output[0].first, 0.4);
        EXPECT_EQ(output[0].second, 0);
    }
}

TEST(SubsetRemapper, SubsetSmallType) {
    // Check that the remapper behaves correctly when the index type is smaller
    // than the mapping size.
    singlepp::internal::SubsetRemapper<uint8_t> remapper;
    remapper.reserve(300);
    remapper.add(200);
    remapper.add(100); 
    remapper.add(10); 
    remapper.add(100); // ignoring duplicates again!
    remapper.add(255); // need this to force the mapping to exceed the max index size.

    singlepp::internal::RankedVector<double, uint8_t> input;
    for (size_t i = 0; i < 250; i += 10) {
        input.emplace_back(static_cast<double>(i) / 100, i);
    }

    singlepp::internal::RankedVector<double, uint8_t> output;
    remapper.remap(input, output);

    EXPECT_EQ(output.size(), 3);
    EXPECT_EQ(output[0].first, 0.1);
    EXPECT_EQ(output[0].second, 2);
    EXPECT_EQ(output[1].first, 1.0);
    EXPECT_EQ(output[1].second, 1);
    EXPECT_EQ(output[2].first, 2.0);
    EXPECT_EQ(output[2].second, 0);
}
