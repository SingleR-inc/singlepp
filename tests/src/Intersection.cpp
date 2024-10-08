#include <gtest/gtest.h>

#include "singlepp/Intersection.hpp"

#include <vector>

TEST(IntersectFeatures, Basic) {
    std::vector<int> first { 1, 2, 5, 3, 6 };
    std::vector<int> second { 3, 1, 4, 8, 2, 7 };
    auto intersection = singlepp::intersect_genes<int>(first.size(), first.data(), second.size(), second.data());

    EXPECT_EQ(intersection.size(), 3);
    EXPECT_EQ(intersection[0], std::make_pair(0, 1));
    EXPECT_EQ(intersection[1], std::make_pair(1, 4));
    EXPECT_EQ(intersection[2], std::make_pair(3, 0));
}

TEST(IntersectFeatures, Duplicates) {
    std::vector<int> first { 1, 3, 1, 3, 2 };
    std::vector<int> second { 3, 2, 3, 1, 2, 1 };
    auto intersection = singlepp::intersect_genes<int>(first.size(), first.data(), second.size(), second.data());

    // We only report the first occurrence of duplicated IDs.
    EXPECT_EQ(intersection.size(), 3);
    EXPECT_EQ(intersection[0], std::make_pair(0, 3));
    EXPECT_EQ(intersection[1], std::make_pair(1, 0));
    EXPECT_EQ(intersection[2], std::make_pair(4, 1));
}
