#include <gtest/gtest.h>

#include <random>

#include "singlepp/utils.hpp"

TEST(SortByFirst, Basic) {
    std::mt19937_64 rng(12345);
    std::normal_distribution<> ndist;

    std::vector<std::pair<int, double> > x;
    for (int i = 0; i < 10; ++i) {
        x.emplace_back(i * 5, ndist(rng));
    }

    auto expected = x;
    for (int it = 0; it < 100; ++it) {
        std::shuffle(x.begin(), x.end(), rng);
        singlepp::sort_by_first(x);
        EXPECT_EQ(x, expected);
    }
}

TEST(IsSortedUnique, Basic) {
    {
        std::vector<int> foo { 0, 1, 3, 5, 7 };
        EXPECT_TRUE(singlepp::is_sorted_unique(foo.size(), foo.data()));
    }

    {
        std::vector<int> foo { 0, 1, 1, 3, 5, 7 };
        EXPECT_FALSE(singlepp::is_sorted_unique(foo.size(), foo.data()));
    }

    {
        std::vector<int> foo { 0, 1, 5, 7, 3 };
        EXPECT_FALSE(singlepp::is_sorted_unique(foo.size(), foo.data()));
    }

    {
        std::vector<std::pair<int, double> > foo { { 0, 2. }, { 1, 5. }, { 7, 3. } };
        EXPECT_TRUE(singlepp::is_sorted_unique(foo.size(), foo.data()));
    }

    {
        std::vector<std::pair<int, double> > foo { { 1, 5. }, { 0, 3. }, { 7, 3. } };
        EXPECT_FALSE(singlepp::is_sorted_unique(foo.size(), foo.data()));
    }
}
