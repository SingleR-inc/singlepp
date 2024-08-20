#include <gtest/gtest.h>

#include "singlepp/SubsetSanitizer.hpp"

TEST(SubsetSanitizer, NoOp) {
    std::vector<int> foo{ 2, 6, 18, 23, 53, 99 };
    singlepp::internal::SubsetSanitizer ss(foo);
    EXPECT_EQ(&(ss.extraction_subset()), &foo);
    EXPECT_EQ(&ss.extraction_subset(), &foo); // exact some object, in fact.

    singlepp::internal::RankedVector<double, int> vec(foo.size());
    std::vector<double> stuff { 0.34817868, 0.24918308, 0.75879770, 0.71893282, 0.78199329, 0.09039928 };
    ss.fill_ranks(stuff.data(), vec);

    double prev = -100000;
    for (size_t i = 0; i < vec.size(); ++i) {
        EXPECT_TRUE(vec[i].first > prev);
        EXPECT_EQ(vec[i].first, stuff[vec[i].second]);
        prev = vec[i].first;
    }
}

TEST(SubsetSanitizer, Resort) {
    std::vector<int> foo{ 5, 2, 29, 12, 23, 0 };
    singlepp::internal::SubsetSanitizer ss(foo);
    EXPECT_NE(&(ss.extraction_subset()), &foo);

    auto foocopy = foo;
    std::sort(foocopy.begin(), foocopy.end());
    EXPECT_EQ(ss.extraction_subset(), foocopy);
    
    // Here, stuff corresponds to _sorted_ foo, i.e., foocopy,
    // as we're extracting based on the sorted subsets.
    std::vector<double> stuff { 0.94472810, 0.31766805, 0.07027965, 0.38385888, 0.89919158, 0.73368374 };
    singlepp::internal::RankedVector<double, int> vec(foocopy.size());
    ss.fill_ranks(stuff.data(), vec); 

    // Check that we get the same results as if we had done a full column
    // extraction and then extracted the subset from the array.
    std::vector<double> expanded(*std::max_element(foo.begin(), foo.end()) + 1);
    for (size_t s = 0; s < foocopy.size(); ++s) {
        expanded[foocopy[s]] = stuff[s];
    }

    for (size_t i = 0; i < vec.size(); ++i) {
        auto s = foo[vec[i].second]; // check it was correctly reindexed back to foo.
        EXPECT_EQ(vec[i].first, expanded[s]);
    }
}

TEST(SubsetSanitizer, Deduplicate) {
    std::vector<int> foo{ 1, 2, 1, 5, 2, 9, 9, 4, 1, 0 };
    singlepp::internal::SubsetSanitizer ss(foo);
    EXPECT_NE(&(ss.extraction_subset()), &foo);

    std::vector<int> foocopy{ 0, 1, 2, 4, 5, 9 };
    EXPECT_EQ(ss.extraction_subset(), foocopy);
    
    // Here, stuff corresponds to _deduplicated_ foo, i.e., foocopy,
    // as we're extracting based on the deduplicated + sorted subsets.
    std::vector<double> stuff { 0.5404277, 0.2643289, 0.1282597, 0.4206395, 0.5222923, 0.4991335 };
    singlepp::internal::RankedVector<double, int> vec(foo.size());
    ss.fill_ranks(stuff.data(), vec); 

    // Check that we get the same results as if we had done a full column
    // extraction and then extracted the subset from the array.
    std::vector<double> expanded(*std::max_element(foo.begin(), foo.end()) + 1);
    for (size_t s = 0; s < foocopy.size(); ++s) {
        expanded[foocopy[s]] = stuff[s];
    }

    for (size_t i = 0; i < vec.size(); ++i) {
        auto s = foo[vec[i].second]; // check it was correctly reindexed back to foo.
        EXPECT_EQ(vec[i].first, expanded[s]);
    }
}
