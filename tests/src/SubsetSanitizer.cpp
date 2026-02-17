#include <gtest/gtest.h>

#include "singlepp/SubsetSanitizer.hpp"

TEST(SubsetSanitizer, DenseNoOp) {
    std::vector<int> foo{ 2, 6, 18, 23, 53, 99 };
    singlepp::SubsetNoop<false, int> ss(foo);
    EXPECT_EQ(&(ss.extraction_subset()), &foo); // exact same object, in fact.

    std::vector<double> stuff { 0.34817868, 0.24918308, 0.75879770, 0.71893282, 0.78199329, 0.09039928 };
    singlepp::RankedVector<double, int> vec;
    ss.fill_ranks(stuff.data(), vec);

    EXPECT_EQ(vec.size(), foo.size());
    std::vector<double> reformatted(stuff.size());
    for (auto rr : vec) {
        reformatted[rr.second] = rr.first;
    }
    EXPECT_EQ(reformatted, stuff);
}

TEST(SubsetSanitizer, SparseNoOp) {
    std::vector<int> foo{ 2, 6, 18, 23, 53, 99 };
    EXPECT_TRUE(singlepp::is_sorted_unique(foo.size(), foo.data()));

    std::vector<double> vstuff{ 0.31, -0.23, 0.45 };
    std::vector<int> istuff{ 6, 18, 99 };
    tatami::SparseRange<double, int> stuff(3, vstuff.data(), istuff.data());

    singlepp::SubsetNoop<true, int> ss(foo);
    EXPECT_EQ(&(ss.extraction_subset()), &foo); // exact same object, in fact.
    singlepp::RankedVector<double, int> vec;
    ss.fill_ranks(stuff, vec);

    singlepp::RankedVector<double, int> expected;
    for (std::size_t i = 0; i < vstuff.size(); ++i) {
        expected.emplace_back(vstuff[i], istuff[i]);
    }
    std::sort(expected.begin(), expected.end());
    auto transformed = vec;
    for (auto& tt : transformed) {
        tt.second = foo[tt.second];
    }
    EXPECT_EQ(transformed, expected);
}

TEST(SubsetSanitizer, DenseResort) {
    std::vector<int> foo{ 5, 2, 29, 12, 23, 0 };
    singlepp::SubsetSanitizer<false, int> ss(foo);
    EXPECT_NE(&(ss.extraction_subset()), &foo);

    auto foocopy = foo;
    std::sort(foocopy.begin(), foocopy.end());
    EXPECT_EQ(ss.extraction_subset(), foocopy);
    
    // Here, stuff corresponds to _sorted_ foo, i.e., foocopy,
    // as we're extracting based on the sorted subsets.
    std::vector<double> stuff { 0.94472810, 0.31766805, 0.07027965, 0.38385888, 0.89919158, 0.73368374 };
    singlepp::RankedVector<double, int> vec(foocopy.size());
    ss.fill_ranks(stuff.data(), vec); 

    // Check that we get the same results as if we had done a full column
    // extraction and then extracted the subset from the array.
    std::vector<double> expanded(*std::max_element(foo.begin(), foo.end()) + 1);
    for (size_t s = 0; s < foocopy.size(); ++s) {
        expanded[foocopy[s]] = stuff[s];
    }

    EXPECT_EQ(vec.size(), foo.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        auto s = foo[vec[i].second]; // check it was correctly reindexed back to foo.
        EXPECT_EQ(vec[i].first, expanded[s]);
    }
}

TEST(SubsetSanitizer, SparseResort) {
    std::vector<int> foo{ 16, 22, 25, 40, 47, 27, 3, 20, 23, 48 };
    singlepp::SubsetSanitizer<true, int> ss(foo);
    EXPECT_NE(&(ss.extraction_subset()), &foo);

    auto foocopy = foo;
    std::sort(foocopy.begin(), foocopy.end());
    EXPECT_EQ(ss.extraction_subset(), foocopy);
    
    // Here, stuff corresponds to _sorted_ foo, i.e., foocopy,
    // as we're extracting based on the sorted subsets.
    std::vector<double> vstuff{ 0.12, -0.22, 0.6, 0.31, -0.23 };
    std::vector<int> istuff{ 3, 22, 23, 40, 47 };
    tatami::SparseRange<double, int> stuff(5, vstuff.data(), istuff.data());
    singlepp::RankedVector<double, int> vec;
    ss.fill_ranks(stuff, vec);

    singlepp::RankedVector<double, int> expected;
    for (std::size_t i = 0; i < vstuff.size(); ++i) {
        expected.emplace_back(vstuff[i], istuff[i]);
    }
    std::sort(expected.begin(), expected.end());
    auto transformed = vec;
    for (auto& tt : transformed) {
        tt.second = foo[tt.second];
    }
    EXPECT_EQ(transformed, expected);
}
