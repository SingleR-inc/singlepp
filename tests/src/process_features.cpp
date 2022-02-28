#include <gtest/gtest.h>

#include "singlepp/process_features.hpp"
#include "mock_markers.h"

#include <algorithm>
#include <vector>
#include <string>

TEST(IntersectFeatures, Basic) {
    std::vector<std::string> first { "gene_1", "gene_2", "gene_5", "gene_3", "gene_6" };
    std::vector<std::string> second { "gene_3", "gene_1", "gene_4", "gene_8", "gene_2", "gene_7" };

    auto intersection = singlepp::intersect_features(first.size(), first.data(), second.size(), second.data());
    EXPECT_EQ(intersection.size(), 3);
    EXPECT_EQ(intersection[0], std::make_pair(0, 1));
    EXPECT_EQ(intersection[1], std::make_pair(1, 4));
    EXPECT_EQ(intersection[2], std::make_pair(3, 0));

    // Unzipping works as expected.
    auto unzipped = singlepp::unzip(intersection);
    std::vector<int> ref1 { 0, 1, 3 };
    EXPECT_EQ(unzipped.first, ref1);
    std::vector<int> ref2 { 1, 4, 0 };
    EXPECT_EQ(unzipped.second, ref2);
}

TEST(SubsetMarkers, Simple) {
    size_t nlabels = 4;
    auto markers = mock_markers(nlabels, 20, 100);
    auto copy = markers;
    int top = 5;
    auto subs = singlepp::subset_markers(copy, top);

    EXPECT_TRUE(std::is_sorted(subs.begin(), subs.end()));
    EXPECT_TRUE(subs.size() >= top);

    std::unordered_map<int, bool> hits;
    for (auto s : subs) {
        EXPECT_TRUE(hits.find(s) == hits.end()); // no duplicates in output.
        hits[s] = false;
    }

    for (size_t i = 0; i < nlabels; ++i) {
        for (size_t j = 0; j < nlabels; ++j) {
            if (i == j) {
                continue;
            }

            const auto& original = markers[i][j];
            const auto& remapped = copy[i][j];
            EXPECT_EQ(remapped.size(), top);

            for (int s = 0; s < top; ++s) {
                auto r = remapped[s];
                EXPECT_TRUE(r >= 0);
                EXPECT_TRUE(r < subs.size());

                auto o = original[s];
                EXPECT_EQ(subs[r], o);

                auto it = hits.find(o);
                EXPECT_TRUE(it != hits.end());
                if (it != hits.end()) {
                    it->second = true;
                }
            }
        }
    }

    for (const auto& h : hits) {
        EXPECT_TRUE(h.second);
    }
}

TEST(SubsetMarkers, Intersect) {
    size_t nlabels = 4;
    size_t ngenes = 100;
    auto markers = mock_markers(nlabels, 20, ngenes);
    auto inter = mock_intersection(ngenes, ngenes, 40);

    int top = 5;
    auto mcopy = markers;
    auto icopy = inter;
    singlepp::subset_markers(icopy, mcopy, top);

    // Checking for uniqueness.
    std::unordered_map<int, bool> available;
    for (auto s : icopy) {
        ASSERT_TRUE(available.find(s.second) == available.end());
        available[s.second] = false;
    }

    // Check for consistency.
    for (size_t i = 0; i < nlabels; ++i) {
        for (size_t j = 0; j < nlabels; ++j) {
            if (i == j) {
                continue;
            }

            const auto& original = markers[i][j];
            const auto& remapped = mcopy[i][j];
            EXPECT_TRUE(remapped.size() <= top); 

            size_t l = 0;
            for (size_t s = 0; s < remapped.size(); ++s) {
                auto current = icopy[remapped[s]].second;
                available[current] = true;
                while (l < original.size() && original[l] != current) {
                    ++l;
                }
            }
            EXPECT_TRUE(l < original.size()); // should never hit the end, as all should be found. 
        }
    }

    // Check that everyone was used.
    bool all_used = true;
    for (auto a : available) {
        all_used &= a.second;
    }
    EXPECT_TRUE(all_used);
}

TEST(SubsetMarkers, TooLargeTop) {
    size_t nlabels = 5;
    auto markers = mock_markers(nlabels, 20, 123);
    auto copy = markers;
    int top = 50;
    auto subs = singlepp::subset_markers(copy, top);

    for (size_t i = 0; i < nlabels; ++i) {
        for (size_t j = 0; j < nlabels; ++j) {
            const auto& original = markers[i][j];
            const auto& remapped = copy[i][j];
            ASSERT_EQ(original.size(), remapped.size());
            for (size_t s = 0; s < original.size(); ++s) {
                EXPECT_EQ(subs[remapped[s]], original[s]);
            }
        }
    }
}

TEST(SubsetMarkers, TooLargeTop2) {
    size_t nlabels = 4;
    size_t ngenes = 100;
    auto markers = mock_markers(nlabels, 20, ngenes);
    auto inter = mock_intersection(ngenes, ngenes, 40);

    int top = 50;
    auto mcopy = markers;
    auto icopy = inter;
    singlepp::subset_markers(icopy, mcopy, top);

    std::unordered_set<int> available;
    for (auto i : inter){
        available.insert(i.second);
    }

    for (size_t i = 0; i < nlabels; ++i) {
        for (size_t j = 0; j < nlabels; ++j) {
            const auto& original = markers[i][j];
            const auto& remapped = mcopy[i][j];

            size_t l = 0;
            for (size_t s = 0; s < original.size(); ++s) {
                if (available.find(original[s]) != available.end()) {
                    EXPECT_EQ(icopy[remapped[l]].second, original[s]);
                    ++l;
                }
            }
            EXPECT_EQ(l, remapped.size());
        }
    }
}
