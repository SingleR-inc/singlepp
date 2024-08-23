#include <gtest/gtest.h>

#include "singlepp/subset_to_markers.hpp"
#include "mock_markers.h"

#include <algorithm>
#include <vector>
#include <string>
#include <unordered_set>

TEST(SubsetToMarkers, Simple) {
    size_t nlabels = 4;
    size_t ngenes = 100;
    auto markers = mock_markers<int>(nlabels, 20, ngenes);
    auto copy = markers;
    int top = 5;
    auto subs = singlepp::internal::subset_to_markers(copy, top);

    EXPECT_TRUE(std::is_sorted(subs.begin(), subs.end()));
    EXPECT_LT(subs.size(), ngenes); // not every gene is there, otherwise it would be a trivial test.
    EXPECT_GE(subs.size(), top);

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
                EXPECT_GE(r, 0);
                EXPECT_LT(r, subs.size());

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

TEST(SubsetToMarkers, TooLargeTop) {
    size_t nlabels = 5;
    size_t ngenes = 123;
    auto markers = mock_markers<int>(nlabels, 20, ngenes);
    auto copy = markers;
    int top = 50;
    auto subs = singlepp::internal::subset_to_markers(copy, top);

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

TEST(SubsetToMarkers, NoTop) {
    size_t nlabels = 4;
    size_t ngenes = 100;
    auto markers = mock_markers<int>(nlabels, 20, ngenes);
    auto copy = markers;

    auto subs = singlepp::internal::subset_to_markers(copy, -1);
    EXPECT_TRUE(subs.size() > 0);

    for (size_t i = 0; i < nlabels; ++i) {
        for (size_t j = 0; j < nlabels; ++j) {
            auto reset = copy[i][j];
            for (auto& r : reset) {
                r = subs[r];
            }
            EXPECT_EQ(reset, markers[i][j]);
        }
    }
}

TEST(SubsetToMarkers, DiagonalOnly) {
    size_t nlabels = 4;
    size_t ngenes = 100;
    auto markers = mock_markers_diagonal<int>(nlabels, 20, ngenes);

    auto copy = markers;
    int top = 5;
    auto subs = singlepp::internal::subset_to_markers(copy, top);
    EXPECT_LT(subs.size(), ngenes); // not every gene is there, otherwise it would be a trivial test.
    EXPECT_GE(subs.size(), top);

    for (size_t i = 0; i < nlabels; ++i) {
        for (size_t j = 0; j < nlabels; ++j) {
            if (i == j) {
                EXPECT_EQ(copy[i][j].size(), top);
                for (size_t k = 0; k < copy[i][j].size(); ++k) {
                    EXPECT_EQ(subs[copy[i][j][k]], markers[i][j][k]);
                }
            } else {
                EXPECT_EQ(copy[i][j].size(), 0);
            }
        }
    }
}

TEST(SubsetToMarkers, Intersect) {
    size_t nlabels = 4;
    size_t ngenes = 100;
    auto markers = mock_markers<int>(nlabels, 20, ngenes);
    auto inter = mock_intersection<int>(ngenes, ngenes, 40);

    int top = 5;
    auto mcopy = markers;
    auto unzipped = singlepp::internal::subset_to_markers(inter, mcopy, top);
    EXPECT_GE(unzipped.first.size(), top);

    // Checking for uniqueness.
    std::unordered_map<int, bool> available;
    for (auto s : unzipped.second) {
        ASSERT_TRUE(available.find(s) == available.end());
        available[s] = false;
    }

    // Check for consistency.
    for (size_t i = 0; i < nlabels; ++i) {
        for (size_t j = 0; j < nlabels; ++j) {
            if (i == j) {
                continue;
            }

            const auto& original = markers[i][j];
            const auto& remapped = mcopy[i][j];
            EXPECT_LE(remapped.size(), top); 

            size_t l = 0;
            for (size_t s = 0; s < remapped.size(); ++s) {
                auto current = unzipped.second[remapped[s]];
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
        all_used = all_used && a.second;
    }
    EXPECT_TRUE(all_used);
}

TEST(SubsetToMarkers, IntersectTooLargeTop) {
    size_t nlabels = 4;
    size_t ngenes = 100;
    auto markers = mock_markers<int>(nlabels, 20, ngenes);
    auto inter = mock_intersection<int>(ngenes, ngenes, 40);

    int top = 50;
    auto mcopy = markers;
    auto unzipped = singlepp::internal::subset_to_markers(inter, mcopy, top);

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
                    EXPECT_EQ(unzipped.second[remapped[l]], original[s]);
                    ++l;
                }
            }
            EXPECT_EQ(l, remapped.size());
        }
    }
}

TEST(SubsetToMarkers, IntersectNoTop) {
    size_t nlabels = 4;
    size_t ngenes = 100;
    auto markers = mock_markers<int>(nlabels, 20, ngenes);
    auto inter = mock_intersection<int>(ngenes, ngenes, 40);

    auto mcopy = markers;
    auto unzipped = singlepp::internal::subset_to_markers(inter, mcopy, -1);

    // Same result as if we took a very large top set.
    auto mcopy2 = markers;
    auto unzipped2 = singlepp::internal::subset_to_markers(inter, mcopy2, 10000);

    EXPECT_EQ(unzipped, unzipped2);

    for (size_t i = 0; i < nlabels; ++i) {
        for (size_t j = 0; j < nlabels; ++j) {
            EXPECT_EQ(mcopy[i][j], mcopy2[i][j]);
        }
    }
}

TEST(SubsetToMarkers, IntersectShuffle) {
    size_t nlabels = 4;
    size_t ngenes = 100;
    auto markers = mock_markers<int>(nlabels, 20, ngenes);
    auto inter = mock_intersection<int>(ngenes, ngenes, ngenes); // all genes are used.

    int top = 5;
    auto mcopy = markers;
    auto unzipped = singlepp::internal::subset_to_markers(inter, mcopy, top);
    EXPECT_GE(unzipped.first.size(), top);

    // Checking for uniqueness.
    std::unordered_map<int, bool> available;
    for (auto s : unzipped.second) {
        ASSERT_TRUE(available.find(s) == available.end());
        available[s] = false;
    }

    // Check for consistency.
    for (size_t i = 0; i < nlabels; ++i) {
        for (size_t j = 0; j < nlabels; ++j) {
            if (i == j) {
                continue;
            }

            const auto& remapped = mcopy[i][j];
            EXPECT_EQ(remapped.size(), top); // exactly 'top' genes should be retained.

            for (size_t s = 0; s < remapped.size(); ++s) {
                auto current = unzipped.second[remapped[s]];
                available[current] = true;
            }
        }
    }

    // Check that everyone was used.
    bool all_used = true;
    for (auto a : available) {
        all_used = all_used && a.second;
    }
    EXPECT_TRUE(all_used);
}

TEST(SubsetToMarkers, IntersectDiagonal) {
    size_t nlabels = 4;
    size_t ngenes = 100;
    auto markers = mock_markers_diagonal<int>(nlabels, 20, ngenes);
    auto inter = mock_intersection<int>(ngenes, ngenes, 40);

    auto mcopy = markers;
    int top = 5;
    auto unzipped = singlepp::internal::subset_to_markers(inter, mcopy, top);
    EXPECT_GE(unzipped.first.size(), top);

    for (size_t i = 0; i < nlabels; ++i) {
        for (size_t j = 0; j < nlabels; ++j) {
            const auto& remapped = mcopy[i][j];

            if (i == j) {
                EXPECT_LE(remapped.size(), top);
                const auto& original = markers[i][j];
                size_t l = 0;
                for (size_t s = 0; s < remapped.size(); ++s) {
                    auto current = unzipped.second[remapped[s]];
                    while (l < original.size() && original[l] != current) {
                        ++l;
                    }
                }
            } else {
                EXPECT_EQ(remapped.size(), 0);
            }
        }
    }
}
