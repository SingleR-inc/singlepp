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
    auto markers = mock_pairwise_markers<int>(nlabels, 20, ngenes, /* seed = */ 42);
    auto copy = markers;
    auto subs = singlepp::subset_to_markers(copy);

    EXPECT_TRUE(std::is_sorted(subs.begin(), subs.end()));
    EXPECT_LT(subs.size(), ngenes); // not every gene is there, otherwise it would be a trivial test.

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
            const std::size_t noriginal = original.size();
            EXPECT_GE(noriginal, remapped.size());

            for (std::size_t s = 0; s < noriginal; ++s) {
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
        EXPECT_TRUE(h.second); // every gene in the universe is present in the marker list.
    }
}

TEST(SubsetToMarkers, DiagonalOnly) {
    size_t nlabels = 4;
    size_t ngenes = 100;
    auto markers = mock_diagonal_markers<int>(nlabels, 20, ngenes, /* seed = */ 666);

    auto copy = markers;
    auto subs = singlepp::subset_to_markers(copy);
    EXPECT_LT(subs.size(), ngenes); // not every gene is there, otherwise it would be a trivial test.

    for (size_t i = 0; i < nlabels; ++i) {
        for (size_t j = 0; j < nlabels; ++j) {
            if (i == j) {
                EXPECT_EQ(copy[i][j].size(), markers[i][j].size());
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
    auto markers = mock_pairwise_markers<int>(nlabels, 20, ngenes, /* seed = */ 1337);
    auto inter = mock_intersection<int>(ngenes, ngenes, 40, /* seed = */ 13);

    auto mcopy = markers;
    auto unzipped = singlepp::subset_to_markers(inter, mcopy);
    EXPECT_LE(unzipped.first.size(), inter.size());

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
            const std::size_t nremapped = remapped.size();
            const std::size_t noriginal = original.size();
            EXPECT_LE(nremapped, noriginal);

            std::size_t l = 0;
            for (std::size_t s = 0; s < nremapped; ++s) {
                auto current = unzipped.second[remapped[s]];
                available[current] = true;
                while (l < noriginal && original[l] != current) {
                    ++l;
                }
            }
            EXPECT_LT(l, noriginal); // should never hit the end, as all should be found. 
        }
    }

    // Check that everyone was used.
    bool all_used = true;
    for (auto a : available) {
        all_used = all_used && a.second;
    }
    EXPECT_TRUE(all_used);
}

TEST(SubsetToMarkers, IntersectShuffle) {
    size_t nlabels = 4;
    size_t ngenes = 100;
    auto markers = mock_pairwise_markers<int>(nlabels, 20, ngenes, /* seed = */ 94103);
    auto inter = mock_intersection<int>(ngenes, ngenes, ngenes, /* seed = */ 21938); // all genes are used.

    auto mcopy = markers;
    auto unzipped = singlepp::subset_to_markers(inter, mcopy);
    EXPECT_LE(unzipped.first.size(), inter.size());

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
            EXPECT_EQ(remapped.size(), markers[i][j].size()); // all genes should be retained.
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
    auto markers = mock_diagonal_markers<int>(nlabels, 20, ngenes, /* seed = */ 94025);
    auto inter = mock_intersection<int>(ngenes, ngenes, 40, /* seed= */ 123908);

    auto mcopy = markers;
    auto unzipped = singlepp::subset_to_markers(inter, mcopy);
    EXPECT_LE(unzipped.first.size(), inter.size());

    for (size_t i = 0; i < nlabels; ++i) {
        for (size_t j = 0; j < nlabels; ++j) {
            const auto& remapped = mcopy[i][j];
            const std::size_t nremapped = remapped.size();

            if (i == j) {
                const auto& original = markers[i][j];
                const std::size_t noriginal = original.size();
                EXPECT_LE(nremapped, noriginal);

                std::size_t l = 0;
                for (std::size_t s = 0; s < nremapped; ++s) {
                    auto current = unzipped.second[remapped[s]];
                    while (l < noriginal && original[l] != current) {
                        ++l;
                    }
                }
            } else {
                EXPECT_EQ(nremapped, 0);
            }
        }
    }
}
