#include <gtest/gtest.h>

#include "singlepp/subset_to_markers.hpp"
#include "mock_markers.h"

#include <algorithm>
#include <vector>
#include <string>
#include <set>

TEST(SubsetToMarkers, Simple) {
    size_t nlabels = 4;
    size_t ngenes = 100;
    auto markers = mock_pairwise_markers<int>(nlabels, 20, ngenes, /* seed = */ 42);
    auto copy = markers;
    auto subs = singlepp::subset_to_markers<int>(ngenes, copy);

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
    auto subs = singlepp::subset_to_markers<int>(ngenes, copy);
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

class SubsetToMarkersIntersectTest : public ::testing::TestWithParam<std::tuple<int, int, int> > {};

TEST_P(SubsetToMarkersIntersectTest, Basic) {
    auto params = GetParam(); 
    const auto ref_ngenes = std::get<0>(params);
    const auto test_ngenes = std::get<1>(params);
    const auto shared = std::get<2>(params);
    unsigned long long base_seed = ref_ngenes + 2 * test_ngenes + 3 * shared;

    size_t nlabels = 4;
    auto markers = mock_pairwise_markers<int>(nlabels, 20, ref_ngenes, /* seed = */ base_seed);
    auto inter = mock_intersection<int>(test_ngenes, ref_ngenes, shared, /* seed = */ base_seed + 13);

    auto mcopy = markers;
    auto unzipped = singlepp::subset_to_markers<int>(test_ngenes, inter, ref_ngenes, mcopy);

    {
        EXPECT_LE(unzipped.first.size(), inter.size());
        EXPECT_EQ(unzipped.first.size(), unzipped.second.size());
        EXPECT_TRUE(std::is_sorted(unzipped.first.begin(), unzipped.first.end()));

        std::set<std::pair<int, int> > all_inters(inter.begin(), inter.end());
        const std::size_t nsubset = unzipped.first.size();
        for (std::size_t s = 0; s < nsubset; ++s) {
            EXPECT_TRUE(all_inters.find(std::pair<int, int>(unzipped.first[s], unzipped.second[s])) != all_inters.end());
        }
    }

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

            if (shared == ref_ngenes) {
                EXPECT_EQ(nremapped, noriginal);
            } else {
                EXPECT_LE(nremapped, noriginal);
            }

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

TEST_P(SubsetToMarkersIntersectTest, Diagonal) {
    auto params = GetParam(); 
    const auto ref_ngenes = std::get<0>(params);
    const auto test_ngenes = std::get<1>(params);
    const auto shared = std::get<2>(params);
    unsigned long long base_seed = ref_ngenes + 5 * test_ngenes + 2 * shared;

    size_t nlabels = 4;
    auto markers = mock_diagonal_markers<int>(nlabels, 20, ref_ngenes, /* seed = */ base_seed + 94025);
    auto inter = mock_intersection<int>(test_ngenes, ref_ngenes, 40, /* seed= */ base_seed + 123908);

    auto mcopy = markers;
    auto unzipped = singlepp::subset_to_markers<int>(test_ngenes, inter, ref_ngenes, mcopy);

    {
        EXPECT_LE(unzipped.first.size(), inter.size());
        EXPECT_EQ(unzipped.first.size(), unzipped.second.size());
        EXPECT_TRUE(std::is_sorted(unzipped.first.begin(), unzipped.first.end()));

        std::set<std::pair<int, int> > all_inters(inter.begin(), inter.end());
        const std::size_t nsubset = unzipped.first.size();
        for (std::size_t s = 0; s < nsubset; ++s) {
            EXPECT_TRUE(all_inters.find(std::pair<int, int>(unzipped.first[s], unzipped.second[s])) != all_inters.end());
        }
    }

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

INSTANTIATE_TEST_SUITE_P(
    SubsetToMarkers,
    SubsetToMarkersIntersectTest,
    ::testing::Values(
        std::tuple<int, int, int>(100, 100, 40),
        std::tuple<int, int, int>(100, 100, 100),
        std::tuple<int, int, int>(50, 150, 40),
        std::tuple<int, int, int>(150, 50, 40)
    )
);
