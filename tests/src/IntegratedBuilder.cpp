#include <gtest/gtest.h>
#include "custom_parallel.h"

#include "singlepp/IntegratedBuilder.hpp"
#include "singlepp/BasicBuilder.hpp"
#include "spawn_matrix.h"
#include "mock_markers.h"

class IntegratedBuilderTestCore {
protected:
    std::vector<std::shared_ptr<tatami::Matrix<double, int> > > matrices;
    std::vector<std::vector<int> > labels;
    std::vector<singlepp::Markers> markers;

    size_t ngenes = 2000;
    size_t nsamples = 50;
    size_t nrefs = 3;
protected:
    void simulate_references() {
        for (size_t r = 0; r < nrefs; ++r) {
            size_t seed = r * 1000;
            size_t nlabels = 3 + r;

            matrices.push_back(spawn_matrix(ngenes, nsamples, seed));
            labels.push_back(spawn_labels(nsamples, nlabels, seed * 2));
            markers.push_back(mock_markers(nlabels, 50, ngenes, seed * 3));
        }
    }

    std::vector<int> simulate_test_ids() const {
        std::vector<int> ids(ngenes);
        for (size_t g = 0; g < ngenes; ++g) {
            ids[g] = g;
        }
        return ids;
    }

    std::vector<int> simulate_ref_ids(int seed) const {
        std::vector<int> keep;
        std::mt19937_64 rng(seed);
        for (size_t s = 0; s < ngenes; ++s) {
            if (rng() % 100 < 50) {
                keep.push_back(s);
            } else {
                keep.push_back(-1); // i.e., unique to the reference.
            }
        }
        return keep;
    }

    singlepp::Markers truncate_markers(singlepp::Markers remarkers, int ntop) const {
        for (auto& x : remarkers) {
            for (auto& y : x) {
                if (y.size() > static_cast<size_t>(ntop)) {
                    y.resize(ntop);
                }
            }
        }
        return remarkers;
    }
};

/********************************************/

class IntegratedBuilderBasicTest : public ::testing::TestWithParam<std::tuple<int, int> >, public IntegratedBuilderTestCore {};

TEST_P(IntegratedBuilderBasicTest, SimpleCombine) {
    auto param = GetParam();
    int ntop = std::get<0>(param);
    int nthreads = std::get<1>(param);

    // Mocking up the test and references.
    simulate_references();

    singlepp::BasicBuilder builder;
    builder.set_top(ntop);

    singlepp::IntegratedBuilder inter;
    inter.set_num_threads(nthreads);

    for (size_t r = 0; r < nrefs; ++r) {
        auto pre = builder.run(matrices[r].get(), labels[r].data(), markers[r]);
        inter.add(matrices[r].get(), labels[r].data(), pre);
    }

    auto output = inter.finish();

    // Checking the values of the built references.
    EXPECT_EQ(output.num_references(), nrefs);
    const auto& universe = output.universe;

    for (size_t r = 0; r < nrefs; ++r) {
        EXPECT_FALSE(output.check_availability[r]);

        size_t nlabels = 3 + r;
        EXPECT_EQ(output.markers[r].size(), nlabels);
        EXPECT_EQ(output.num_labels(r), nlabels);
        EXPECT_EQ(output.num_profiles(r), nsamples);

        // Checking the contents of the markers. 
        const auto& cur_markers = markers[r];
        EXPECT_EQ(cur_markers.size(), nlabels);

        for (size_t l = 0; l < nlabels; ++l) {
            std::unordered_set<int> kept;
            for (size_t l2 = 0; l2 < nlabels; ++l2) {
                if (l != l2) {
                    const auto& current = cur_markers[l][l2];
                    kept.insert(current.begin(), current.begin() + ntop);
                }
            }

            std::vector<int> to_use(kept.begin(), kept.end());
            std::sort(to_use.begin(), to_use.end());

            auto copy = output.markers[r][l];
            for (auto& x : copy) {
                x = universe[x];
            }
            std::sort(copy.begin(), copy.end());

            EXPECT_EQ(to_use, copy);
        }

        // Checking the ranked values.
        std::vector<int> offsets(nlabels);
        auto wrk = matrices[r]->dense_column();

        for (size_t s = 0; s < nsamples; ++s) {
            int lab = labels[r][s]; 
            const auto& target = output.ranked[r][lab][offsets[lab]];
            ++offsets[lab];

            double last_original = -100000000;
            auto col = wrk->fetch(s);
            std::vector<int> test_in_use;
            test_in_use.push_back(universe[target[0].second]);

            for (size_t i = 1; i < target.size(); ++i) {
                const auto& prev = target[i-1];
                const auto& x = target[i];
                EXPECT_TRUE(prev.first < x.first); // no ties in this simulation.
                EXPECT_TRUE(col[universe[prev.second]] < col[universe[x.second]]);
                test_in_use.push_back(universe[x.second]);
            }

            // Checking that all features are represented.
            std::sort(test_in_use.begin(), test_in_use.end());
            EXPECT_EQ(test_in_use, universe);
        }
    }
}

INSTANTIATE_TEST_CASE_P(
    IntegratedBuilder,
    IntegratedBuilderBasicTest,
    ::testing::Combine(
        ::testing::Values(5, 10, 20), // number of top genes.
        ::testing::Values(1, 3) // number of threads
    )
);

/********************************************/

class IntegratedBuilderMoreTest : public ::testing::TestWithParam<int>, public IntegratedBuilderTestCore {};

TEST_P(IntegratedBuilderMoreTest, SimpleCombineNaked) {
    // Mocking up the test and references.
    int ntop = GetParam();
    simulate_references();

    singlepp::BasicBuilder builder;
    builder.set_top(ntop);
    singlepp::IntegratedBuilder alpha, bravo;

    for (size_t r = 0; r < nrefs; ++r) {
        auto pre = builder.run(matrices[r].get(), labels[r].data(), markers[r]);
        alpha.add(matrices[r].get(), labels[r].data(), pre);

        // Comparing what happens with direct input of markers of interest.
        auto remarkers = truncate_markers(markers[r], ntop);
        bravo.add(matrices[r].get(), labels[r].data(), remarkers);
    }

    auto alpha_output = alpha.finish();
    auto bravo_output = bravo.finish();

    EXPECT_EQ(alpha_output.universe, bravo_output.universe);
    EXPECT_EQ(alpha_output.num_references(), bravo_output.num_references());

    // Checking the rank values for equality.
    for (size_t r = 0; r < nrefs; ++r) {
        const auto& cur_alpha = alpha_output.ranked[r];
        const auto& cur_bravo = bravo_output.ranked[r];
        ASSERT_EQ(cur_alpha.size(), cur_bravo.size());

        for (size_t l = 0; l < cur_alpha.size(); ++l) {
            const auto& rank_alpha = cur_alpha[l];
            const auto& rank_bravo = cur_bravo[l];
            ASSERT_EQ(rank_alpha.size(), rank_bravo.size());

            for (size_t s = 0; s < rank_alpha.size(); ++s) {
                EXPECT_EQ(rank_alpha[s], rank_bravo[s]);
            }
        }
    }
}

TEST_P(IntegratedBuilderMoreTest, IntersectedCombine) {
    // Mocking up the test and references.
    int ntop = GetParam();
    simulate_references();

    auto ids = simulate_test_ids();
    std::vector<std::vector<int> > kept;
    for (size_t r = 0; r < nrefs; ++r) {
        size_t seed = r * 555;
        kept.push_back(simulate_ref_ids(seed));
    }

    singlepp::BasicBuilder builder;
    builder.set_top(ntop);
    singlepp::IntegratedBuilder inter;

    // Adding each reference to the list. We store the single prebuilts for testing later.
    std::vector<typename singlepp::BasicBuilder::PrebuiltIntersection> single_ref;
    for (size_t r = 0; r < nrefs; ++r) {
        auto pre = builder.run(ngenes, ids.data(), matrices[r].get(), kept[r].data(), labels[r].data(), markers[r]);
        single_ref.push_back(pre);
        inter.add(ngenes, ids.data(), matrices[r].get(), kept[r].data(), labels[r].data(), pre);
    }

    auto output = inter.finish();

    EXPECT_EQ(output.num_references(), nrefs);
    const auto& universe = output.universe;

    for (size_t r = 0; r < nrefs; ++r) {
        EXPECT_TRUE(output.check_availability[r]);

        // Creating a mapping.
        std::unordered_map<int, int> mapping;
        const auto& keep = kept[r];
        for (size_t g = 0; g < keep.size(); ++g) {
            if (keep[g] != -1) {
                mapping[keep[g]] = g;
            }
        }

        // Check consistency of the availability set.
        int not_found = false;
        const auto& available = output.available[r];
        for (const auto& a : available) {
            not_found += (mapping.find(universe[a]) == mapping.end());
        }
        EXPECT_EQ(not_found, 0);

        std::vector<int> local_universe(available.begin(), available.end());
        for (auto& x : local_universe) {
            x = universe[x];
        }
        std::sort(local_universe.begin(), local_universe.end());

        // Check consistency of the markers.
        const auto& outmarkers = output.markers[r];
        const auto& cur_markers = single_ref[r].markers;
        size_t nlabels = outmarkers.size();
        EXPECT_EQ(cur_markers.size(), nlabels);

        for (size_t l = 0; l < nlabels; ++l) {
            std::unordered_set<int> kept;
            for (size_t l2 = 0; l2 < nlabels; ++l2) {
                if (l != l2) {
                    const auto& current = cur_markers[l][l2];
                    for (auto x : current) {
                        kept.insert(single_ref[r].mat_subset[x]); // for comparison to universe indices, which are all relative to the test matrix.
                    }
                }
            }

            int not_found = 0;
            for (auto& x : outmarkers[l]) {
                not_found += (kept.find(universe[x]) == kept.end());
            }
            EXPECT_EQ(not_found, 0);
        }

        // Checking rankings for consistency with the availabilities.
        std::vector<int> offsets(nlabels);
        auto wrk = matrices[r]->dense_column();

        for (size_t s = 0; s < nsamples; ++s) {
            int lab = labels[r][s]; 
            const auto& target = output.ranked[r][lab][offsets[lab]];
            ++offsets[lab];

            double last_original = -100000000;
            auto col = wrk->fetch(s);
            std::vector<int> test_in_use;
            test_in_use.push_back(universe[target[0].second]);

            for (size_t i = 1; i < target.size(); ++i) {
                const auto& prev = target[i-1];
                const auto& x = target[i];
                EXPECT_TRUE(prev.first < x.first); // no ties in this simulation.
                EXPECT_TRUE(col[mapping[universe[prev.second]]] < col[mapping[universe[x.second]]]);
                test_in_use.push_back(universe[x.second]);
            }

            // Checking that all features are represented.
            std::sort(test_in_use.begin(), test_in_use.end());
            EXPECT_EQ(test_in_use, local_universe);
        }
    }
}

TEST_P(IntegratedBuilderMoreTest, IntersectedCombineAgain) {
    // Mocking up the test and references.
    int ntop = GetParam();

    auto ids = simulate_test_ids();
    size_t seed = ntop * 100;
    std::vector<int> keep = simulate_ref_ids(seed);

    size_t nlabels = 3;
    auto mat = spawn_matrix(keep.size(), nsamples, seed);
    auto lab = spawn_labels(nsamples, nlabels, seed * 2);
    auto mrk = mock_markers(nlabels, 50, keep.size(), seed * 3);

    // Generating the prebuilts with and without an intersection. Note that we
    // can't just use build() to generate the no-intersection Prebuilt, as this
    // will not choose markers among the intersection of features.
    singlepp::BasicBuilder builder;
    builder.set_top(ntop);

    auto interpre = builder.run(ngenes, ids.data(), mat.get(), keep.data(), lab.data(), mrk);
    singlepp::BasicBuilder::Prebuilt pre(interpre.markers, interpre.ref_subset, interpre.references);

    // Applying the addition operation.
    singlepp::IntegratedBuilder inter;
    singlepp::IntegratedBuilder inter2;

    inter.add(ngenes, ids.data(), mat.get(), keep.data(), lab.data(), pre);
    inter2.add(ngenes, ids.data(), mat.get(), keep.data(), lab.data(), interpre);

    auto fin = inter.finish();
    auto fin2 = inter2.finish();

    // Checking for identical bits and pieces.
    EXPECT_EQ(fin.check_availability[0], fin2.check_availability[0]);

    std::vector<int> avail(fin.available[0].begin(), fin.available[0].end());
    std::sort(avail.begin(), avail.end());
    std::vector<int> avail2(fin2.available[0].begin(), fin2.available[0].end());
    std::sort(avail2.begin(), avail2.end());
    EXPECT_EQ(avail, avail2);

    const auto& markers = fin.markers[0];
    const auto& markers2 = fin2.markers[0];
    ASSERT_EQ(markers.size(), markers2.size());
    for (size_t m = 0; m < markers.size(); ++m) {
        EXPECT_EQ(markers[m], markers2[m]);
    }

    const auto& ranked = fin.ranked[0];
    const auto& ranked2 = fin2.ranked[0];
    ASSERT_EQ(ranked.size(), ranked2.size());
    for (size_t i = 0; i < ranked.size(); ++i) {
        ASSERT_EQ(ranked[i].size(), ranked2[i].size());
        for (size_t j = 0; j < ranked[i].size(); ++j) {
            EXPECT_EQ(ranked[i][j], ranked2[i][j]);
        }
    }
}

TEST_P(IntegratedBuilderMoreTest, IntersectedCombineNaked) {
    // Mocking up the test and references.
    int ntop = GetParam();

    auto ids = simulate_test_ids();
    size_t seed = ntop * 100;
    std::vector<int> keep = simulate_ref_ids(seed);

    size_t nlabels = 3;
    auto mat = spawn_matrix(keep.size(), nsamples, seed);
    auto lab = spawn_labels(nsamples, nlabels, seed * 2);
    auto mrk = mock_markers(nlabels, 50, keep.size(), seed * 3);

    singlepp::BasicBuilder builder;
    builder.set_top(ntop);
    auto pre = builder.run(mat.get(), lab.data(), mrk);

    singlepp::IntegratedBuilder alpha;
    alpha.add(ngenes, ids.data(), mat.get(), keep.data(), lab.data(), pre);
    auto aout = alpha.finish();

    // Comparing what happens with direct input of markers of interest.
    auto remarkers = truncate_markers(mrk, ntop);
    singlepp::IntegratedBuilder bravo;
    bravo.add(ngenes, ids.data(), mat.get(), keep.data(), lab.data(), remarkers);
    auto bout = bravo.finish();

    const auto& aranked = aout.ranked[0];
    const auto& branked = bout.ranked[0];

    ASSERT_EQ(aranked.size(), branked.size());
    for (size_t i = 0; i < aranked.size(); ++i) {
        ASSERT_EQ(aranked[i].size(), branked[i].size());
        for (size_t j = 0; j < aranked[i].size(); ++j) {
            EXPECT_EQ(aranked[i][j], branked[i][j]);
        }
    }
}

INSTANTIATE_TEST_CASE_P(
    IntegratedBuilder,
    IntegratedBuilderMoreTest,
    ::testing::Values(5, 10, 20) // number of top genes.
);
