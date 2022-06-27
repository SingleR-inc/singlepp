#include <gtest/gtest.h>
#include "custom_parallel.h"

#include "singlepp/IntegratedBuilder.hpp"
#include "spawn_matrix.h"
#include "mock_markers.h"

class IntegratedBuilderTest : public ::testing::TestWithParam<int> {
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

TEST_P(IntegratedBuilderTest, SimpleCombine) {
    // Mocking up the test and references.
    int ntop = GetParam();
    simulate_references();

    singlepp::Classifier runner;
    runner.set_top(ntop);
    singlepp::IntegratedBuilder inter;

    for (size_t r = 0; r < nrefs; ++r) {
        auto pre = runner.build(matrices[r].get(), labels[r].data(), markers[r]);
        inter.add(matrices[r].get(), labels[r].data(), pre);
    }

    auto output = inter.finish();

    // Checking the values of the built references.
    EXPECT_EQ(output.size(), nrefs);

    std::vector<int> in_use;
    for (size_t r = 0; r < nrefs; ++r) {
        EXPECT_FALSE(output[r].check_availability);

        size_t nlabels = 3 + r;
        EXPECT_EQ(output[r].markers.size(), nlabels);
        EXPECT_EQ(output[r].num_labels(), nlabels);
        EXPECT_EQ(output[r].num_profiles(), nsamples);

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
            auto copy = output[r].markers[l];
            std::sort(copy.begin(), copy.end());
            EXPECT_EQ(to_use, copy);
        }

        // Checking the ranked values.
        std::vector<int> offsets(nlabels);
        for (size_t s = 0; s < nsamples; ++s) {
            int lab = labels[r][s]; 
            const auto& target = output[r].ranked[lab][offsets[lab]];
            ++offsets[lab];

            double last_original = -100000000;
            auto col = matrices[r]->column(s);
            std::vector<int> test_in_use;
            test_in_use.push_back(target[0].second);

            for (size_t i = 1; i < target.size(); ++i) {
                const auto& prev = target[i-1];
                const auto& x = target[i];
                EXPECT_TRUE(prev.first < x.first); // no ties in this simulation.
                EXPECT_TRUE(col[prev.second] < col[x.second]);
                test_in_use.push_back(x.second);
            }

            // Checking that all values are represented.
            std::sort(test_in_use.begin(), test_in_use.end());
            if (in_use.size()) {
                EXPECT_EQ(test_in_use, in_use);
            } else {
                in_use.swap(test_in_use);
            }
        }
    }
}

TEST_P(IntegratedBuilderTest, SimpleCombineNaked) {
    // Mocking up the test and references.
    int ntop = GetParam();
    simulate_references();

    singlepp::Classifier runner;
    runner.set_top(ntop);
    singlepp::IntegratedBuilder alpha, bravo;

    for (size_t r = 0; r < nrefs; ++r) {
        auto pre = runner.build(matrices[r].get(), labels[r].data(), markers[r]);
        alpha.add(matrices[r].get(), labels[r].data(), pre);

        // Comparing what happens with direct input of markers of interest.
        auto remarkers = truncate_markers(markers[r], ntop);
        bravo.add(matrices[r].get(), labels[r].data(), remarkers);
    }

    auto alpha_output = alpha.finish();
    auto bravo_output = bravo.finish();

    // Checking the rank values for equality.
    for (size_t r = 0; r < nrefs; ++r) {
        const auto& cur_alpha = alpha_output[r];
        const auto& cur_bravo = bravo_output[r];

        ASSERT_EQ(cur_alpha.ranked.size(), cur_bravo.ranked.size());
        for (size_t l = 0; l < cur_alpha.ranked.size(); ++l) {
            const auto& rank_alpha = cur_alpha.ranked[l];
            const auto& rank_bravo = cur_bravo.ranked[l];
            ASSERT_EQ(rank_alpha.size(), rank_bravo.size());

            for (size_t s = 0; s < rank_alpha.size(); ++s) {
                EXPECT_EQ(rank_alpha[s], rank_bravo[s]);
            }
        }
    }
}

TEST_P(IntegratedBuilderTest, IntersectedCombine) {
    // Mocking up the test and references.
    int ntop = GetParam();
    simulate_references();

    auto ids = simulate_test_ids();
    std::vector<std::vector<int> > kept;
    for (size_t r = 0; r < nrefs; ++r) {
        size_t seed = r * 555;
        kept.push_back(simulate_ref_ids(seed));
    }

    singlepp::Classifier runner;
    runner.set_top(ntop);
    singlepp::IntegratedBuilder inter;

    // Adding each one to the list.
    for (size_t r = 0; r < nrefs; ++r) {
        auto pre = runner.build(ngenes, ids.data(), matrices[r].get(), kept[r].data(), labels[r].data(), markers[r]);
        inter.add(ngenes, ids.data(), matrices[r].get(), kept[r].data(), labels[r].data(), pre);
    }

    auto output = inter.finish();

    EXPECT_EQ(output.size(), nrefs);
    for (size_t r = 0; r < nrefs; ++r) {
        EXPECT_TRUE(output[r].check_availability);

        size_t nlabels = 3 + r;
        EXPECT_EQ(output[r].markers.size(), nlabels);
 
        // Creating a mapping.
        std::unordered_map<int, int> mapping;
        const auto& keep = kept[r];
        for (size_t g = 0; g < keep.size(); ++g) {
            if (keep[g] != -1) {
                mapping[keep[g]] = g;
            }
        }

        // Check consistency of the availability set.
        bool not_found = false;
        const auto& available = output[r].available;
        EXPECT_TRUE(available.size() > 0);

        for (const auto& a : available) {
            if (mapping.find(a) == mapping.end()) {
                not_found = true;
            }
        }
        EXPECT_FALSE(not_found);
        std::vector<int> in_use(available.begin(), available.end());
        std::sort(in_use.begin(), in_use.end());

        // Checking rankings for consistency with the availabilities.
        std::vector<int> offsets(nlabels);
        for (size_t s = 0; s < nsamples; ++s) {
            int lab = labels[r][s]; 
            const auto& target = output[r].ranked[lab][offsets[lab]];
            ++offsets[lab];

            double last_original = -100000000;
            auto col = matrices[r]->column(s);
            std::vector<int> test_in_use;
            test_in_use.push_back(target[0].second);

            for (size_t i = 1; i < target.size(); ++i) {
                const auto& prev = target[i-1];
                const auto& x = target[i];
                EXPECT_TRUE(prev.first < x.first); // no ties in this simulation.
                EXPECT_TRUE(col[mapping[prev.second]] < col[mapping[x.second]]);
                test_in_use.push_back(x.second);
            }

            // Checking that all values are represented.
            std::sort(test_in_use.begin(), test_in_use.end());
            EXPECT_EQ(test_in_use, in_use);
        }
    }
}

TEST_P(IntegratedBuilderTest, IntersectedCombineAgain) {
    // Mocking up the test and references.
    int ntop = GetParam();

    auto ids = simulate_test_ids();
    size_t seed = ntop * 100;
    std::vector<int> keep = simulate_ref_ids(seed);

    size_t nlabels = 3;
    auto mat = spawn_matrix(keep.size(), nsamples, seed);
    auto lab = spawn_labels(nsamples, nlabels, seed * 2);
    auto mrk = mock_markers(nlabels, 50, keep.size(), seed * 3);

    // Generating the prebuilts. Note that we can't just use build() to generate the Prebuilt,
    // as this will not choose markers among the intersection of features.
    singlepp::Classifier runner;
    runner.set_top(ntop);

    auto interpre = runner.build(ngenes, ids.data(), mat.get(), keep.data(), lab.data(), mrk);
    singlepp::Classifier::Prebuilt pre(interpre.markers, interpre.ref_subset, interpre.references);

    // Applying the addition operation.
    singlepp::IntegratedBuilder inter;
    singlepp::IntegratedBuilder inter2;

    inter.add(ngenes, ids.data(), mat.get(), keep.data(), lab.data(), pre);
    inter2.add(ngenes, ids.data(), mat.get(), keep.data(), lab.data(), interpre);

    auto fin = inter.finish()[0];
    auto fin2 = inter2.finish()[0];

    // Checking for identical bits and pieces.
    EXPECT_EQ(fin.check_availability, fin2.check_availability);

    std::vector<int> avail(fin.available.begin(), fin.available.end());
    std::sort(avail.begin(), avail.end());
    std::vector<int> avail2(fin2.available.begin(), fin2.available.end());
    std::sort(avail2.begin(), avail2.end());
    EXPECT_EQ(avail, avail2);

    ASSERT_EQ(fin.markers.size(), fin2.markers.size());
    for (size_t m = 0; m < fin.markers.size(); ++m) {
        EXPECT_EQ(fin.markers[m], fin2.markers[m]);
    }

    ASSERT_EQ(fin.ranked.size(), fin2.ranked.size());
    for (size_t i = 0; i < fin.ranked.size(); ++i) {
        ASSERT_EQ(fin.ranked[i].size(), fin2.ranked[i].size());
        for (size_t j = 0; j < fin.ranked[i].size(); ++j) {
            EXPECT_EQ(fin.ranked[i][j], fin2.ranked[i][j]);
        }
    }
}

TEST_P(IntegratedBuilderTest, IntersectedCombineNaked) {
    // Mocking up the test and references.
    int ntop = GetParam();

    auto ids = simulate_test_ids();
    size_t seed = ntop * 100;
    std::vector<int> keep = simulate_ref_ids(seed);

    size_t nlabels = 3;
    auto mat = spawn_matrix(keep.size(), nsamples, seed);
    auto lab = spawn_labels(nsamples, nlabels, seed * 2);
    auto mrk = mock_markers(nlabels, 50, keep.size(), seed * 3);

    singlepp::Classifier runner;
    runner.set_top(ntop);
    auto pre = runner.build(mat.get(), lab.data(), mrk);

    singlepp::IntegratedBuilder alpha;
    alpha.add(ngenes, ids.data(), mat.get(), keep.data(), lab.data(), pre);
    auto aout = alpha.finish()[0];

    // Comparing what happens with direct input of markers of interest.
    auto remarkers = truncate_markers(mrk, ntop);
    singlepp::IntegratedBuilder bravo;
    bravo.add(ngenes, ids.data(), mat.get(), keep.data(), lab.data(), remarkers);
    auto bout = bravo.finish()[0];

    ASSERT_EQ(aout.ranked.size(), bout.ranked.size());
    for (size_t i = 0; i < aout.ranked.size(); ++i) {
        ASSERT_EQ(aout.ranked[i].size(), bout.ranked[i].size());
        for (size_t j = 0; j < aout.ranked[i].size(); ++j) {
            EXPECT_EQ(aout.ranked[i][j], bout.ranked[i][j]);
        }
    }
}

INSTANTIATE_TEST_CASE_P(
    IntegratedBuilder,
    IntegratedBuilderTest,
    ::testing::Values(5, 10, 20) // number of top genes.
);
