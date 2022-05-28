#include <gtest/gtest.h>

#include "singlepp/IntegratedBuilder.hpp"
#include "spawn_matrix.h"
#include "mock_markers.h"

class IntegratedBuilderTest : public ::testing::TestWithParam<int> {};

TEST_P(IntegratedBuilderTest, SimpleCombine) {
    // Mocking up the test and references.
    size_t ngenes = 2000;
    size_t nsamples = 50;
    size_t nrefs = 3;
    int ntop = GetParam();

    std::vector<std::shared_ptr<tatami::Matrix<double, int> > > matrices;
    std::vector<std::vector<int> > labels;
    std::vector<singlepp::Markers> markers;
    singlepp::SinglePP runner;
    runner.set_top(ntop);
    singlepp::IntegratedBuilder inter;

    for (size_t r = 0; r < nrefs; ++r) {
        size_t seed = r * 1000;
        size_t nlabels = 3 + r;

        matrices.push_back(spawn_matrix(ngenes, nsamples, seed));
        labels.push_back(spawn_labels(nsamples, nlabels, seed * 2));
        markers.push_back(mock_markers(nlabels, 50, ngenes, seed * 3));

        // Adding each one to the list.
        auto pre = runner.build(matrices.back().get(), labels.back().data(), markers.back());
        inter.add(matrices.back().get(), labels.back().data(), pre);
    }

    auto output = inter.finish();

    std::vector<int> in_use;
    EXPECT_EQ(output.size(), nrefs);
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

TEST_P(IntegratedBuilderTest, IntersectedCombine) {
    // Mocking up the test and references.
    size_t ngenes = 2000;
    size_t nsamples = 50;
    size_t nrefs = 3;
    int ntop = GetParam();

    std::vector<std::shared_ptr<tatami::Matrix<double, int> > > matrices;
    std::vector<std::vector<int> > labels;
    std::vector<singlepp::Markers> markers;

    std::vector<int> ids(ngenes);
    for (size_t g = 0; g < ngenes; ++g) {
        ids[g] = g;
    }
    std::vector<std::vector<int> > kept(nrefs);

    singlepp::SinglePP runner;
    runner.set_top(ntop);
    singlepp::IntegratedBuilder inter;

    for (size_t r = 0; r < nrefs; ++r) {
        size_t seed = r * 100;
        size_t nlabels = 3 + r;

        std::mt19937_64 rng(seed);
        auto& keep = kept[r];
        for (size_t s = 0; s < ngenes; ++s) {
            if (rng() % 100 < 50) {
                keep.push_back(s);
            } else {
                keep.push_back(-1); // i.e., unique to the reference.
            }
        }

        matrices.push_back(spawn_matrix(keep.size(), nsamples, seed));
        labels.push_back(spawn_labels(nsamples, nlabels, seed * 2));
        markers.push_back(mock_markers(nlabels, 50, keep.size(), seed * 3));
        
        // Adding each one to the list.
        auto pre = runner.build(ngenes, ids.data(), matrices.back().get(), keep.data(), labels.back().data(), markers.back());
        inter.add(ngenes, ids.data(), matrices.back().get(), keep.data(), labels.back().data(), pre);
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
    size_t ngenes = 2000;
    size_t nsamples = 50;
    size_t nrefs = 3;
    int ntop = GetParam();

    std::vector<int> ids(ngenes);
    for (size_t g = 0; g < ngenes; ++g) {
        ids[g] = g;
    }
    std::vector<std::vector<int> > kept(nrefs);

    size_t seed = ntop * 100;
    size_t nlabels = 3;

    std::mt19937_64 rng(seed);
    std::vector<int> keep;
    for (size_t s = 0; s < ngenes; ++s) {
        if (rng() % 100 < 50) {
            keep.push_back(s);
        } else {
            keep.push_back(-1); // i.e., unique to the reference.
        }
    }

    auto mat = spawn_matrix(keep.size(), nsamples, seed);
    auto lab = spawn_labels(nsamples, nlabels, seed * 2);
    auto mrk = mock_markers(nlabels, 50, keep.size(), seed * 3);

    // Generating the prebuilts. Note that we can't just use build() to generate the Prebuilt,
    // as this will not choose markers among the intersection of features.
    singlepp::SinglePP runner;
    runner.set_top(ntop);

    auto interpre = runner.build(ngenes, ids.data(), mat.get(), keep.data(), lab.data(), mrk);
    singlepp::SinglePP::Prebuilt pre(interpre.markers, interpre.ref_subset, interpre.references);

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
        for (size_t j = 0; j < fin.ranked[i].size(); ++j) {
            EXPECT_EQ(fin.ranked[i][j], fin2.ranked[i][j]);
        }
    }
}

INSTANTIATE_TEST_CASE_P(
    IntegratedBuilder,
    IntegratedBuilderTest,
    ::testing::Values(5, 10, 20) // number of top genes.
);
