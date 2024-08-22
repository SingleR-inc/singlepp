#include <gtest/gtest.h>
#include "custom_parallel.h"

#include "singlepp/classify_integrated.hpp"
#include "singlepp/classify_single.hpp"

#include "spawn_matrix.h"
#include "mock_markers.h"

class TrainIntegratedTestCore {
protected:
    inline static std::vector<std::shared_ptr<tatami::Matrix<double, int> > > references;
    inline static std::vector<std::vector<int> > labels;
    inline static std::vector<singlepp::Markers<int> > markers;

    inline static size_t ngenes = 2000;
    inline static size_t nsamples = 50;
    inline static size_t nrefs = 3;

    static void assemble() {
        if (references.size()) { 
            return;
        }
        for (size_t r = 0; r < nrefs; ++r) {
            size_t seed = r * 1000;
            size_t nlabels = 3 + r;

            references.push_back(spawn_matrix(ngenes, nsamples, seed));
            labels.push_back(spawn_labels(nsamples, nlabels, seed * 2));
            markers.push_back(mock_markers<int>(nlabels, 50, ngenes, seed * 3));
        }
    }

protected:
    static std::vector<int> simulate_ids(size_t ngenes, int seed) {
        std::vector<int> keep;
        keep.reserve(ngenes);
        std::mt19937_64 rng(seed);
        for (size_t s = 0; s < ngenes; ++s) {
            if (rng() % 100 < 90) {
                keep.push_back(s);
            } else {
                keep.push_back(-1); // i.e., unique to the reference.
            }
        }
        return keep;
    }

    static singlepp::Markers<int> truncate_markers(singlepp::Markers<int> remarkers, int ntop) {
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

class TrainIntegratedTest : public ::testing::TestWithParam<std::tuple<int, int> >, public TrainIntegratedTestCore {
protected:
    static void SetUpTestSuite() {
        assemble();
    }
};

TEST_P(TrainIntegratedTest, Simple) {
    auto param = GetParam();
    int ntop = std::get<0>(param);
    int nthreads = std::get<1>(param);

    singlepp::TrainSingleOptions<int, double> bopt;
    bopt.top = ntop;
    std::vector<singlepp::TrainIntegratedInput<double, int, int> > inputs;
    for (size_t r = 0; r < nrefs; ++r) {
        const auto& ref = *(references[r]);
        auto pre = singlepp::train_single(ref, labels[r].data(), markers[r], bopt);
        inputs.push_back(singlepp::prepare_integrated_input(ref, labels[r].data(), pre));
    }

    singlepp::TrainIntegratedOptions iopt;
    iopt.num_threads = nthreads;
    auto output = singlepp::train_integrated(std::move(inputs), iopt);

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
        auto wrk = references[r]->dense_column();
        std::vector<double> buffer(references[r]->nrow());

        for (size_t s = 0; s < nsamples; ++s) {
            int lab = labels[r][s]; 
            const auto& target = output.ranked[r][lab][offsets[lab]];
            ++offsets[lab];

            double last_original = -100000000;
            auto col = wrk->fetch(s, buffer.data());
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

TEST_P(TrainIntegratedTest, Intersect) {
    auto param = GetParam();
    int ntop = std::get<0>(param);
    int nthreads = std::get<1>(param);

    auto test_ids = simulate_ids(ngenes, ntop + nthreads);
    std::vector<std::vector<int> > ref_ids;
    for (size_t r = 0; r < nrefs; ++r) {
        size_t seed = (ntop + nthreads) * 10;
        ref_ids.push_back(simulate_ids(ngenes, seed));
    }

    // Adding each reference to the list. We store the single prebuilts for testing later.
    singlepp::TrainSingleOptions<int, double> bopt;
    bopt.top = ntop;
    std::vector<singlepp::TrainedSingleIntersect<int, double> > single_ref;
    std::vector<singlepp::TrainIntegratedInput<double, int, int> > inputs;

    auto idptr = test_ids.data();
    for (size_t r = 0; r < nrefs; ++r) {
        auto refptr = ref_ids[r].data();
        auto labptr = labels[r].data();
        const auto& refmat = *references[r];
        auto pre = singlepp::train_single_intersect<int>(ngenes, idptr, refmat, refptr, labptr, markers[r], bopt);
        inputs.push_back(singlepp::prepare_integrated_input_intersect<int>(ngenes, idptr, refmat, refptr, labptr, pre));
        single_ref.push_back(std::move(pre));
    }

    singlepp::TrainIntegratedOptions iopt;
    iopt.num_threads = nthreads;
    auto output = singlepp::train_integrated(std::move(inputs), iopt);

    EXPECT_EQ(output.num_references(), nrefs);
    const auto& universe = output.universe;

    for (size_t r = 0; r < nrefs; ++r) {
        EXPECT_TRUE(output.check_availability[r]);

        // Creating a mapping.
        std::unordered_map<int, int> mapping;
        const auto& keep = ref_ids[r];
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
        const auto& cur_markers = single_ref[r].get_markers();
        size_t nlabels = outmarkers.size();
        EXPECT_EQ(cur_markers.size(), nlabels);

        for (size_t l = 0; l < nlabels; ++l) {
            std::unordered_set<int> kept;
            for (size_t l2 = 0; l2 < nlabels; ++l2) {
                if (l != l2) {
                    const auto& current = cur_markers[l][l2];
                    for (auto x : current) {
                        kept.insert(single_ref[r].get_test_subset()[x]); // for comparison to universe indices, which are all relative to the test matrix.
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
        auto wrk = references[r]->dense_column();
        std::vector<double> buffer(references[r]->nrow());

        for (size_t s = 0; s < nsamples; ++s) {
            int lab = labels[r][s]; 
            const auto& target = output.ranked[r][lab][offsets[lab]];
            ++offsets[lab];

            double last_original = -100000000;
            auto col = wrk->fetch(s, buffer.data());
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

INSTANTIATE_TEST_SUITE_P(
    TrainIntegrated,
    TrainIntegratedTest,
    ::testing::Combine(
        ::testing::Values(5, 10, 20), // number of top genes.
        ::testing::Values(1, 3) // number of threads
    )
);
