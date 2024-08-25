#include <gtest/gtest.h>
#include "custom_parallel.h"

#include "singlepp/classify_integrated.hpp"
#include "singlepp/classify_single.hpp"

#include "spawn_matrix.h"
#include "mock_markers.h"
#include "naive_method.h"

class IntegratedTestCore {
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
    static std::vector<int> simulate_ids(size_t ngenes, int seed, int placeholder) {
        std::vector<int> keep;
        keep.reserve(ngenes);
        std::mt19937_64 rng(seed);
        for (size_t s = 0; s < ngenes; ++s) {
            if (rng() % 100 < 90) {
                keep.push_back(s);
            } else {
                keep.push_back(placeholder);
            }
        }
        return keep;
    }

    static std::vector<int> simulate_test_ids(size_t ngenes, int seed) {
        return simulate_ids(ngenes, seed, -2); // -2 => unique to the test dataset.
    }

    static constexpr int MISSING_REF_ID = -1;
    static std::vector<int> simulate_ref_ids(size_t ngenes, int seed) {
        return simulate_ids(ngenes, seed, MISSING_REF_ID); // -1 => unique to the reference.
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

class TrainIntegratedTest : public ::testing::TestWithParam<std::tuple<int, int> >, public IntegratedTestCore {
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

            auto col = wrk->fetch(s, buffer.data());
            std::vector<int> test_in_use;
            test_in_use.push_back(universe[target[0].second]);

            for (size_t i = 1; i < target.size(); ++i) {
                const auto& prev = target[i-1];
                const auto& x = target[i];
                EXPECT_LT(prev.first, x.first); // no ties in this simulation.
                EXPECT_LT(col[universe[prev.second]], col[universe[x.second]]);
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

    int base_seed = (ntop + nthreads) * 100;
    auto test_ids = simulate_test_ids(ngenes, base_seed * 10);

    std::vector<std::vector<int> > ref_ids;
    for (size_t r = 0; r < nrefs; ++r) {
        ref_ids.push_back(simulate_ref_ids(ngenes, base_seed * 20 + r));
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
            if (keep[g] != MISSING_REF_ID) {
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

            auto col = wrk->fetch(s, buffer.data());
            std::vector<int> test_in_use;
            test_in_use.push_back(universe[target[0].second]);

            for (size_t i = 1; i < target.size(); ++i) {
                const auto& prev = target[i-1];
                const auto& x = target[i];
                EXPECT_LT(prev.first, x.first); // no ties in this simulation.
                EXPECT_LT(col[mapping[universe[prev.second]]], col[mapping[universe[x.second]]]);
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

/********************************************/

class ClassifyIntegratedTest : public ::testing::TestWithParam<std::tuple<int, double> >, public IntegratedTestCore {
protected:
    static void SetUpTestSuite() {
        assemble();
        test = spawn_matrix(ngenes, ntest, 69);
    }

    inline static size_t ntest = 20;
    inline static std::shared_ptr<tatami::Matrix<double, int> > test;

protected:
    template<class Prebuilt_>
    static std::vector<std::vector<int> > mock_best_choices(size_t ntest, const std::vector<Prebuilt_>& prebuilts, size_t seed) {
        size_t nrefs = prebuilts.size();
        std::vector<std::vector<int> > chosen(nrefs);

        std::mt19937_64 rng(seed);
        for (size_t r = 0; r < nrefs; ++r) {
            size_t nlabels = prebuilts[r].get_markers().size();
            for (size_t t = 0; t < ntest; ++t) {
                chosen[r].push_back(rng() % nlabels);
            }
        }

        return chosen;
    }

    static auto split_by_labels(const std::vector<std::vector<int> >& labels) {
        std::vector<std::vector<std::vector<int> > > by_labels(labels.size());
        for (size_t r = 0, nrefs = labels.size(); r < nrefs; ++r) {
            const auto& current = labels[r];
            size_t nlabels = *std::max_element(current.begin(), current.end()) + 1;
            by_labels[r] = split_by_label(nlabels, current);
        }
        return by_labels;
    }

    template<class Prebuilt_>
    static std::unordered_set<int> create_universe(size_t cell, const std::vector<Prebuilt_>& prebuilts, const std::vector<std::vector<int> >& chosen) {
        std::unordered_set<int> tmp;
        for (size_t r = 0; r < prebuilts.size(); ++r) {
            const auto& pre = prebuilts[r];
            const auto& best_markers = pre.get_markers()[chosen[r][cell]];
            for (const auto& x : best_markers) {
                for (auto y : x) {
                    if constexpr(std::is_same<Prebuilt_, singlepp::TrainedSingle<int, double> >::value) {
                        tmp.insert(pre.get_subset()[y]);
                    } else {
                        tmp.insert(pre.get_test_subset()[y]);
                    }
                }
            }
        }
        return tmp;
    }

    static std::vector<double> quick_scaled_ranks(const std::vector<double>& col, const std::vector<int>& universe) {
        std::vector<double> copy;
        copy.reserve(universe.size());
        for (auto u : universe) {
            copy.push_back(col[u]);
        }
        return ::quick_scaled_ranks(copy);
    }
};

TEST_P(ClassifyIntegratedTest, Basic) {
    auto param = GetParam();
    int ntop = std::get<0>(param);
    double quantile = std::get<1>(param);
    int base_seed = ntop + quantile * 50;

    // Creating the integrated set of references.
    singlepp::TrainSingleOptions<int, double> bopt;
    bopt.top = ntop;
    std::vector<singlepp::TrainedSingle<int, double> > prebuilts;
    std::vector<singlepp::TrainIntegratedInput<double, int, int> > integrated_inputs;

    for (size_t r = 0; r < nrefs; ++r) {
        auto pre = singlepp::train_single(*(references[r]), labels[r].data(), markers[r], bopt);
        prebuilts.push_back(std::move(pre));
        integrated_inputs.push_back(singlepp::prepare_integrated_input(*(references[r]), labels[r].data(), prebuilts.back()));
    }

    singlepp::TrainIntegratedOptions iopt;
    auto integrated = singlepp::train_integrated(std::move(integrated_inputs), iopt);

    // Mocking up some of the best choices.
    auto chosen = mock_best_choices(ntest, prebuilts, /* seed = */ base_seed);
    std::vector<const int*> chosen_ptrs(nrefs);
    for (size_t r = 0; r < nrefs; ++r) {
        chosen_ptrs[r] = chosen[r].data();
    }

    // Comparing the classify_integrated output to a reference calculation.
    singlepp::ClassifyIntegratedOptions<double> copt;
    copt.fine_tune = false;
    copt.quantile = quantile;
    auto output = singlepp::classify_integrated<int>(*test, chosen_ptrs, integrated, copt);
    auto by_labels = split_by_labels(labels);
    auto wrk = test->dense_column();
    std::vector<double> buffer(test->nrow());

    for (size_t t = 0; t < ntest; ++t) {
        auto my_universe = create_universe(t, prebuilts, chosen);
        std::vector<int> universe(my_universe.begin(), my_universe.end());
        std::sort(universe.begin(), universe.end());

        auto col = wrk->fetch(t, buffer.data());
        tatami::copy_n(col, test->nrow(), buffer.data());
        auto scaled = quick_scaled_ranks(buffer, universe);

        std::vector<double> all_scores;
        for (size_t r = 0; r < nrefs; ++r) {
            double score = naive_score(scaled, by_labels[r][chosen[r][t]], references[r].get(), universe, quantile);
            EXPECT_FLOAT_EQ(score, output.scores[r][t]);
            all_scores.push_back(score);
        }

        auto best = std::max_element(all_scores.begin(), all_scores.end());
        EXPECT_EQ(output.best[t], best - all_scores.begin());

        double best_score = *best;
        *best = -100000;
        EXPECT_FLOAT_EQ(output.delta[t], best_score - *std::max_element(all_scores.begin(), all_scores.end()));
    }

    // Same results in parallel.
    copt.num_threads = 3;
    auto poutput = singlepp::classify_integrated<int>(*test, chosen_ptrs, integrated, copt);
    EXPECT_EQ(output.best, poutput.best);
    EXPECT_EQ(output.delta, poutput.delta);
    for (size_t r = 0; r < nrefs; ++r) {
        EXPECT_EQ(output.scores[r], poutput.scores[r]);
    }
}

TEST_P(ClassifyIntegratedTest, Intersected) {
    auto param = GetParam();
    int ntop = std::get<0>(param);
    double quantile = std::get<1>(param);
    int base_seed = ntop + quantile * 50;

    // Creating the integrated set of references.
    auto test_ids = simulate_test_ids(ngenes, base_seed * 20);

    singlepp::TrainSingleOptions<int, double> bopt;
    bopt.top = ntop;
    std::vector<std::vector<int> > ref_ids;
    std::vector<singlepp::TrainedSingleIntersect<int, double> > prebuilts;
    std::vector<singlepp::TrainIntegratedInput<double, int, int> > integrated_inputs;

    for (size_t r = 0; r < nrefs; ++r) {
        size_t seed = base_seed * 20 + r * 321;
        ref_ids.push_back(simulate_ref_ids(ngenes, seed + 3));
        const auto& ref_id = ref_ids.back();
        auto pre = singlepp::train_single_intersect<int>(ngenes, test_ids.data(), *(references[r]), ref_id.data(), labels[r].data(), markers[r], bopt);
        prebuilts.push_back(std::move(pre));
        integrated_inputs.push_back(singlepp::prepare_integrated_input_intersect<int>(ngenes, test_ids.data(), *(references[r]), ref_id.data(), labels[r].data(), prebuilts.back()));
    }

    singlepp::TrainIntegratedOptions iopt;
    auto integrated = singlepp::train_integrated(std::move(integrated_inputs), iopt);

    // Mocking up some of the best choices.
    auto chosen = mock_best_choices(ntest, prebuilts, base_seed + 2468);
    std::vector<const int*> chosen_ptrs(nrefs);
    for (size_t r = 0; r < nrefs; ++r) {
        chosen_ptrs[r] = chosen[r].data();
    }

    // Comparing the ClassifyIntegrated to a reference calculation.
    singlepp::ClassifyIntegratedOptions<double> copt;
    copt.fine_tune = false;
    copt.quantile = quantile;
    auto output = singlepp::classify_integrated<int>(*test, chosen_ptrs, integrated, copt);
    auto by_labels = split_by_labels(labels);

    std::vector<std::unordered_map<int, int> > reverser(nrefs);
    for (size_t r = 0; r < nrefs; ++r) {
        const auto& ref_id = ref_ids[r];
        for (size_t i = 0; i < ref_id.size(); ++i) {
            if (ref_id[i] != MISSING_REF_ID) {
                reverser[r][ref_id[i]] = i;
            }
        }
    }

    auto wrk = test->dense_column();
    std::vector<double> buffer(test->nrow());
    for (size_t t = 0; t < ntest; ++t) {
        auto my_universe = create_universe(t, prebuilts, chosen);

        std::vector<double> all_scores;
        for (size_t r = 0; r < nrefs; ++r) {
            std::vector<int> universe_test, universe_ref;
            for (auto s : my_universe) {
                auto it = reverser[r].find(s);
                if (it != reverser[r].end()) {
                    universe_test.push_back(s);
                    universe_ref.push_back(it->second);
                }
            }

            auto col = wrk->fetch(t, buffer.data());
            tatami::copy_n(col, buffer.size(), buffer.data());
            auto scaled = quick_scaled_ranks(buffer, universe_test);

            double score = naive_score(scaled, by_labels[r][chosen[r][t]], references[r].get(), universe_ref, quantile);
            EXPECT_FLOAT_EQ(score, output.scores[r][t]);
            all_scores.push_back(score);
        }

        auto best = std::max_element(all_scores.begin(), all_scores.end());
        EXPECT_EQ(output.best[t], best - all_scores.begin());

        double best_score = *best;
        *best = -100000;
        EXPECT_FLOAT_EQ(output.delta[t], best_score - *std::max_element(all_scores.begin(), all_scores.end()));
    }

    // Same results in parallel.
    copt.num_threads = 3;
    auto poutput = singlepp::classify_integrated<int>(*test, chosen_ptrs, integrated, copt);
    EXPECT_EQ(output.best, poutput.best);
    EXPECT_EQ(output.delta, poutput.delta);
    for (size_t r = 0; r < nrefs; ++r) {
        EXPECT_EQ(output.scores[r], poutput.scores[r]);
    }
}

TEST_P(ClassifyIntegratedTest, IntersectedComparison) {
    auto param = GetParam();
    int ntop = std::get<0>(param);
    double quantile = std::get<1>(param);
    int base_seed = ntop + quantile * 50;

    // Creating the integrated set of references with intersection, along with
    // a comparison to the simple method where we do the reorganization externally.
    // The aim is to check that all the various gene indexing steps are done correctly.
    std::vector<int> test_ids(ngenes);
    std::iota(test_ids.begin(), test_ids.end(), 0);
    std::vector<std::shared_ptr<tatami::Matrix<double, int> > > reorganized_references;

    singlepp::TrainSingleOptions<int, double> bopt;
    bopt.top = ntop;
    std::vector<singlepp::TrainedSingleIntersect<int, double> > prebuilts;
    std::vector<singlepp::TrainIntegratedInput<double, int, int> > integrated_inputs;
    std::vector<singlepp::TrainIntegratedInput<double, int, int> > reorganized_integrated_inputs;

    std::mt19937_64 rng(base_seed);
    for (size_t r = 0; r < nrefs; ++r) {
        auto ref_ids = test_ids;
        std::shuffle(ref_ids.begin(), ref_ids.end(), rng);

        auto pre = singlepp::train_single_intersect<int>(ngenes, test_ids.data(), *(references[r]), ref_ids.data(), labels[r].data(), markers[r], bopt);
        integrated_inputs.push_back(singlepp::prepare_integrated_input_intersect<int>(ngenes, test_ids.data(), *(references[r]), ref_ids.data(), labels[r].data(), pre));
        prebuilts.push_back(std::move(pre));

        // Doing a reference calculation with reorganization before calling the non-intersection methods.
        std::vector<int> remapping(ngenes);
        for (size_t i = 0; i < ngenes; ++i) {
            remapping[ref_ids[i]] = i;
        }
        auto mcopy = markers[r];
        for (auto& mm : mcopy) {
            for (auto& m : mm) {
                for (auto& x : m) {
                    x = ref_ids[x];
                }
            }
        }
        auto shuffled = tatami::make_DelayedSubset(references[r], std::move(remapping), true);
        auto pre_reorg = singlepp::train_single(*shuffled, labels[r].data(), std::move(mcopy), bopt);
        reorganized_integrated_inputs.push_back(singlepp::prepare_integrated_input(*shuffled, labels[r].data(), pre_reorg));
        reorganized_references.push_back(std::move(shuffled));
    }

    singlepp::TrainIntegratedOptions iopt;
    auto integrated = singlepp::train_integrated(std::move(integrated_inputs), iopt);
    auto reorganized_integrated = singlepp::train_integrated(std::move(reorganized_integrated_inputs), iopt);

    // Mocking up some of the best choices.
    auto chosen = mock_best_choices(ntest, prebuilts, base_seed + 2468);
    std::vector<const int*> chosen_ptrs(nrefs);
    for (size_t r = 0; r < nrefs; ++r) {
        chosen_ptrs[r] = chosen[r].data();
    }

    // Classifying, even with fine-tuning.
    singlepp::ClassifyIntegratedOptions<double> copt;
    copt.quantile = quantile;
    auto output = singlepp::classify_integrated<int>(*test, chosen_ptrs, integrated, copt);
    auto reorganized_output = singlepp::classify_integrated<int>(*test, chosen_ptrs, reorganized_integrated, copt);

    EXPECT_EQ(output.best, reorganized_output.best);
    EXPECT_EQ(output.scores, reorganized_output.scores);
    EXPECT_EQ(output.delta, reorganized_output.delta);
}

INSTANTIATE_TEST_SUITE_P(
    ClassifyIntegrated,
    ClassifyIntegratedTest,
    ::testing::Combine(
        ::testing::Values(5, 10, 20), // number of top genes.
        ::testing::Values(0.5, 0.8, 0.9) // number of quantiles.
    )
);
