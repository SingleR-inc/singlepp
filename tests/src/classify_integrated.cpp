#include <gtest/gtest.h>

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

    static constexpr int MISSING_TEST_ID = -2;
    static std::vector<int> simulate_test_ids(size_t ngenes, int seed) {
        return simulate_ids(ngenes, seed, MISSING_TEST_ID); // -2 => unique to the test dataset.
    }

    static constexpr int MISSING_REF_ID = -1;
    static std::vector<int> simulate_ref_ids(size_t ngenes, int seed) {
        return simulate_ids(ngenes, seed, MISSING_REF_ID); // -1 => unique to the reference(s), i.e., won't overlap with the test.
    }
};

///********************************************/
//
//class TrainIntegratedTest : public ::testing::TestWithParam<std::tuple<int, int> >, public IntegratedTestCore {
//protected:
//    static void SetUpTestSuite() {
//        assemble();
//    }
//};
//
//TEST_P(TrainIntegratedTest, Simple) {
//    auto param = GetParam();
//    int ntop = std::get<0>(param);
//    int nthreads = std::get<1>(param);
//
//    singlepp::TrainSingleOptions bopt;
//    bopt.top = ntop;
//    std::vector<singlepp::TrainIntegratedInput<double, int, int> > inputs;
//    for (size_t r = 0; r < nrefs; ++r) {
//        const auto& ref = *(references[r]);
//        auto pre = singlepp::train_single(ref, labels[r].data(), markers[r], bopt);
//        inputs.push_back(singlepp::prepare_integrated_input(ref, labels[r].data(), pre));
//    }
//
//    singlepp::TrainIntegratedOptions iopt;
//    iopt.num_threads = nthreads;
//    auto output = singlepp::train_integrated(std::move(inputs), iopt);
//
//    // Checking the values of the built references.
//    EXPECT_EQ(output.num_references(), nrefs);
//    const auto& universe = output.universe;
//
//    for (size_t r = 0; r < nrefs; ++r) {
//        size_t nlabels = 3 + r;
//        EXPECT_EQ(output.my_references[r].size(), nlabels);
//        EXPECT_EQ(output.num_labels(r), nlabels);
//        EXPECT_EQ(output.num_profiles(r), nsamples);
//
//        // Checking the contents of the markers. 
//        const auto& cur_markers = markers[r];
//        EXPECT_EQ(cur_markers.size(), nlabels);
//
//        for (size_t l = 0; l < nlabels; ++l) {
//            std::unordered_set<int> kept;
//            for (size_t l2 = 0; l2 < nlabels; ++l2) {
//                if (l != l2) {
//                    const auto& current = cur_markers[l][l2];
//                    kept.insert(current.begin(), current.begin() + ntop);
//                }
//            }
//
//            std::vector<int> to_use(kept.begin(), kept.end());
//            std::sort(to_use.begin(), to_use.end());
//
//            auto copy = output.my_references[r][l].markers;
//            for (auto& x : copy) {
//                x = universe[x];
//            }
//            std::sort(copy.begin(), copy.end());
//
//            EXPECT_EQ(to_use, copy);
//        }
//
//        // Checking the ranked values.
//        std::vector<int> offsets(nlabels);
//        auto wrk = references[r]->dense_column();
//        std::vector<double> buffer(references[r]->nrow());
//
//        for (size_t s = 0; s < nsamples; ++s) {
//            int lab = labels[r][s]; 
//            auto target_start = output.my_references[r][lab].all_ranked.begin() + static_cast<std::size_t>(offsets[lab]) * output.universe.size();
//            auto target_end = target_start + output.universe.size();
//            ++offsets[lab];
//
//            auto col = wrk->fetch(s, buffer.data());
//            std::vector<int> test_in_use;
//            test_in_use.push_back(universe[target_start->second]);
//
//            for (auto it = target_start + 1; it != target_end; ++it) {
//                const auto& prev = *(it - 1);
//                const auto& x = *it;
//                EXPECT_LT(prev.first, x.first); // no ties in this simulation.
//                EXPECT_LT(col[universe[prev.second]], col[universe[x.second]]);
//                test_in_use.push_back(universe[x.second]);
//            }
//
//            // Checking that all features are represented.
//            std::sort(test_in_use.begin(), test_in_use.end());
//            EXPECT_EQ(test_in_use, universe);
//        }
//    }
//}
//
//TEST_P(TrainIntegratedTest, Intersect) {
//    auto param = GetParam();
//    int ntop = std::get<0>(param);
//    int nthreads = std::get<1>(param);
//
//    int base_seed = (ntop + nthreads) * 100;
//    auto test_ids = simulate_test_ids(ngenes, base_seed * 10);
//
//    std::vector<std::vector<int> > ref_ids;
//    for (size_t r = 0; r < nrefs; ++r) {
//        ref_ids.push_back(simulate_ref_ids(ngenes, base_seed * 20 + r));
//    }
//
//    // Adding each reference to the list. We store the single prebuilts for testing later.
//    singlepp::TrainSingleOptions bopt;
//    bopt.top = ntop;
//    std::vector<singlepp::TrainedSingleIntersect<int, double> > single_ref;
//    std::vector<singlepp::TrainIntegratedInput<double, int, int> > inputs;
//
//    auto idptr = test_ids.data();
//    for (size_t r = 0; r < nrefs; ++r) {
//        auto refptr = ref_ids[r].data();
//        auto labptr = labels[r].data();
//        const auto& refmat = *references[r];
//        auto pre = singlepp::train_single_intersect<double, int>(ngenes, idptr, refmat, refptr, labptr, markers[r], bopt);
//        inputs.push_back(singlepp::prepare_integrated_input_intersect<int>(ngenes, idptr, refmat, refptr, labptr, pre));
//        single_ref.push_back(std::move(pre));
//    }
//
//    singlepp::TrainIntegratedOptions iopt;
//    iopt.num_threads = nthreads;
//    auto output = singlepp::train_integrated(std::move(inputs), iopt);
//
//    EXPECT_EQ(output.num_references(), nrefs);
//    const auto& universe = output.universe;
//
//    for (size_t r = 0; r < nrefs; ++r) {
//        // Creating a mapping.
//        std::unordered_map<int, int> mapping;
//        const auto& keep = ref_ids[r];
//        for (size_t g = 0; g < keep.size(); ++g) {
//            if (keep[g] != MISSING_REF_ID) {
//                mapping[keep[g]] = g;
//            }
//        }
//
//        // Check consistency of the markers.
//        const auto& outmarkers = output.markers[r];
//        const auto& cur_markers = single_ref[r].get_markers();
//        size_t nlabels = outmarkers.size();
//        EXPECT_EQ(cur_markers.size(), nlabels);
//
//        for (size_t l = 0; l < nlabels; ++l) {
//            std::unordered_set<int> kept;
//            for (size_t l2 = 0; l2 < nlabels; ++l2) {
//                if (l != l2) {
//                    const auto& current = cur_markers[l][l2];
//                    for (auto x : current) {
//                        kept.insert(single_ref[r].get_test_subset()[x]); // for comparison to universe indices, which are all relative to the test matrix.
//                    }
//                }
//            }
//
//            int not_found = 0;
//            for (auto& x : outmarkers[l]) {
//                not_found += (kept.find(universe[x]) == kept.end());
//            }
//            EXPECT_EQ(not_found, 0);
//        }
//
//        // Checking rankings for consistency with the availabilities.
//        std::vector<int> offsets(nlabels);
//        auto wrk = references[r]->dense_column();
//        std::vector<double> buffer(references[r]->nrow());
//
//        for (size_t s = 0; s < nsamples; ++s) {
//            int lab = labels[r][s]; 
//            const auto& target = output.ranked[r][lab][offsets[lab]];
//            ++offsets[lab];
//
//            auto col = wrk->fetch(s, buffer.data());
//            std::vector<int> test_in_use;
//            test_in_use.push_back(universe[target[0].second]);
//
//            for (size_t i = 1; i < target.size(); ++i) {
//                const auto& prev = target[i-1];
//                const auto& x = target[i];
//                EXPECT_LT(prev.first, x.first); // no ties in this simulation.
//                EXPECT_LT(col[mapping[universe[prev.second]]], col[mapping[universe[x.second]]]);
//                test_in_use.push_back(universe[x.second]);
//            }
//
//            // Checking that all features are represented.
//            std::sort(test_in_use.begin(), test_in_use.end());
//            EXPECT_EQ(test_in_use, local_universe);
//        }
//    }
//}
//
//INSTANTIATE_TEST_SUITE_P(
//    TrainIntegrated,
//    TrainIntegratedTest,
//    ::testing::Combine(
//        ::testing::Values(5, 10, 20), // number of top genes.
//        ::testing::Values(1, 3) // number of threads
//    )
//);
//
///********************************************/
//
//class TrainIntegratedMismatchTest : public ::testing::Test, public IntegratedTestCore {
//protected:
//    inline static std::vector<std::shared_ptr<tatami::Matrix<double, int> > > sub_references;
//
//    void SetUp() {
//        assemble();
//        for (size_t r = 0; r < nrefs; ++r) {
//            sub_references.emplace_back(new tatami::DelayedSubsetBlock<double, int>(references[r], r, ngenes - r, true));
//        }
//    }
//};
//
//TEST_F(TrainIntegratedMismatchTest, Simple) {
//    singlepp::TrainSingleOptions bopt;
//    std::vector<singlepp::TrainIntegratedInput<double, int, int> > inputs;
//    for (size_t r = 0; r < nrefs; ++r) {
//        const auto& sref = *(sub_references[r]);
//        auto pre = singlepp::train_single(sref, labels[r].data(), markers[r], bopt);
//        inputs.push_back(singlepp::prepare_integrated_input(sref, labels[r].data(), pre));
//    }
//
//    bool failed = false;
//    singlepp::TrainIntegratedOptions iopt;
//    try {
//        singlepp::train_integrated(std::move(inputs), iopt);
//    } catch (std::exception& e) {
//        EXPECT_TRUE(std::string(e.what()).find("inconsistent number of rows") != std::string::npos);
//        failed = true;
//    }
//    EXPECT_TRUE(failed);
//}
//
//TEST_F(TrainIntegratedMismatchTest, Intersect) {
//    singlepp::TrainSingleOptions bopt;
//    std::vector<singlepp::TrainedSingleIntersect<int, double> > single_ref;
//    std::vector<singlepp::TrainIntegratedInput<double, int, int> > inputs;
//
//    int base_seed = 6969;
//    for (size_t r = 0; r < nrefs; ++r) {
//        const auto& srefmat = *sub_references[r];
//        size_t curgenes = srefmat.nrow();
//        auto test_ids = simulate_test_ids(curgenes, base_seed * 10 + r);
//        auto ref_ids = simulate_ref_ids(curgenes, base_seed * 20 + r);
//        auto labptr = labels[r].data();
//        auto pre = singlepp::train_single_intersect<double, int>(test_ids.size(), test_ids.data(), srefmat, ref_ids.data(), labptr, markers[r], bopt);
//        inputs.push_back(singlepp::prepare_integrated_input_intersect<int>(curgenes, test_ids.data(), srefmat, ref_ids.data(), labptr, pre));
//        single_ref.push_back(std::move(pre));
//    }
//
//    bool failed = false;
//    singlepp::TrainIntegratedOptions iopt;
//    try {
//        singlepp::train_integrated(std::move(inputs), iopt);
//    } catch (std::exception& e) {
//        EXPECT_TRUE(std::string(e.what()).find("inconsistent number of rows") != std::string::npos);
//        failed = true;
//    }
//    EXPECT_TRUE(failed);
//}

/********************************************/

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

class ClassifyIntegratedTest : public ::testing::TestWithParam<std::tuple<int, double> >, public IntegratedTestCore {
protected:
    static void SetUpTestSuite() {
        assemble();
        test = spawn_matrix(ngenes, ntest, 69);
    }

    inline static size_t ntest = 20;
    inline static std::shared_ptr<tatami::Matrix<double, int> > test;

protected:
    static auto split_by_labels(const std::vector<std::vector<int> >& labels) {
        std::vector<std::vector<std::vector<int> > > by_labels(labels.size());
        for (size_t r = 0, nrefs = labels.size(); r < nrefs; ++r) {
            const auto& current = labels[r];
            size_t nlabels = *std::max_element(current.begin(), current.end()) + 1;
            by_labels[r] = split_by_label(nlabels, current);
        }
        return by_labels;
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
    singlepp::TrainSingleOptions bopt;
    bopt.top = ntop;

    std::vector<singlepp::TrainedSingle<int, double> > prebuilts;
    prebuilts.reserve(nrefs); // ensure that no reallocations happen that might invalidate pointers in the integrated_inputs.
    std::vector<singlepp::TrainIntegratedInput<double, int, int> > integrated_inputs;
    integrated_inputs.reserve(nrefs);

    for (size_t r = 0; r < nrefs; ++r) {
        const auto labptr = labels[r].data();
        const auto& refmat = *(references[r]);
        prebuilts.push_back(singlepp::train_single(refmat, labptr, markers[r], bopt));
        integrated_inputs.push_back(singlepp::prepare_integrated_input(refmat, labptr, prebuilts.back()));
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
    // This requires disabling of fine-tuning as there's no easy way to test that.
    singlepp::ClassifyIntegratedOptions<double> copt;
    copt.fine_tune = false;
    copt.quantile = quantile;
    auto output = singlepp::classify_integrated<int>(*test, chosen_ptrs, integrated, copt);

    {
        auto by_labels = split_by_labels(labels);
        auto wrk = test->dense_column();
        std::vector<double> buffer(test->nrow());

        for (size_t t = 0; t < ntest; ++t) {
            std::unordered_set<int> tmp;
            for (size_t r = 0; r < prebuilts.size(); ++r) {
                const auto& pre = prebuilts[r];
                const auto& best_markers = pre.get_markers()[chosen[r][t]];
                for (const auto& x : best_markers) {
                    for (auto y : x) {
                        tmp.insert(pre.get_subset()[y]);
                    }
                }
            }
            std::vector<int> universe(tmp.begin(), tmp.end());
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
    }

    // Same results in parallel.
    {
        copt.num_threads = 3;
        auto poutput = singlepp::classify_integrated<int>(*test, chosen_ptrs, integrated, copt);
        EXPECT_EQ(output.best, poutput.best);
        EXPECT_EQ(output.delta, poutput.delta);
        for (size_t r = 0; r < nrefs; ++r) {
            EXPECT_EQ(output.scores[r], poutput.scores[r]);
        }
    }
}

TEST_P(ClassifyIntegratedTest, Intersected) {
    auto param = GetParam();
    int ntop = std::get<0>(param);
    double quantile = std::get<1>(param);
    int base_seed = ntop + quantile * 50;

    // Creating the integrated set of references.
    auto test_ids = simulate_test_ids(ngenes, base_seed * 20);
    singlepp::TrainSingleOptions bopt;
    bopt.top = ntop;

    std::vector<std::vector<int> > ref_ids;
    ref_ids.reserve(nrefs);
    std::vector<singlepp::TrainedSingleIntersect<int, double> > prebuilts;
    prebuilts.reserve(nrefs); // ensure that no reallocations happen that might invalidate pointers in the integrated_inputs.
    std::vector<singlepp::TrainIntegratedInput<double, int, int> > integrated_inputs;
    integrated_inputs.reserve(nrefs);

    for (size_t r = 0; r < nrefs; ++r) {
        size_t seed = base_seed * 20 + r * 321;
        ref_ids.push_back(simulate_ref_ids(ngenes, seed + 3));
        const auto refptr = ref_ids.back().data();

        const auto testptr = test_ids.data();
        const auto labptr = labels[r].data();
        const auto& refmat = *(references[r]);
        prebuilts.push_back(singlepp::train_single_intersect<double, int>(ngenes, testptr, refmat, refptr, labptr, markers[r], bopt));
        integrated_inputs.push_back(singlepp::prepare_integrated_input_intersect<int>(ngenes, testptr, refmat, refptr, labptr, prebuilts.back()));
    }

    singlepp::TrainIntegratedOptions iopt;
    auto integrated = singlepp::train_integrated(std::move(integrated_inputs), iopt);

    // Mocking up some of the best choices.
    auto chosen = mock_best_choices(ntest, prebuilts, base_seed + 2468);
    std::vector<const int*> chosen_ptrs(nrefs);
    for (size_t r = 0; r < nrefs; ++r) {
        chosen_ptrs[r] = chosen[r].data();
    }

    // Comparing to a reference calculation.
    // This requires disabling of fine-tuning as there's no easy way to test that.
    singlepp::ClassifyIntegratedOptions<double> copt;
    copt.fine_tune = false;
    copt.quantile = quantile;
    auto output = singlepp::classify_integrated<int>(*test, chosen_ptrs, integrated, copt);
    auto by_labels = split_by_labels(labels);

    {
        std::vector<singlepp::Intersection<int> > intersections;
        for (size_t r = 0; r < nrefs; ++r) {
            intersections.push_back(singlepp::intersect_genes<int>(test_ids.size(), test_ids.data(), ref_ids[r].size(), ref_ids[r].data()));
        }
        std::unordered_map<int, std::size_t> availability;
        for (size_t r = 0; r < nrefs; ++r) {
            for (const auto& ii : intersections[r]) {
                auto it = availability.find(ii.second);
                if (it == availability.end()) {
                    availability[ii.second] = 1;
                } else {
                    it->second += 1;
                }
            }
        }

        auto wrk = test->dense_column();
        std::vector<double> buffer(test->nrow());
        for (size_t t = 0; t < ntest; ++t) {

            // First, we find the current set of markers for the assigned labels of test sample 't',
            // filtering it to only those genes that are present in all references.
            std::unordered_set<int> current_markers;
            for (std::size_t r = 0; r < nrefs; ++r) {
                const auto& pre = prebuilts[r];
                const auto& best_markers = pre.get_markers()[chosen[r][t]];
                for (const auto& x : best_markers) {
                    for (auto y : x) {
                        const auto yref = pre.get_ref_subset()[y];
                        const auto aIt = availability.find(yref);
                        if (aIt != availability.end() && aIt->second == nrefs) {
                            current_markers.insert(yref);
                        }
                    }
                }
            }
            std::vector<int> universe_ref(current_markers.begin(), current_markers.end());
            std::sort(universe_ref.begin(), universe_ref.end()); 
            std::unordered_map<int, int> map_to_universe;
            for (std::size_t u = 0; u < universe_ref.size(); ++u) {
                map_to_universe[universe_ref[u]] = u;
            }

            std::vector<double> all_scores;
            for (std::size_t r = 0; r < nrefs; ++r) {
                // Now, we traverse the intersections to figure out the test rows that we need to match universe_ref.
                std::vector<int> universe_test(universe_ref.size());
                for (const auto& ii : intersections[r]) {
                    auto it = map_to_universe.find(ii.second);
                    if (it != map_to_universe.end()) {
                        universe_test[it->second] = ii.first;
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
    }

    // Same results in parallel.
    {
        copt.num_threads = 3;
        auto poutput = singlepp::classify_integrated<int>(*test, chosen_ptrs, integrated, copt);
        EXPECT_EQ(output.best, poutput.best);
        EXPECT_EQ(output.delta, poutput.delta);
        for (size_t r = 0; r < nrefs; ++r) {
            EXPECT_EQ(output.scores[r], poutput.scores[r]);
        }
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

    singlepp::TrainSingleOptions bopt;
    bopt.top = ntop;
    std::vector<singlepp::TrainedSingleIntersect<int, double> > prebuilts;
    prebuilts.reserve(nrefs);
    std::vector<singlepp::TrainIntegratedInput<double, int, int> > integrated_inputs;
    integrated_inputs.reserve(nrefs);

    std::vector<std::shared_ptr<tatami::Matrix<double, int> > > references_reorg;
    references_reorg.reserve(nrefs);
    std::vector<singlepp::TrainedSingle<int, double> > prebuilts_reorg;
    prebuilts_reorg.reserve(nrefs);
    std::vector<singlepp::TrainIntegratedInput<double, int, int> > integrated_inputs_reorg;
    integrated_inputs_reorg.reserve(nrefs);

    std::vector<singlepp::TrainIntegratedInput<double, int, int> > integrated_inputs_shuffled;
    integrated_inputs_shuffled.reserve(nrefs);
    std::vector<singlepp::Intersection<int> > intersections_shuffled;
    intersections_shuffled.resize(nrefs);

    std::mt19937_64 rng(base_seed);
    for (size_t r = 0; r < nrefs; ++r) {
        const auto& refmat = *(references[r]);
        const auto labptr = labels[r].data();

        auto testptr = test_ids.data();
        auto ref_ids = test_ids;
        std::shuffle(ref_ids.begin(), ref_ids.end(), rng);
        auto refptr = ref_ids.data();

        prebuilts.emplace_back(singlepp::train_single_intersect<double, int>(ngenes, testptr, refmat, refptr, labptr, markers[r], bopt));
        integrated_inputs.push_back(singlepp::prepare_integrated_input_intersect<int>(ngenes, testptr, refmat, refptr, labptr, prebuilts.back()));

        // Doing a reference calculation by reordering the matrix and then calling the non-intersection methods.
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

        references_reorg.emplace_back(tatami::make_DelayedSubset(references[r], std::move(remapping), true));
        prebuilts_reorg.push_back(singlepp::train_single(*(references_reorg.back()), labptr, std::move(mcopy), bopt));
        integrated_inputs_reorg.push_back(singlepp::prepare_integrated_input(*(references_reorg.back()), labptr, prebuilts_reorg.back()));

        // Also shuffling the intersection to check that the input order doesn't affect the results.
        intersections_shuffled.push_back(singlepp::intersect_genes<int>(ngenes, testptr, refmat.nrow(), refptr));
        std::shuffle(intersections_shuffled.back().begin(), intersections_shuffled.back().end(), rng);
        integrated_inputs_shuffled.push_back(singlepp::prepare_integrated_input_intersect<int>(ngenes, intersections_shuffled.back(), refmat, labptr, prebuilts.back()));
    }

    singlepp::TrainIntegratedOptions iopt;
    auto integrated = singlepp::train_integrated(std::move(integrated_inputs), iopt);
    auto integrated_reorg = singlepp::train_integrated(std::move(integrated_inputs_reorg), iopt);
    auto integrated_shuffled = singlepp::train_integrated(std::move(integrated_inputs_shuffled), iopt);

    // Mocking up some of the best choices.
    auto chosen = mock_best_choices(ntest, prebuilts, base_seed + 2468);
    std::vector<const int*> chosen_ptrs(nrefs);
    for (size_t r = 0; r < nrefs; ++r) {
        chosen_ptrs[r] = chosen[r].data();
    }

    // Comparing classification results (with fine-tuning, for some test coverage).
    singlepp::ClassifyIntegratedOptions<double> copt;
    copt.quantile = quantile;
    auto output = singlepp::classify_integrated<int>(*test, chosen_ptrs, integrated, copt);

    auto reorganized_output = singlepp::classify_integrated<int>(*test, chosen_ptrs, integrated_reorg, copt);
    EXPECT_EQ(output.best, reorganized_output.best);
    EXPECT_EQ(output.scores, reorganized_output.scores);
    EXPECT_EQ(output.delta, reorganized_output.delta);

    auto shuffled_output = singlepp::classify_integrated<int>(*test, chosen_ptrs, integrated_shuffled, copt);
    EXPECT_EQ(output.best, shuffled_output.best);
    EXPECT_EQ(output.scores, shuffled_output.scores);
    EXPECT_EQ(output.delta, shuffled_output.delta);
}

INSTANTIATE_TEST_SUITE_P(
    ClassifyIntegrated,
    ClassifyIntegratedTest,
    ::testing::Combine(
        ::testing::Values(5, 10, 20), // number of top genes.
        ::testing::Values(0.5, 0.8, 0.9) // number of quantiles.
    )
);

/********************************************/

class ClassifyIntegratedMismatchTest : public ::testing::Test, public IntegratedTestCore {
protected:
    static void SetUpTestSuite() {
        assemble();
    }
};

TEST_F(ClassifyIntegratedMismatchTest, Basic) {
    singlepp::TrainSingleOptions bopt;

    std::vector<singlepp::TrainedSingle<int, double> > prebuilts;
    prebuilts.reserve(nrefs); // ensure that no reallocations happen that might invalidate pointers in the integrated_inputs.
    std::vector<singlepp::TrainIntegratedInput<double, int, int> > integrated_inputs;
    integrated_inputs.reserve(nrefs);

    for (size_t r = 0; r < nrefs; ++r) {
        const auto& refmat = *(references[r]);
        const auto labptr = labels[r].data();
        prebuilts.push_back(singlepp::train_single(refmat, labptr, markers[r], bopt));
        integrated_inputs.push_back(singlepp::prepare_integrated_input(refmat, labptr, prebuilts.back()));
    }

    singlepp::TrainIntegratedOptions iopt;
    auto integrated = singlepp::train_integrated(std::move(integrated_inputs), iopt);

    // Mocking up the test dataset and its choices.
    size_t ntest = 20;
    auto test = spawn_matrix(ngenes * 2, ntest, 69); // more genes than expected.

    int base_seed = 70;
    auto chosen = mock_best_choices(ntest, prebuilts, /* seed = */ base_seed);
    std::vector<const int*> chosen_ptrs(nrefs);
    for (size_t r = 0; r < nrefs; ++r) {
        chosen_ptrs[r] = chosen[r].data();
    }

    // Verifying that it does, in fact, fail.
    singlepp::ClassifyIntegratedOptions<double> copt;
    bool failed = false;
    try {
        singlepp::classify_integrated(*test, chosen_ptrs, integrated, copt);
    } catch (std::exception& e) {
        EXPECT_TRUE(std::string(e.what()).find("number of rows") != std::string::npos);
        failed = true;
    }
    EXPECT_TRUE(failed);
}
