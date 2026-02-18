#include <gtest/gtest.h>

#include "singlepp/classify_integrated.hpp"
#include "singlepp/classify_single.hpp"

#include "spawn_matrix.h"
#include "mock_markers.h"
#include "naive_method.h"
#include "compare.h"

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

        for (std::size_t r = 0; r < nrefs; ++r) {
            unsigned long long seed = r * 1000u;
            std::size_t nlabels = 3 + r;
            references.push_back(spawn_matrix(ngenes, nsamples, /* seed = */ seed));
            labels.push_back(spawn_labels(nsamples, nlabels, /* seed = */ seed * 2));
            markers.push_back(mock_markers<int>(nlabels, 50, ngenes, /* seed = */ seed * 3));
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

template<class Prebuilt_>
static std::vector<std::vector<int> > mock_best_choices(std::size_t ntest, const std::vector<Prebuilt_>& prebuilts, unsigned long long seed) {
    const auto nrefs = prebuilts.size();
    std::vector<std::vector<int> > chosen(nrefs);

    std::mt19937_64 rng(seed);
    for (std::size_t r = 0; r < nrefs; ++r) {
        const auto nlabels = prebuilts[r].markers().size();
        for (std::size_t t = 0; t < ntest; ++t) {
            chosen[r].push_back(rng() % nlabels);
        }
    }

    return chosen;
}

/********************************************/

class ClassifyIntegratedTest : public ::testing::TestWithParam<std::tuple<int, double> >, public IntegratedTestCore {
protected:
    static void SetUpTestSuite() {
        assemble();
        test = spawn_matrix(ngenes, ntest, /* seed = */ 69);
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
    unsigned long long base_seed = ntop + quantile * 50u;

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
                const auto& best_markers = pre.markers()[chosen[r][t]];
                for (const auto& x : best_markers) {
                    for (auto y : x) {
                        tmp.insert(pre.subset()[y]);
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
    auto test_ids = simulate_test_ids(ngenes, /* seed = */ base_seed * 20);
    singlepp::TrainSingleOptions bopt;
    bopt.top = ntop;

    std::vector<std::vector<int> > ref_ids;
    ref_ids.reserve(nrefs);
    std::vector<singlepp::TrainedSingle<int, double> > prebuilts;
    prebuilts.reserve(nrefs); // ensure that no reallocations happen that might invalidate pointers in the integrated_inputs.
    std::vector<std::vector<int> > ref_subsets(nrefs);
    std::vector<singlepp::TrainIntegratedInput<double, int, int> > integrated_inputs;
    integrated_inputs.reserve(nrefs);

    for (size_t r = 0; r < nrefs; ++r) {
        size_t seed = base_seed * 20 + r * 321;
        ref_ids.push_back(simulate_ref_ids(ngenes, seed + 3));
        const auto refptr = ref_ids.back().data();

        const auto testptr = test_ids.data();
        const auto labptr = labels[r].data();
        const auto& refmat = *(references[r]);
        prebuilts.push_back(singlepp::train_single<double, int>(ngenes, testptr, refmat, refptr, labptr, markers[r], &(ref_subsets[r]), bopt));
        integrated_inputs.push_back(singlepp::prepare_integrated_input<int>(ngenes, testptr, refmat, refptr, labptr, prebuilts.back()));
    }

    singlepp::TrainIntegratedOptions iopt;
    auto integrated = singlepp::train_integrated(std::move(integrated_inputs), iopt);

    // Mocking up some of the best choices.
    auto chosen = mock_best_choices(ntest, prebuilts, /* seed = */ base_seed + 2468);
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
                const auto& ref_subset = ref_subsets[r];
                const auto& best_markers = pre.markers()[chosen[r][t]];

                for (const auto& x : best_markers) {
                    for (auto y : x) {
                        const auto yref = ref_subset[y];
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
    unsigned long long base_seed = ntop + quantile * 50;

    // Creating the integrated set of references with intersection, along with
    // a comparison to the simple method where we do the reorganization externally.
    // The aim is to check that all the various gene indexing steps are done correctly.
    std::vector<int> test_ids(ngenes);
    std::iota(test_ids.begin(), test_ids.end(), 0);

    singlepp::TrainSingleOptions bopt;
    bopt.top = ntop;
    std::vector<singlepp::TrainedSingle<int, double> > prebuilts;
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

        prebuilts.emplace_back(singlepp::train_single<double, int>(ngenes, testptr, refmat, refptr, labptr, markers[r], NULL, bopt));
        integrated_inputs.push_back(singlepp::prepare_integrated_input<int>(ngenes, testptr, refmat, refptr, labptr, prebuilts.back()));

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
        integrated_inputs_shuffled.push_back(singlepp::prepare_integrated_input<int>(ngenes, intersections_shuffled.back(), refmat, labptr, prebuilts.back()));
    }

    singlepp::TrainIntegratedOptions iopt;
    auto integrated = singlepp::train_integrated(std::move(integrated_inputs), iopt);
    auto integrated_reorg = singlepp::train_integrated(std::move(integrated_inputs_reorg), iopt);
    auto integrated_shuffled = singlepp::train_integrated(std::move(integrated_inputs_shuffled), iopt);

    // Mocking up some of the best choices.
    auto chosen = mock_best_choices(ntest, prebuilts, /* seed = */ base_seed + 1379);
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

class ClassifyIntegratedSparseTest : public ::testing::TestWithParam<std::tuple<int, double> > {
protected:
    inline static std::vector<std::shared_ptr<tatami::Matrix<double, int> > > dense_references, sparse_references;
    inline static std::vector<std::vector<int> > labels;
    inline static std::vector<singlepp::Markers<int> > markers;

    inline static size_t ngenes = 1234;
    inline static size_t nsamples = 40;
    inline static size_t nrefs = 4;

    inline static size_t ntest = 20;
    inline static std::shared_ptr<tatami::Matrix<double, int> > dense_test, sparse_test; 

protected:
    static void SetUpTestSuite() {
        for (std::size_t r = 0; r < nrefs; ++r) {
            unsigned long long seed = r * 123u;
            std::size_t nlabels = 3 + r;

            dense_references.push_back(spawn_sparse_matrix(ngenes, nsamples, /* seed = */ seed, /* density = */ 0.3));
            sparse_references.push_back(tatami::convert_to_compressed_sparse<double, int>(*(dense_references.back()), true, {}));

            labels.push_back(spawn_labels(nsamples, nlabels, /* seed = */ seed * 2));
            markers.push_back(mock_markers<int>(nlabels, 50, ngenes, /* seed = */ seed * 3));
        }

        dense_test = spawn_sparse_matrix(ngenes, ntest, /* seed = */ 6969, /* density = */ 0.3);
        sparse_test = tatami::convert_to_compressed_sparse<double, int>(*dense_test, true, {});
    }

    void check_almost_equal_sparse_results(
        const singlepp::ClassifyIntegratedResults<int, double>& expected,
        const singlepp::ClassifyIntegratedResults<int, double>& results
    ) const {
        ASSERT_EQ(expected.best.size(), ntest);
        ASSERT_EQ(results.best.size(), ntest);
        ASSERT_EQ(expected.delta.size(), ntest);
        ASSERT_EQ(results.delta.size(), ntest);
        for (std::size_t t = 0; t < ntest; ++t) {
            check_almost_equal_assignment(expected.best[t], expected.delta[t], results.best[t], results.delta[t]);
        }

        ASSERT_EQ(expected.scores.size(), nrefs);
        ASSERT_EQ(results.scores.size(), nrefs);
        for (std::size_t l = 0; l < nrefs; ++l) {
            ASSERT_EQ(expected.scores[l].size(), ntest);
            ASSERT_EQ(results.scores[l].size(), ntest);
            check_almost_equal_vectors(expected.scores[l], results.scores[l]);
        }
    }
};

TEST_P(ClassifyIntegratedSparseTest, Basic) {
    auto param = GetParam();
    int ntop = std::get<0>(param);
    double quantile = std::get<1>(param);
    unsigned long long base_seed = ntop + quantile * 50;

    // Creating the integrated set of references.
    singlepp::TrainSingleOptions bopt;
    bopt.top = ntop;

    std::vector<singlepp::TrainedSingle<int, double> > prebuilts;
    prebuilts.reserve(nrefs); // ensure that no reallocations happen that might invalidate pointers in the integrated_inputs.
    std::vector<singlepp::TrainIntegratedInput<double, int, int> > dense_integrated_inputs, sparse_integrated_inputs;
    dense_integrated_inputs.reserve(nrefs);
    sparse_integrated_inputs.reserve(nrefs);

    // Sparse-dense and dense-dense compute the exact same L2, so we can do this comparison without fear of discrepancies due to numerical differences.
    for (size_t r = 0; r < nrefs; ++r) {
        const auto labptr = labels[r].data();
        const auto& refmat = *(dense_references[r]);
        prebuilts.push_back(singlepp::train_single(refmat, labptr, markers[r], bopt));
        dense_integrated_inputs.push_back(singlepp::prepare_integrated_input(refmat, labptr, prebuilts.back()));
        const auto& spmat = *(sparse_references[r]);
        sparse_integrated_inputs.push_back(singlepp::prepare_integrated_input(spmat, labptr, prebuilts.back()));
    }

    singlepp::TrainIntegratedOptions iopt;
    auto dense_integrated = singlepp::train_integrated(dense_integrated_inputs, iopt);
    auto sparse_integrated = singlepp::train_integrated(sparse_integrated_inputs, iopt);

    // Mocking up some of the best choices.
    auto chosen = mock_best_choices(ntest, prebuilts, /* seed = */ base_seed);
    std::vector<const int*> chosen_ptrs(nrefs);
    for (size_t r = 0; r < nrefs; ++r) {
        chosen_ptrs[r] = chosen[r].data();
    }

    singlepp::ClassifyIntegratedOptions<double> copt;
    copt.quantile = quantile;
    auto expected = singlepp::classify_integrated<int>(*dense_test, chosen_ptrs, dense_integrated, copt);

    auto sparse_to_dense = singlepp::classify_integrated<int>(*sparse_test, chosen_ptrs, dense_integrated, copt);
    check_almost_equal_sparse_results(expected, sparse_to_dense);

    auto dense_to_sparse = singlepp::classify_integrated<int>(*dense_test, chosen_ptrs, sparse_integrated, copt);
    check_almost_equal_sparse_results(expected, dense_to_sparse);

    auto sparse_to_sparse = singlepp::classify_integrated<int>(*sparse_test, chosen_ptrs, sparse_integrated, copt);
    check_almost_equal_sparse_results(expected, sparse_to_sparse);
}

TEST_P(ClassifyIntegratedSparseTest, Intersect) {
    auto param = GetParam();
    int ntop = std::get<0>(param);
    double quantile = std::get<1>(param);
    unsigned long long base_seed = ntop + quantile * 50;

    // Creating the integrated set of references.
    singlepp::TrainSingleOptions bopt;
    bopt.top = ntop;

    std::vector<singlepp::Intersection<int> > intersections;
    intersections.reserve(nrefs);
    for (size_t r = 0; r < nrefs; ++r) {
        intersections.push_back(mock_intersection<int>(ngenes, ngenes, ngenes * 0.75, /* seed = */ base_seed + r));
    }

    std::vector<singlepp::TrainedSingle<int, double> > prebuilts;
    prebuilts.reserve(nrefs); // ensure that no reallocations happen that might invalidate pointers in the integrated_inputs.
    std::vector<singlepp::TrainIntegratedInput<double, int, int> > dense_integrated_inputs, sparse_integrated_inputs;
    dense_integrated_inputs.reserve(nrefs);
    sparse_integrated_inputs.reserve(nrefs);

    // Sparse-dense and dense-dense compute the exact same L2, so we can do this comparison without fear of discrepancies due to numerical differences.
    for (size_t r = 0; r < nrefs; ++r) {
        const auto labptr = labels[r].data();
        const auto& refmat = *(dense_references[r]);
        prebuilts.push_back(singlepp::train_single<double, int>(ngenes, intersections.back(), refmat, labptr, markers[r], NULL, bopt));
        dense_integrated_inputs.push_back(singlepp::prepare_integrated_input<int>(ngenes, intersections.back(), refmat, labptr, prebuilts.back()));
        const auto& spmat = *(sparse_references[r]);
        sparse_integrated_inputs.push_back(singlepp::prepare_integrated_input<int>(ngenes, intersections.back(), spmat, labptr, prebuilts.back()));
    }

    singlepp::TrainIntegratedOptions iopt;
    auto dense_integrated = singlepp::train_integrated(dense_integrated_inputs, iopt);
    auto sparse_integrated = singlepp::train_integrated(sparse_integrated_inputs, iopt);

    // Mocking up some of the best choices.
    auto chosen = mock_best_choices(ntest, prebuilts, /* seed = */ base_seed);
    std::vector<const int*> chosen_ptrs(nrefs);
    for (size_t r = 0; r < nrefs; ++r) {
        chosen_ptrs[r] = chosen[r].data();
    }

    singlepp::ClassifyIntegratedOptions<double> copt;
    copt.quantile = quantile;
    auto expected = singlepp::classify_integrated<int>(*dense_test, chosen_ptrs, dense_integrated, copt);

    auto sparse_to_dense = singlepp::classify_integrated<int>(*sparse_test, chosen_ptrs, dense_integrated, copt);
    check_almost_equal_sparse_results(expected, sparse_to_dense);

    auto dense_to_sparse = singlepp::classify_integrated<int>(*dense_test, chosen_ptrs, sparse_integrated, copt);
    check_almost_equal_sparse_results(expected, dense_to_sparse);

    auto sparse_to_sparse = singlepp::classify_integrated<int>(*sparse_test, chosen_ptrs, sparse_integrated, copt);
    check_almost_equal_sparse_results(expected, sparse_to_sparse);
}

INSTANTIATE_TEST_SUITE_P(
    ClassifyIntegrated,
    ClassifyIntegratedSparseTest,
    ::testing::Combine(
        ::testing::Values(5, 10, 20), // number of top genes.
        ::testing::Values(0.5, 0.8, 0.9) // number of quantiles.
    )
);

/********************************************/

class ClassifyIntegratedOtherTest : public ::testing::Test, public IntegratedTestCore {
protected:
    static void SetUpTestSuite() {
        assemble();
    }
};

TEST_F(ClassifyIntegratedOtherTest, Mismatch) {
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
    auto test = spawn_matrix(ngenes * 2, ntest, /* seed = */ 69); // more genes than expected.

    auto chosen = mock_best_choices(ntest, prebuilts, /* seed = */ 70);
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

TEST_F(ClassifyIntegratedOtherTest, FineTuneEdgeCase) {
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

    singlepp::AnnotateIntegrated<false, int, double, double> ft(integrated);
    singlepp::RankedVector<double, int> query_ranked;

    // We need at least 3 references to have any kind of fine-tuning.
    ASSERT_GE(nrefs, 3);

    // Checking that we abort early in various circumstances.
    std::vector<int> dummy_assigned(1);
    std::vector<const int*> assigned_ptrs(nrefs, dummy_assigned.data());

    std::vector<double> scores(nrefs, 0.5);
    scores[1] = 0.7;
    std::vector<int> reflabels_in_use;
    auto out = ft.run_fine(0, query_ranked, integrated, assigned_ptrs, 0.8, 0.05, scores, reflabels_in_use);
    EXPECT_EQ(out.first, 1);
    EXPECT_FLOAT_EQ(out.second, 0.2);

    scores[1] = 0.5;
    scores[nrefs - 1] = 0.51;
    out = ft.run_fine(0, query_ranked, integrated, assigned_ptrs, 0.8, 0.05, scores, reflabels_in_use);
    EXPECT_EQ(out.first, nrefs - 1);
    EXPECT_FLOAT_EQ(out.second, 0.01);
}

TEST_F(ClassifyIntegratedOtherTest, FineTuneExactRecovery) {
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

    singlepp::AnnotateIntegrated<false, int, double, double> ft(integrated);
    singlepp::RankedVector<double, int> query_ranked;

    // We need at least 3 references to have any kind of fine-tuning.
    ASSERT_GE(nrefs, 3);

    std::vector<double> scores(nrefs);
    std::vector<int> reflabels_in_use;

    for (size_t r = 0; r < nrefs; ++r) {
        const auto& refmat = *(references[r]);
        const int NC = refmat.ncol();
        const int NR = refmat.nrow();
        auto ext = refmat.dense_column();
        std::vector<double> buffer(NR);

        std::vector<int> dummy_assigned(NC);
        std::vector<const int*> assigned_ptrs(nrefs);
        for (size_t r2 = 0; r2 < nrefs; ++r2) {
            if (r2 == r) {
                assigned_ptrs[r2] = labels[r2].data();
            } else {
                assigned_ptrs[r2] = dummy_assigned.data();
            }
        }

        for (int c = 0; c < NC; ++c) {
            auto ptr = ext->fetch(c, buffer.data());
            query_ranked.clear();
            int counter = 0;
            for (auto r : integrated.subset()) {
                query_ranked.emplace_back(ptr[r], counter);
                ++counter;
            }
            std::sort(query_ranked.begin(), query_ranked.end());

            // Mock up the scores to have two labels, to force it to go through one fine-tuning iteration. 
            scores.clear();
            scores.resize(nrefs);
            scores[r] = 0.5;
            scores[(r + 1) % nrefs] = 0.51;

            // Use a quantile of 1 so that an exact match is respected with the maximum correlation.
            auto out = ft.run_fine(c, query_ranked, integrated, assigned_ptrs, 1, 0.05, scores, reflabels_in_use);
            EXPECT_EQ(out.first, r);
        }
    }
}

/********************************************/

TEST(ClassifyIntegrated, FineTuneSparse) {
    size_t ngenes = 1000;
    size_t nsamples = 50;
    size_t nrefs = 4; // Needs at least 3 references for fine-tuning to actually happen.

    std::vector<std::vector<int> > labels;
    std::vector<singlepp::Markers<int> > markers;
    labels.reserve(nrefs);
    markers.reserve(nrefs);

    for (std::size_t r = 0; r < nrefs; ++r) {
        unsigned long long seed = r * 1055u;
        std::size_t nlabels = 3 + r;
        labels.push_back(spawn_labels(nsamples, nlabels, /* seed = */ seed * 10));
        markers.push_back(mock_markers<int>(nlabels, 50, ngenes, /* seed = */ seed * 20));
    }

    std::vector<std::shared_ptr<tatami::Matrix<double, int> > > dense_references, sparse_references;
    dense_references.reserve(nrefs);
    sparse_references.reserve(nrefs);

    // Creating the integrated set of references.
    singlepp::TrainSingleOptions bopt;
    std::vector<singlepp::TrainedSingle<int, double> > prebuilts;
    prebuilts.reserve(nrefs); // ensure that no reallocations happen that might invalidate pointers in the integrated_inputs.
    std::vector<singlepp::TrainIntegratedInput<double, int, int> > dense_integrated_inputs, sparse_integrated_inputs;
    dense_integrated_inputs.reserve(nrefs);
    sparse_integrated_inputs.reserve(nrefs);

    for (std::size_t r = 0; r < nrefs; ++r) {
        std::size_t seed = r * 456;
        dense_references.push_back(spawn_sparse_matrix(ngenes, nsamples, /* seed = */ seed, /* density = */ 0.3));
        sparse_references.push_back(tatami::convert_to_compressed_sparse<double, int>(*(dense_references.back()), true, {}));

        const auto labptr = labels[r].data();
        const auto& refmat = *(dense_references[r]);
        prebuilts.push_back(singlepp::train_single(refmat, labptr, markers[r], bopt));
        dense_integrated_inputs.push_back(singlepp::prepare_integrated_input(refmat, labptr, prebuilts.back()));

        const auto& spmat = *(sparse_references[r]);
        sparse_integrated_inputs.push_back(singlepp::prepare_integrated_input(spmat, labptr, prebuilts.back()));
    }

    singlepp::TrainIntegratedOptions iopt;
    auto dense_integrated = singlepp::train_integrated(dense_integrated_inputs, iopt);
    auto sparse_integrated = singlepp::train_integrated(sparse_integrated_inputs, iopt);

    const int ntest = 100; 
    auto new_test = spawn_sparse_matrix(ngenes, ntest, /* seed = */ 302, /* density = */ 0.2);

    // Mocking up some of the best choices.
    auto chosen = mock_best_choices(ntest, prebuilts, /* seed = */ 88);
    std::vector<const int*> chosen_ptrs(nrefs);
    for (size_t r = 0; r < nrefs; ++r) {
        chosen_ptrs[r] = chosen[r].data();
    }

    singlepp::AnnotateIntegrated<false, int, double, double> dense_ft(dense_integrated);
    singlepp::AnnotateIntegrated<true, int, double, double> dense_ft2(dense_integrated);
    singlepp::AnnotateIntegrated<false, int, double, double> sparse_ft(sparse_integrated);
    singlepp::AnnotateIntegrated<true, int, double, double> sparse_ft2(sparse_integrated);

    const auto nmarkers = dense_integrated.subset().size();
    auto wrk = new_test->dense_column(dense_integrated.subset());
    std::vector<double> buffer(nmarkers);
    std::vector<int> refs_in_use;

    for (int t = 0; t < ntest; ++t) {
        auto vec = wrk->fetch(t, buffer.data()); 
        auto ranked = fill_ranks<int>(nmarkers, vec);

        std::vector<double> scores(nrefs, 0.5);
        const auto empty = t % nrefs;
        scores[empty] = 0; // forcing one of the labels to be zero so that it actually does the fine-tuning.

        auto score_copy = scores;
        auto expected = dense_ft.run_fine(t, ranked, dense_integrated, chosen_ptrs, 0.8, 0.05, score_copy, refs_in_use);
        EXPECT_NE(expected.first, empty);

        // Due to differences in numerical precision between dense/sparse calculations, comparisons may not be exact.
        // This results in different 'best' labels in the presence of near-ties, so if there's a mismatch,
        // we check that the delta is indeed near-zero, i.e., there is a near-tie. 
        score_copy = scores;
        auto dense_to_sparse = sparse_ft.run_fine(t, ranked, sparse_integrated, chosen_ptrs, 0.8, 0.05, score_copy, refs_in_use);
        check_almost_equal_assignment(expected.first, expected.second, dense_to_sparse.first, dense_to_sparse.second);

        singlepp::RankedVector<double, int> sparse_ranked;
        for (auto r : ranked) {
            if (r.first) {
                sparse_ranked.push_back(r);
            }
        }

        score_copy = scores;
        auto sparse_to_dense = dense_ft2.run_fine(t, sparse_ranked, dense_integrated, chosen_ptrs, 0.8, 0.05, score_copy, refs_in_use);
        check_almost_equal_assignment(expected.first, expected.second, sparse_to_dense.first, sparse_to_dense.second);

        score_copy = scores;
        auto sparse_to_sparse = sparse_ft2.run_fine(t, sparse_ranked, sparse_integrated, chosen_ptrs, 0.8, 0.05, score_copy, refs_in_use);
        check_almost_equal_assignment(expected.first, expected.second, sparse_to_sparse.first, sparse_to_sparse.second);
    }
}
