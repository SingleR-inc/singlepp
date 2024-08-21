#ifndef SINGLEPP_CLASSIFY_INTEGRATED_HPP
#define SINGLEPP_CLASSIFY_INTEGRATED_HPP

#include "macros.hpp"

#include "tatami/tatami.hpp"

#include "compute_scores.hpp"
#include "scaled_ranks.hpp"
#include "train_integrated.hpp"

#include <vector>
#include <unordered_map>
#include <unordered_set>

/**
 * @file IntegratedScorer.hpp
 *
 * @brief Integrate classifications from multiple references.
 */

namespace singlepp {

/**
 * @brief Options for `classify_integrated()`.
 */
template<typename Float_>
struct ClassifyIntegratedOptions {
    /**
     * Quantile to use to compute a per-label score from the correlations.
     * This has the same interpretation as `ClassifySingleOptions::quantile`.
     */
    Float_ quantile = 0.8;

    /**
     * Number of threads to use.
     */
    int num_threads = 1;
};

/**
 * @brief Output buffers for `classify_single()`.
 * @tparam Label_ Integer type for the reference labels.
 * @tparam Float_ Floating-point type for the correlations and scores.
 */
template<typename RefLabel_, typename Float_>
struct ClassifySingleBuffers {
    /** 
     * Pointer to an array of length equal to the number of test cells.
     * On output, this is filled with the index of the assigned label for each cell.
     */
    RefLabel_* best;

    /** 
     * Vector of length equal to the number of labels.
     * Each entry contains a pointer to an array of length equal to the number of test cells.
     * On output, this is filled with the (non-fine-tuned) score for each label for each cell.
     * Any pointer may be `NULL` in which case the scores for that label will not be reported.
     */
    std::vector<Float_*> scores;

    /**
     * Pointer to an array of length equal to the number of test cells.
     * On output, this is filled with the difference between the highest and second-highest scores, possibly after fine-tuning.
     * This may also be `NULL` in which case the deltas are not reported.
     */
    Float_* delta;
};

/**
 * @brief Integrate classifications from multiple references.
 *
 * In situations where multiple reference datasets are available,
 * we would like to obtain a single prediction for each cell from all of those references.
 * This is somewhat tricky as the different references are likely to contain strong batch effects,
 * complicating the calculation of marker genes between labels from different references (and thus precluding direct use of the usual `Classifier::run()`).
 * The labels themselves also tend to be inconsistent, e.g., different vocabularies and resolutions, making it difficult to define sensible groups in a combined "super-reference".
 *
 * To avoid these issues, we first perform classification within each reference individually.
 * For each test cell, we identify its predicted label from a given reference, and we collect all the marker genes for that label (across all pairwise comparisons in that reference).
 * After doing this for each reference, we pool all of the collected markers to obtain a common set of interesting genes.
 * We then compute the correlation-based score between the test cell's expression profile and its predicted label from each reference, using that common set of genes.
 * The label with the highest score is considered the best representative across all references.
 *
 * This strategy is similar to using `Classifier::run()` without fine-tuning, 
 * except that we are choosing between the best labels from all references rather than between all labels from one reference.
 * The main idea is to create a common feature set so that the correlations can be reasonably compared across references.
 * Note that differences in the feature sets across references are tolerated by simply ignoring missing genes when computing the correlations.
 * This reduces the comparability of the scores as the effective feature set will vary a little (or a lot, depending) across references;
 * nonetheless, it is preferred to taking the intersection, which is liable to leave us with very few genes.
 *
 * Our approach avoids any direct comparison between the expression profiles of different references,
 * allowing us to side-step the question of how to deal with the batch effects.
 * Similarly, we defer responsibility on solving the issue of label heterogeneity,
 * by just passing along the existing labels and leaving it to the user's interpretation.
 * 
 * @param mat Expression matrix of the test dataset, where rows are genes and columns are cells.
 * The identity of the rows should be consistent with the arguments used in `IntegratedBuilder::add()`.
 * @param[in] assigned Vector of pointers of length equal to the number of references.
 * Each pointer should point to an array of length equal to the number of columns in `mat`,
 * containing the assigned label for each column in each reference.
 * @param built Set of integrated references produced by `IntegratedBuilder::finish()`.
 * @param[out] best Pointer to an array of length equal to the number of columns in `mat`.
 * On output, this is filled with the index of the reference with the best label for each cell.
 * @param[out] scores Vector of pointers of length equal to the number of references.
 * Each pointer should point to an array of length equal to the number of columns in `mat`.
 * On output, this is filled with the (non-fine-tuned) score for the best label of that reference for each cell.
 * Any pointer may be `NULL` in which case the scores for that label will not be reported.
 * @param[out] delta Pointer to an array of length equal to the number of columns in `mat`.
 * On output, this is filled with the difference between the highest and second-highest scores.
 * This may also be `NULL` in which case the deltas are not reported.
 */
template<typename Value_, typename Index_, typename Label_, typename RefLabel_, typename Float_>
void classify_integrated(
    const tatami::Matrix<Value_, Index_>& test,
    const std::vector<const Label_*>& assigned,
    const TrainedIntegrated<Index_>& trained,
    ClassifyIntegratedBuffers<RefLabel_, Float_>& buffers,
    const ClassifyIntegratedOptions<Float_>& options)
{
    auto NR = test.nrow();
    auto nref = trained.num_references();

    tatami::parallelize([&](size_t, Index_ start, Index_ len) -> void {
        // We perform an indexed extraction, so all subsequent indices
        // will refer to indices into this subset (i.e., 'trained.universe').
        tatami::VectorPtr<Index_> universe_ptr(tatami::VectorPtr<Index_>{}, &(trained.universe));
        auto wrk = tatami::consecutive_extractor<false>(&test, false, start, len, std::move(universe_ptr));
        std::vector<Value_> buffer(trained.universe.size());

        std::unordered_set<Index_> miniverse_tmp;
        std::vector<Index_> miniverse;
        internal::RankRemapper<Index_> intersect_mapping, direct_mapping;

        RankedVector<Value_, Index_> test_ranked_all, test_ranked;
        test_ranked_all.reserve(NR);
        test_ranked.reserve(NR);
        RankedVector<Index_, Index_> ref_ranked;
        ref_ranked.reserve(NR);

        std::vector<Value_> test_scaled(NR);
        std::vector<Value_> ref_scaled(NR);
        std::vector<Value_> all_correlations;

        for (Index_ i = start, end = start + len; i < end; ++i) {
            // Pulling out the markers that we'll be using for this cell.
            miniverse_tmp.clear();
            for (size_t r = 0; r < nref; ++r) {
                auto best = assigned[r][i];
                const auto& markers = trained.markers[r][best];
                miniverse_tmp.insert(markers.begin(), markers.end());
            }

            miniverse.clear();
            miniverse.insert(miniverse.end(), miniverse_tmp.begin(), miniverse_tmp.end());
            std::sort(miniverse.begin(), miniverse.end());

            test_ranked_all.clear();
            auto ptr = wrk->fetch(buffer.data());
            for (auto u : miniverse) {
                test_ranked_all.emplace_back(ptr[u], u);
            }
            std::sort(test_ranked_all.begin(), test_ranked_all.end());

            // Scanning through each reference and computing the score for the best group.
            Value_ best_score = -1000, next_best = -1000;
            Index_ best_ref = 0;
            bool direct_mapping_filled = false;

            for (size_t r = 0; r < nref; ++r) {
                // Further subsetting to the intersection of markers that are
                // present in each reference.
                const internal::RankRemapper<Index_>* mapping;
                if (trained.check_availability[r]) {
                    const auto& cur_available = trained.available[r];
                    intersect_mapping.clear();
                    intersect_mapping.reserve(miniverse.size());
                    for (auto c : miniverse) {
                        if (cur_available.find(c) != cur_available.end()) {
                            intersect_mapping.add(c);
                        }
                    }
                    mapping = &intersect_mapping;

                } else {
                    if (!direct_mapping_filled) {
                        direct_mapping.clear();
                        direct_mapping.reserve(miniverse.size());
                        for (auto c : miniverse) {
                            direct_mapping.add(c);
                        }
                        direct_mapping_filled = true;
                    }
                    mapping = &direct_mapping;
                } 

                mapping->remap(test_ranked_all, test_ranked);
                test_scaled.resize(test_ranked.size());
                internal::scaled_ranks(test_ranked, test_scaled.data());

                // Now actually calculating the score for the best group for
                // this cell in this reference. This assumes that 'ref.ranked'
                // already contains sorted pairs where the indices refer to the
                // rows of the original data matrix.
                auto best = assigned[r][i];
                const auto& best_ranked = built.ranked[r][best];
                all_correlations.clear();
                ref_scaled.resize(test_scaled.size());

                for (size_t s = 0; s < best_ranked.size(); ++s) {
                    ref_ranked.clear();
                    mapping->remap(best_ranked[s], ref_ranked);
                    internal::scaled_ranks(ref_ranked, ref_scaled.data());
                    Value_ cor = internal::distance_to_correlation(ref_scaled.size(), test_scaled, ref_scaled);
                    all_correlations.push_back(cor);
                }

                Value_ score = correlations_to_scores(all_correlations, options.quantile);
                if (buffers.scores[r]) {
                    buffers.scores[r][i] = score;
                }
                if (score > best_score) {
                    next_best = best_score;
                    best_score = score;
                    best_ref = r;
                } else if (score > next_best) {
                    next_best = score;
                }
            }

            if (buffers.best) {
                buffers.best[i] = best_ref;
            }
            if (buffers.delta) {
                if (nref > 1) {
                    buffers.delta[i] = best_score - next_best;
                } else {
                    buffers.delta[i] = std::numeric_limits<Float_>::quiet_NaN();
                }
            }
        }

    }, test.ncol(), options.num_threads);
}

/**
 * @brief Results of `classify_integrated()`.
 */
template<typename RefLabel_, typename Float_>
struct ClassifyIntegratedResults {
    /**
     * @cond
     */
    Results(size_t ncells, size_t nrefs) : best(ncells), scores(nrefs, std::vector<double>(ncells)), delta(ncells) {}
    /**
     * @endcond
     */

    /** 
     * Vector of length equal to the number of cells in the test dataset,
     * containing the index of the reference with the top-scoring label for each cell.
     */
    std::vector<RefLabel_> best;

    /**
     * Vector of length equal to the number of references,
     * containing vectors of length equal to the number of cells in the test dataset.
     * Each vector corresponds to a reference and contains the score for the best label in that reference for each cell.
     */
    std::vector<std::vector<Float_> > scores;

    /** 
     * Vector of length equal to the number of cells in the test dataset.
     * This contains the difference between the highest and second-highest scores for each cell.
     */
    std::vector<Float_> delta;
};

/**
 * @param mat Expression matrix of the test dataset, where rows are genes and columns are cells.
 * The identity of the rows should be consistent with the arguments used in `IntegratedBuilder::add()`.
 * @param[in] assigned Vector of pointers of length equal to the number of references.
 * Each pointer should point to an array of length equal to the number of columns in `mat`,
 * containing the assigned label for each column in each reference.
 * @param built Set of integrated references produced by `IntegratedBuilder::finish()`.
 *
 * @return A `Results` object containing the assigned labels and scores.
 */
template<typename RefLabel_, typename Value_, typename Index_, typename Label_, typename Float_>
ClassifyIntegratedResults<RefLabel_, Float_> classify_integrated(
    const tatami::Matrix<Value_, Index_>& test,
    const std::vector<const Label_*>& assigned,
    const TrainedIntegrated<Index_>& trained,
    const ClassifyIntegratedOptions<Float_>& options)
{
    ClassifySingleResults<RefLabel_, Float_> results(test.ncol(), trained.num_references());
    ClassifySingleBuffers<RefLabel_, Float_> buffers;
    buffers.best = results.best.data();
    buffers.delta = results.delta.data();
    buffers.scores.reserve(results.scores.size());
    for (auto& s : results.scores) {
        buffers.scores.emplace_back(s.data());
    }
    classify_integrated(test, assigned, trained, buffers);
    return output;
}

}

#endif
