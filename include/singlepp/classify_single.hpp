#ifndef SINGLEPP_CLASSIFY_SINGLE_HPP
#define SINGLEPP_CLASSIFY_SINGLE_HPP

#include "macros.hpp"

#include "tatami/tatami.hpp"

#include "annotate_cells.hpp"
#include "train_single.hpp"

#include <vector> 
#include <stdexcept>

/**
 * @file BasicScorer.hpp
 *
 * @brief Defines the `BasicScorer` class.
 */

namespace singlepp {

/**
 * @brief Options for `classify_single()`.
 * @tparam Float_ Floating-point type for the correlations and scores.
 */
template<typename Float_>
struct ClassifySingleOptions {
    /**
     * Quantile to use to define a per-label score for each test cell,
     * by applying it to the distribution of correlations between the test cell and that label's reference profiles.
     * Values closer to 0.5 focus on the behavior of the majority of a label's reference profiles.
     * Smaller values will be more sensitive to the presence of a subset of profiles that are more similar to the test cell,
     * which can be useful when the reference profiles themselves are heterogeneous.
     */
    Float_ quantile = 0.8;

    /**
     * Score threshold to use to select the top-scoring subset of labels during fine-tuning.
     * Larger values increase the chance of recovering the correct label at the cost of computational time.
     *
     * This threshold should not be set to to a value that is too large.
     * Otherwise, the first fine-tuning iteration would just contain all labels and there would be no reduction of the marker space.
     */
    Float_ fine_tune_threshold = 0.05;

    /**
     * Whether to perform fine-tuning.
     * This can be disabled to improve speed at the cost of accuracy.
     */
    bool fine_tune = true;

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
template<typename Label_, typename Float_>
struct ClassifySingleBuffers {
    /** 
     * Pointer to an array of length equal to the number of test cells.
     * On output, this is filled with the index of the assigned label for each cell.
     */
    Label_* best;

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
 * @param mat Expression matrix of the test dataset, where rows are genes and columns are cells.
 * This should have the same order and identity of genes as the reference matrix used to create `built`.
 * @param built An object produced by `BasicBuilder::build()`.
 * @param[out] best Pointer to an array of length equal to the number of columns in `mat`.
 * On output, this is filled with the index of the assigned label for each cell.
 * @param[out] scores Vector of pointers to arrays of length equal to the number of columns in `mat`.
 * On output, this is filled with the (non-fine-tuned) score for each label for each cell.
 * Any pointer may be `NULL` in which case the scores for that label will not be reported.
 * @param[out] delta Pointer to an array of length equal to the number of columns in `mat`.
 * On output, this is filled with the difference between the highest and second-highest scores, possibly after fine-tuning.
 * This may also be `NULL` in which case the deltas are not reported.
 */
template<typename Value_, typename Index_, typename Float_, typename Label_>
void classify_single(
    const tatami::Matrix<Value_, Index_>& test, 
    const TrainedSingle<Index_, Float_>& built,
    const ClassifySingleBuffers<Label_, Float_>& buffers,
    const ClassifySingleOptions<Float_>& options) 
{
    annotate_cells_simple(
        test, 
        built.get_subset().size(), 
        built.get_subset().data(), 
        built.get_references(), 
        built.get_markers(), 
        options.quantile, 
        options.fine_tune, 
        options.fine_tune_threshold, 
        buffers.best, 
        buffers.scores, 
        buffers.delta,
        options.num_threads
    );
}

/**
 * @brief Classify cells from a single pre-built reference dataset.
 *
 * This class uses the pre-built reference from `BasicBuilder` to classify each cell in a test dataset.
 * The algorithm and parameters are the same as described for the `Classifier` class;
 * in fact, `Classifier` just calls `BasicBuilder::run()` and then `BasicScorer::run()`.
 *
 * It is occasionally useful to call these two functions separately if the same reference dataset is to be used multiple times,
 * e.g., on different test datasets or with different parameters.
 * In such cases, we can save time by avoiding redundant builds;
 * we just have to call `BasicScorer::run()` in all subsequent uses of the pre-built reference.
 *
 * @param mat Expression matrix of the test dataset, where rows are genes and columns are cells.
 * This may have a different ordering of genes compared to the reference matrix used to create `built`,
 * provided that all genes corresponding to `built.subset` are present.
 * @param built An object produced by `BasicBuilder::build()`.
 * @param[in] mat_subset Pointer to an array of length equal to that of `built.subset`,
 * containing the index of the row of `mat` corresponding to each gene in `built.subset`.
 * That is, row `mat_subset[i]` in `mat` should be the same gene as row `built.subset[i]` in the reference matrix.
 */
template<typename Value_, typename Index_, typename Float_, typename Label_>
void classify_single(
    const tatami::Matrix<Value_, Index_>& test,
    const Index_* test_subset,
    const TrainedSingle<Index_, Float_>& built,
    const ClassifySingleBuffers<Label_, Float_>& buffers,
    const ClassifySingleOptions<Float_>& options) 
{
    annotate_cells_simple(
        test, 
        built.get_subset().size(), 
        test_subset, 
        built.get_references(), 
        built.get_markers(), 
        options.quantile, 
        options.fine_tune, 
        options.fine_tune_threshold, 
        buffers.best, 
        buffers.scores, 
        buffers.delta,
        options.num_threads
    );
}

/**
 * @param mat Expression matrix of the test dataset, where rows are genes and columns are cells.
 * @param built An object produced by `build()` with intersections.
 * @param[out] best Pointer to an array of length equal to the number of columns in `mat`.
 * On output, this is filled with the index of the assigned label for each cell.
 * @param[out] scores Vector of pointers to arrays of length equal to the number of columns in `mat`.
 * On output, this is filled with the (non-fine-tuned) score for each label for each cell.
 * Any pointer may be `NULL` in which case the scores for that label will not be reported.
 * @param[out] delta Pointer to an array of length equal to the number of columns in `mat`.
 * On output, tkkhis is filled with the difference between the highest and second-highest scores, possibly after fine-tuning.
 * This may also be `NULL` in which case the deltas are not reported.
 */
template<typename Value_, typename Index_, typename Float_, typename Label_>
void classify_single(
    const tatami::Matrix<Value_, Index_>* mat, 
    const TrainedSingleIntersect<Index_, Float_>& built,
    const ClassifySingleBuffers<Label_, Float_>& buffers,
    const ClassifySingleOptions<Float_>& options) 
{
    annotate_cells_simple(mat, 
        built.get_test_subset().size(), 
        built.get_test_subset().data(), 
        built.get_references(), 
        built.get_markers(), 
        options.quantile, 
        options.fine_tune, 
        options.fine_tune_threshold, 
        buffers.best, 
        buffers.scores, 
        buffers.delta,
        options.num_threads
    );
    return;
}

/**
 * @brief Results of `classify_single()`.
 * @tparam Label_ Integer type for the reference labels.
 * @tparam Float_ Floating-point type for the correlations and scores.
 */
template<typename Label_, typename Float_>
struct ClassifySingleResults {
    /**
     * @cond
     */
    ClassifySingleResults(size_t num_cells, size_t num_labels) : best(num_cells), delta(num_cells) {
        scores.reserve(num_labels);
        for (size_t l = 0; l < num_labels; ++l) {
            scores.emplace_back(num_cells);
        }
    }
    /**
     * @endcond
     */

    /** 
     * Vector of length equal to the number of cells in the test dataset,
     * containing the index of the assigned label for each cell.
     */
    std::vector<Label_> best;

    /**
     * Vector of length equal to the number of labels,
     * containing vectors of length equal to the number of cells in the test dataset.
     * Each vector corresponds to a label and contains the (non-fine-tuned) score for each cell.
     */
    std::vector<std::vector<Float_> > scores;

    /** 
     * Vector of length equal to the number of cells in the test dataset.
     * This contains the difference between the highest and second-highest scores for each cell, possibly after fine-tuning.
     */
    std::vector<Float_> delta;
};

/**
 * @cond
 */
namespace internal {

template<typename Label_, typename Float_>
ClassifySingleBuffers<Label_, Float_> results_to_buffers(ClassifySingleResults<Label_, Float_>& results) {
    ClassifySingleBuffers<Label_, Float_> output;
    output.best = results.best.data();
    output.delta = results.delta.data();
    output.scores.reserve(results.scores.size());
    for (auto& s : results.scores) {
        output.scores.emplace_back(s.data());
    }
    return output;
}

}
/**
 * @endcond
 */

/**
 * @param mat Expression matrix of the test dataset, where rows are genes and columns are cells.
 * Each row should correspond to an element of `Prebuilt::subset`.
 * @param built An object produced by `BasicBuilder::build()`.
 *
 * @return A `Results` object containing the assigned labels and scores.
 */
template<typename Label_, typename Value_, typename Index_, typename Float_>
ClassifySingleResults<Label_, Float_> classify_single(
    const tatami::Matrix<Value_, Index_>& test,
    const TrainedSingle<Index_, Float_>& built,
    const ClassifySingleOptions<Float_>& options) 
{
    ClassifySingleResults<Label_, Float_> output(test.ncol(), built.get_references().size());
    auto buffers = internal::results_to_buffers(output);
    classify_single(test, built, buffers, options);
    return output;
}

/**
 * @param mat Expression matrix of the test dataset, where rows are genes and columns are cells.
 * This may have a different ordering of genes compared to the reference matrix used in `build()`,
 * provided that all genes corresponding to `Prebuilt::subset` are present.
 * @param built An object produced by `BasicBuilder::build()`.
 * @param[in] mat_subset Pointer to an array of length equal to that of `Prebuilt::subset`,
 * containing the index of the row of `mat` corresponding to each gene in `Prebuilt::subset`.
 *
 * @return A `Results` object containing the assigned labels and scores.
 */
template<typename Label_, typename Value_, typename Index_, typename Float_>
ClassifySingleResults<Label_, Float_> classify_single(
    const tatami::Matrix<Value_, Index_>& test,
    const Index_* test_subset,
    const TrainedSingle<Index_, Float_>& built,
    const ClassifySingleOptions<Float_>& options) 
{
    ClassifySingleResults<Label_, Float_> output(test.ncol(), built.get_references().size());
    auto buffers = internal::results_to_buffers(output);
    classify_single(test, test_subset, built, buffers, options);
    return output;
}

/**
 * @param mat Expression matrix of the test dataset, where rows are genes and columns are cells.
 * @param built An object produced by `build()` with intersections.
 *
 * @return A `Results` object containing the assigned labels and scores.
 */ 
template<typename Label_, typename Value_, typename Index_, typename Float_>
ClassifySingleResults<Label_, Float_> classify_single(
    const tatami::Matrix<Value_, Index_>& test,
    const TrainedSingleIntersect<Index_, Float_>& built,
    const ClassifySingleOptions<Float_>& options) 
{
    ClassifySingleResults<Label_, Float_> output(test.ncol(), built.get_references().size());
    auto buffers = internal::results_to_buffers(output);
    classify_single(test, built, buffers, options);
    return output;
}

}

#endif
