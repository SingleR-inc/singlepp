#ifndef SINGLEPP_BASIC_BUILDER_HPP
#define SINGLEPP_BASIC_BUILDER_HPP

#include "macros.hpp"

#include "knncolle/knncolle.hpp"
#include "tatami/tatami.hpp"

#include "build_indices.hpp"
#include "subset_to_markers.hpp"

#include <vector>
#include <memory>

/**
 * @file BasicBuilder.hpp
 *
 * @brief Defines the `BasicBuilder` class.
 */

namespace singlepp {

/**
 * @brief Options for `build_single()`.
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam Float_ Floating-point type for the correlation calculations.
 */
template<typename Index_, typename Float_>
struct BuildSingleOptions {
    /**
     * Number of top markers to use from each pairwise comparison between labels.
     * Larger values improve the stability of the correlations at the cost of increasing noise and computational work.
     *
     * Setting this to a negative value will instruct `build_single()` to use all supplied markers.
     * This is useful in situations where the supplied markers have already been curated.
     */
    int top = -1;

    /**
     * Algorithm for the nearest-neighbor search.
     * This allows us to skip the explicit calculation of correlations between each test cell and every reference sample.
     * If NULL, this defaults to an exact search based on `knncolle::VptreeBuilder`.
     */
    std::shared_ptr<knncolle::Builder<knncolle::SimpleMatrix<Index_, Index_, Float_>, Float_> builder;

    /**
     * Number of threads to use.
     */
    int num_threads = 1;
};

/**
 * @cond
 */
namespace internal {

template<typename Value_, typename Index_, typename Label_, typename Float_>
std::vector<PerLabelReference<Index_, Float_> > build_references(
    const tatami::Matrix<Value_, Index_>& ref,
    const Label_* labels,
    const std::vector<Index_>& subset,
    const BuildSingleOptions<Index_, Float_>& options) 
{
    if (options.builder) {
        return build_indices(ref, labels, subset, *(options.builder), num_threads);
    } else {
        return build_indices(ref, labels, subset, knncolle::VptreeBuilder<knncolle::SimpleMatrix<Index_, Index_, Float_>, Float_>(), num_threads);
    }
}

}
/**
 * @endcond
 */

/**
 * @brief Prebuilt classifier from a single reference.
 *
 * Instances of this class should not be directly constructed, but instead returned by calling `build_single()` on a reference dataset.
 * Each instance can be used in `classify_single()` with a test dataset that has the same number and order of genes as the reference dataset. 
 * 
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam Float_ Floating-point type for the correlation calculations.
 */
template<typename Index_, typename Float_>
class PrebuiltSingle {
    /**
     * @cond
     */
    Prebuilt(Markers<Index_> markers, std::vector<Index_> subset, std::vector<PerLabelReference<Index_, Float_> > references) :
        my_markers(std::move(markers)), my_subset(std::move(subset)), my_references(std::move(references)) {}
    /**
     * @endcond
     */

private:
    Markers<Index_> my_markers;
    std::vector<Index_> my_subset;
    std::vector<PerLabelReference<Index_, Float_> > references;

public:
    /**
     * @return A vector of vectors of ranked marker genes to be used in the classification.
     * Values are indices into the subset vector (see `get_subset()`).
     * The set of marker genes is a subset of those in the input `markers` in `build_single()`.
     */
    const Markers<Index_>& get_markers() const {
        return my_markers;
    }

    /**
     * @return The subset of genes in the test/reference datasets that were used in the classification.
     * Values are row indices into each matrix.
     */
    const std::vector<Index_>& get_subset() const {
        return my_subset;
    }

    /**
     * @return Number of labels in this reference.
     */
    size_t num_labels() const {
        return my_references.size();
    }

    /**
     * @return Number of profiles in this reference.
     */
    size_t num_profiles() const {
        size_t n = 0;
        for (const auto& ref : my_references) {
            n += ref.ranked.size();
        }
        return n;
    }
};

/**
 * Prepare a single labelled reference dataset for use in `classify_single()`.
 * This involves pre-ranking the markers based on the expression in each sample of the reference dataset,
 * so that the Spearman correlations can be computed without redundant sorting.
 * We also construct neighbor search indices for rapid calculation of the classification score.
 *
 * Note that the pre-built classifier should only be used with test datasets that have the same genes as the reference dataset.
 * If the test dataset has different genes, consider using `build_single_intersect()` instead.
 * 
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam Label_ Integer type for the reference labels.
 * @tparam Float_ Floating-point type for the correlation calculations.
 *
 * @param ref Matrix for the reference expression profiles.
 * Rows are genes while columns are samples.
 * @param[in] labels An array of length equal to the number of columns of `ref`, containing the label for each sample.
 * Labels should be integers in \f$[0, L)\f$ where \f$L\f$ is the total number of unique labels.
 * @param markers A vector of vectors of ranked marker genes for each pairwise comparison between labels, see `Markers` for more details.
 * @param options Further options.
 *
 * @return A pre-built classifier that can be used in `classify_single()` with a test dataset that has the same genes as `ref`.
 */
template<typename Value_, typename Index_, typename Label_, typename Float_>
PrebuiltSingle<Index_, Float_> build_single(
    const tatami::Matrix<Value_, Index_>& ref,
    const Label_* labels,
    Markers<Index_> markers,
    const BuildSingleOptions<Index_, Float_>& options)
{
    auto subset = subset_to_markers(markers, options.top);
    auto subref = internal::build_references(ref, labels, subset);
    return PrebuiltSingle<Index_, Float_>(std::move(markers), std::move(subset), std::move(subref), options);
}

/**
 * @brief Prebuilt classifier from an intersection of genes.
 *
 * Instances of this class should not be directly constructed, but instead returned by calling `build_single_intersect()`.
 * This uses the intersection of genes between the test dataset and those of the reference dataset.
 * Each instance can be used in `classify_single()` with a test dataset that has the specified genes.
 * 
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam Float_ Floating-point type for the correlation calculations.
 */
template<typename Index_, typename Float_>
struct PrebuiltSingleIntersect {
    /**
     * @cond
     */
    PrebuiltSingleIntersect(Markers<Index_> markers, std::vector<Index_> test_subset, std::vector<Index_> ref_subset, std::vector<PerLabelReference<Index_, Float_> > references) :
        my_markers(std::move(markers)), my_test_subset(std::move(test_subset)), my_ref_subset(std::move(ref_subset)), my_references(std::move(references)) {}
    /**
     * @endcond
     */

private:
    Markers<Index_> my_markers;
    std::vector<Index_> my_test_subset;
    std::vector<Index_> my_ref_subset;
    std::vector<PerLabelReference<Index_, Float_> > my_references;

public:
    /**
     * @return A vector of vectors of ranked marker genes to be used in the classification.
     * Values are indices into the subset vectors from `get_test_subset()` and `get_ref_subset()` for their respective matrices.
     * The set of marker genes is typically a subset of those in the input `markers` in `build_single_intersect()`.
     */
    const Markers<Index_>& get_markers() const {
        return my_markers;
    }

    /**
     * @return Subset of genes in the intersection for the test dataset.
     * These are sorted and unique indices into the `test_id` array supplied to `build_single_intersect()`.
     * This has the same length as the subset vector returned by `get_ref_subset(0`, where corresponding entries refer to the same genes in the respective datasets.
     */
    const std::vector<Index_>& get_test_subset() const {
        return my_test_subset;
    }

    /**
     * @return Subset of genes in the intersection for the test dataset.
     * These are unique (but not necessarily sorted!) indices into the `ref_id` matrix supplied to `build_single_intersect()`.
     * This has the same length as the subset vector returned by `get_test_subset()`, where corresponding entries refer to the same genes in the respective datasets.
     */
    const std::vector<Index_>& get_ref_subset() const {
        return my_ref_subset;
    }

    /**
     * @return Number of labels in this reference.
     */
    size_t num_labels() const {
        return my_references.size();
    }

    /**
     * @return Number of profiles in this reference.
     */
    size_t num_profiles() const {
        size_t n = 0;
        for (const auto& ref : my_references) {
            n += ref.ranked.size();
        }
        return n;
    }
};

/**
 * @cond
 */
namespace internal {

template<typename Value_, typename Index_, typename Label_, typename Float_>
std::vector<PerLabelReference<Index_, Float_> > build_intersection(
    internal::Intersection<Index_> intersection,
    const tatami::Matrix<Value_, Index_>& ref,
    const Label_* labels,
    Markers<Index_> markers,
    const BuildSingleOptions<Index_, Float_>& options) 
{
    subset_to_markers(intersection, markers, options.top);
    auto pairs = unzip(intersection);
    auto subref = internal::build_references(ref, labels, pairs.second, options);
    return PrebuiltSingleIntersect<Index_, Float_>(std::move(markers), std::move(pairs.first), std::move(pairs.second), std::move(subref));
}

}
/**
 * @endcond
 */

/**
 * Variant of `build_single()` that uses a pre-computed intersection of genes between the reference dataset and an as-yet-unspecified test dataset.
 * Most users will prefer to use the other `build_single_intersect()` overload that accepts `test_id` and `ref_id` and computes the intersection automatically;
 *
 * Note that the pre-built classifier should only be used with a test dataset that has all genes specified in `intersection`.
 * (Specifically, in the first entry of each element of `intersection`.)
 *
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam Label_ Integer type for the reference labels.
 * @tparam Float_ Floating-point type for the correlation calculations.
 *
 * @tparam test_ngenes Number of genes in the test dataset (i.e., rows of the test matrix).
 * @param intersection Vector defining the intersection of genes betweent the test and reference datasets.
 * Each entry is a pair where the first element is the row index in the test matrix,
 * and the second element is the row index for the corresponding feature in the reference matrix.
 * Each row index for either matrix should occur no more than once in `intersection`.
 * @param ref An expression matrix for the reference expression profiles, where rows are genes and columns are cells.
 * This should have non-zero columns.
 * @param[in] labels An array of length equal to the number of columns of `ref`, containing the label for each sample.
 * Labels should be integers in \f$[0, L)\f$ where \f$L\f$ is the total number of unique labels.
 * @param markers A vector of vectors of ranked marker genes for each pairwise comparison between labels, see `Markers` for more details.
 * @param options Further options.
 *
 * @return A pre-built classifier that can be used in `classify_single()` with a test dataset that has all genes in `intersection`.
 *
 */
template<typename Index_, typename Value_, typename Label_, typename Float_>
PrebuiltSingleIntersect<Index_, Float_> build_single_intersect(
    Index_ test_ngenes,
    std::vector<std::pair<Index_, Index_> > intersection,
    const tatami::Matrix<Value_, Index_>& ref, 
    const Label_* labels,
    Markers<Index_> markers,
    const BuildSingleOptions<Index_, Float_>& options)
{
    // Sorting it if it wasn't already.
    if (std::is_unsorted(intersection.begin(), intersection.end())) {
        std::sort(intersection.begin(), intersection.end());
    }

    internal::Intersection<Index_> temp;
    temp.pairs.swap(intersection);
    temp.test_n = test_ngenes;
    temp.ref_n = ref.nrow();

    return internal::build_intersection(std::move(temp), ref, labels, std::move(markers), options);
}

/**
 * Variant of `build_single()` that uses the intersection of genes between the reference dataset and a (future) test dataset.
 * This is useful when the genes are not in the same order and number across the test and reference datasets.
 *
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam Id_ Integer type to use as the gene identifier for each row.
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam Label_ Integer type for the reference labels.
 * @tparam Float_ Floating-point type for the correlation calculations.
 *
 * @param test_nrow Number of rows (genes) in the test dataset.
 * @param[in] test_id Pointer to an array of identifiers of length equal to `test_nrow`.
 * This should contain a unique identifier for each row of `mat` (typically a gene name or index).
 * If any duplicate IDs are present, only the first occurrence is used.
 * @param ref An expression matrix for the reference expression profiles, where rows are genes and columns are cells.
 * This should have non-zero columns.
 * @param[in] ref_id Pointer to an array of identifiers of length equal to the number of rows of any `ref`.
 * This should contain a unique identifier for each row in `ref`, and should be comparable to `test_id`.
 * If any duplicate IDs are present, only the first occurrence is used.
 * @param[in] labels An array of length equal to the number of columns of `ref`, containing the label for each sample.
 * Labels should be integers in \f$[0, L)\f$ where \f$L\f$ is the total number of unique labels.
 * @param markers A vector of vectors of ranked marker genes for each pairwise comparison between labels, see `Markers` for more details.
 *
 * @return A `PrebuiltIntersection` instance that can be used in `run()` for annotation of a test dataset with the same order of genes as specified in `test_id`.
 * @param options Further options.
 */
template<typename Index_, typename Id_, typename Value_, typename Label_, typename Float_>
PrebuiltIntersection<Index_, Float_> run(
    Index_ test_nrow,
    const Id_* test_id, 
    const tatami::Matrix<Value_, Index_>& ref, 
    const Id_* ref_id, 
    const Label_* labels,
    Markers<Index_> markers,
    const BuildSingleOptions<Index_, Float_>& options)
{
    auto intersection = intersect_genes(test_nrow, test_id, ref->nrow(), ref_id);
    return internal::build_intersection(std::move(intersection), ref, labels, std::move(markers), options);
}

}

#endif
