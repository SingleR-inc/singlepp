#ifndef SINGLEPP_MARKERS_HPP
#define SINGLEPP_MARKERS_HPP

#include "defs.hpp"

#include <vector>

/**
 * @file Markers.hpp
 *
 * @brief Define the `Markers` typedef.
 */

namespace singlepp {

/**
 * A vector of vectors of marker lists, with one list for each pairwise comparison between labels in the reference dataset.
 * This is used to determine which genes should be used to compute correlations in `train_single()`. 
 *
 * For a `PairwiseMarkers` object `markers`, consider the vector at `markers[i][j]`.
 * This vector should contain a list of marker genes for label `i` compared to label `j`.
 * Each gene is represented as the row index of the reference expression matrix, i.e., `ref` in `train_single()`. 
 * The vector should also be sorted by the "strength" of the markers such that the earliest entries are the strongest markers for that pairwise comparison.
 * Typically, this vector is created by identifying the genes that are upregulated in label `i` compared to `j` and sorting by decreasing effect size.
 * So, for example, `markers[i][j][0]` would contain the row index of the most upregulated gene in this comparison.
 *
 * For a given reference dataset, the corresponding `Markers` object should have length equal to the number of labels in that reference.
 * Each middle vector (i.e., `markers[i]` for non-negative `i` less than the number of labels) should also have length equal to the number of labels.
 * Any innermost vector along the "diagonal" (i.e., `markers[i][i]`) is typically of zero length.
 * The innermost vectors that are not on the diagonal (i.e., `markers[i][j]` for `i != j`) may be of any positive length and should contain unique row indices.
 * Note that the length of all innermost vectors will be be capped by any non-negative `TrainSingleOptions::top` in `train_single()` and friends.
 *
 * As mentioned above, the diagonal innermost vectors are typically empty, given that it makes little sense to identify upregulated markers in a label compared to itself.
 * That said, any genes stored on the diagonal will be respected and used in all gene subsets for the corresponding label.
 * This can be exploited by advanced users to store "universal" markers for a label, i.e., markers that are applicable in all comparisons to other labels.
 * (See also `PerLabelMarkers` to store label-specific markers.)
 *
 * @tparam Index_ Integer type for the gene (row) indices.
 */
template<typename Index_ = DefaultIndex>
using PairwiseMarkers = std::vector<std::vector<std::vector<Index_> > >;

/**
 * Vector of marker lists, with one list for each label in the reference dataset.
 * This is used to determine which genes should be used to compute correlations in `train_integrated()`. 
 *
 * For a `PerLabelMarkers` object `markers`, the vector at `markers[i]` will contain the markers for label `i`.
 * This combination of markers is expected to distinguish `i` from all other labels.
 * We suggest using functions like `score_markers_summary()` from the [**scran_markers**](https://github.com/libscran/scran_markers) library to obtain a suitable set of markers.
 */
template<typename Index_ = DefaultIndex>
using PerLabelMarkers = std::vector<std::vector<Index_> >;

/**
 * @cond
 */
// For back-compatibility.
template<typename Index_ = DefaultIndex>
using Markers = PairwiseMarkers<Index_>;
/**
 * @endcond
 */

}

#endif
