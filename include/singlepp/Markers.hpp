#ifndef SINGLEPP_MARKERS_HPP
#define SINGLEPP_MARKERS_HPP

#include "macros.hpp"

#include <vector>

/**
 * @file Markers.hpp
 *
 * @brief Define the `Markers` typedef.
 */

namespace singlepp {

/**
 * A vector of vectors of ranked marker lists, used to determine which genes should be used to compute correlations in `Classifier`.
 *
 * For a `Markers` object `markers`, let us consider the vector at `markers[0][1]`.
 * This vector is expected to contain the ranked indices of the marker genes for label 0 compared to label 1.
 * Typically, this vector is created by identifying the genes that are upregulated in label 0 compared to 1 and sorting by decreasing effect size.
 * Indices should refer to the rows of the reference expression matrices (i.e., `ref` in `train_single()`).
 * So, for example, `markers[0][1][0]` should contain the row index of the most upregulated gene in label 0 compared to 1.
 *
 * For a given reference dataset, the corresponding `Markers` object should have length equal to the number of labels in that reference.
 * Each middle vector (i.e., `markers[i]` for non-negative `i` less than the number of labels) should also have length equal to the number of labels.
 * Any innermost vector along the "diagonal" (i.e., `markers[i][i]`) is typically of zero length.
 * The innermost vectors that are not on the diagonal (i.e., `markers[i][j]` for `i != j`) may be of any positive length and should contain unique row indices.
 * Note that the length of all innermost vectors will be be capped by any non-negative `TrainSingleOptions::top` in `train_single()` and friends.
 *
 * As mentioned previously, the diagonal innermost vectors are typically empty, given that it makes little sense to identify upregulated markers in a label compared to itself.
 * That said, any genes stored on the diagonal will be respected and used in all gene subsets for the corresponding label.
 * This can be exploited by advanced users to efficiently store "universal" markers for a label, i.e., markers that are applicable in all comparisons to other labels.
 *
 * @tparam Index_ Integer type for the gene (row) indices.
 */
template<typename Index_>
using Markers = std::vector<std::vector<std::vector<Index_> > >;

}

#endif
