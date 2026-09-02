#ifndef SINGLEPP_DEFS_HPP
#define SINGLEPP_DEFS_HPP

/**
 * @file defs.hpp
 * @brief Common definitions for **singlepp**.
 */

#ifndef SINGLEPP_CUSTOM_PARALLEL
#include "subpar/subpar.hpp"
#endif

namespace singlepp {

/**
 * Default type of the `Index_` template argument.
 * This is the type of the gene (and sample) indices, typically from the row/column indices of a `tatami::Matrix`.
 */
typedef int DefaultIndex;

/**
 * Default type of the `Label_` template argument.
 * This is the type of the label identifiers within each reference.
 */
typedef int DefaultLabel;

/**
 * Default type of the `RefLabel_` template argument.
 * This is the type of the reference identifiers during integrated classification.
 */
typedef int DefaultRefLabel;

/**
 * Default type of the `Float_` template argument.
 * This is the type of the correlations and classification scores.
 */
typedef double DefaultFloat;

/**
 * Default type of the `Value_` template argument.
 * This is the type of input data in the `tatami::Matrix`.
 */
typedef double DefaultValue;

}

#endif
