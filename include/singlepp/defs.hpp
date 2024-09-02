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
 * Default type for the `Index_` template argument.
 * This is the type for the gene (and sample) indices, typically from the row/column indices of a `tatami::Matrix`.
 */
typedef int DefaultIndex;

/**
 * Default type for the `Label_` template argument.
 * This is the type for the label identifiers within each reference.
 */
typedef int DefaultLabel;

/**
 * Default type for the `RefLabel_` template argument.
 * This is the type for the reference identifiers during integrated classification.
 */
typedef int DefaultRefLabel;

/**
 * Default type for the `Float_` template argument.
 * This is the type for the correlations and classification scores.
 */
typedef double DefaultFloat;

/**
 * Default type for the `Value_` template argument.
 * This is the type for input data in the `tatami::Matrix`.
 */
typedef double DefaultValue;

}

#endif
