#ifndef SINGLEPP_MACROS_HPP
#define SINGLEPP_MACROS_HPP

#ifndef SINGLEPP_CUSTOM_PARALLEL
#include "subpar/subpar.hpp"
#define SINGLEPP_CUSTOM_PARALLEL subpar::parallelize
#endif

#endif
