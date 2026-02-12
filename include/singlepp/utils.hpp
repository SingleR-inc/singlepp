#ifndef SINGLEPP_UTILS_HPP
#define SINGLEPP_UTILS_HPP

#include <type_traits>

namespace singlepp {

template<typename Input_>
using I = std::remove_cv_t<std::remove_reference_t<Input_> >;

}

#endif
