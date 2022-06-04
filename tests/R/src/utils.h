#ifndef UTILS_H
#define UTILS_H

#include "Rcpp.h"
#include "singlepp/SinglePP.hpp"
#include <vector>

std::vector<int> setup_labels(Rcpp::IntegerVector);

singlepp::Markers setup_markers(Rcpp::List);

#endif
