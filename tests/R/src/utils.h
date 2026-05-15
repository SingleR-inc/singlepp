#ifndef UTILS_H
#define UTILS_H

#include "Rcpp.h"
#include "singlepp/singlepp.hpp"
#include <vector>

std::vector<int> setup_labels(Rcpp::IntegerVector);

singlepp::PairwiseMarkers<int> setup_pairwise_markers(Rcpp::List);

singlepp::PerLabelMarkers<int> setup_per_label_markers(Rcpp::List);

#endif
