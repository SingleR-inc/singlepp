#include "utils.h"

std::vector<int> setup_labels(Rcpp::IntegerVector labels) {
    std::vector<int> labels2(labels.size());
    std::copy(labels.begin(), labels.end(), labels2.begin());
    for (auto& l : labels2) {
        --l; // 0-based indexing.
    }
    return labels2;
}

singlepp::Markers<int> setup_markers(Rcpp::List markers) {
    size_t nlabels = markers.size();
    singlepp::Markers<int> markers2(nlabels);

    for (size_t l = 0; l < nlabels; ++l) {
        Rcpp::List inner = markers[l];
        auto& inner2 = markers2[l];
        inner2.resize(nlabels);
        for (size_t l2 = 0; l2 < nlabels; ++l2) {
            Rcpp::IntegerVector ranking = inner[l2];
            inner2[l2] = std::vector<int>(ranking.begin(), ranking.end());  
            for (auto& i : inner2[l2]) {
                --i; // 0-based indexing.
            }
        }
    }

    return markers2;
}
