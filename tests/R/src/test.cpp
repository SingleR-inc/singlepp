#include "Rcpp.h"
#include "singlepp/SinglePP.hpp"
#include <algorithm>

//' @useDynLib singlepp.tests
//' @importFrom Rcpp sourceCpp
// [[Rcpp::export(rng=false)]]
Rcpp::List run_singlepp(Rcpp::NumericMatrix mat, Rcpp::NumericMatrix ref, Rcpp::IntegerVector labels, Rcpp::List markers) {
    // Setting up the inputs.
    size_t mNR = mat.nrow();
    size_t mNC = mat.ncol();
    std::vector<double> mat_copy(mNR * mNC);
    std::copy(mat.begin(), mat.end(), mat_copy.begin());
    auto mat_ptr = tatami::DenseColumnMatrix<double, int>(mNR, mNC, std::move(mat_copy));

    size_t rNR = ref.nrow();
    size_t rNC = ref.ncol();
    std::vector<double> ref_copy(rNR * rNC);
    std::copy(ref.begin(), ref.end(), ref_copy.begin());
    auto ref_ptr = tatami::DenseColumnMatrix<double, int>(rNR, rNC, std::move(ref_copy));

    std::vector<int> labels2(labels.size());
    std::copy(labels.begin(), labels.end(), labels2.begin());
    for (auto& l : labels2) {
        --l; // 0-based indexing.
    }

    size_t nlabels = markers.size();
    singlepp::Markers markers2(nlabels);

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

    // Setting up the outputs.
    Rcpp::IntegerVector output_best(mNC);
    Rcpp::NumericVector output_delta(mNC);
    Rcpp::NumericMatrix output_scores(mNC, nlabels);

    std::vector<double*> score_ptrs(nlabels);
    for (size_t l = 0; l < nlabels; ++l) {
        score_ptrs[l] = static_cast<double*>(output_scores.begin()) + l * mNC;
    }

    // Running everything.
    singlepp::SinglePP runner;
    runner.set_top(100000); // no top filter, implicitly done by 'markers'.
    runner.run(
        &mat_ptr, 
        &ref_ptr,
        labels2.data(),
        markers2,
        static_cast<int*>(output_best.begin()),
        score_ptrs,
        static_cast<double*>(output_delta.begin())
    );

    for (auto& o : output_best) {
        ++o; // 1-based indexing.
    }

    return Rcpp::List::create(
        Rcpp::Named("best") = output_best,
        Rcpp::Named("scores") = output_scores, 
        Rcpp::Named("delta") = output_delta
    ); 
}
