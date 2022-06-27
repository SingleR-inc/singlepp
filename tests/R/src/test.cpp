#include "Rcpp.h"
#include "singlepp/singlepp.hpp"
#include "utils.h"
#include <algorithm>

//' @useDynLib singlepp.tests
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export(rng=false)]]
Rcpp::List run_singlepp(
    Rcpp::NumericMatrix mat, 
    Rcpp::NumericMatrix ref, 
    Rcpp::IntegerVector labels, 
    Rcpp::List markers, 
    double quantile = 0.8, 
    bool fine_tune = true, 
    double tune_thresh = 0.05,
    int top = 20) 
{
    // Setting up the inputs.
    auto parsed_mat = tatami::DenseColumnMatrix<double, int>(mat.nrow(), mat.ncol(), std::vector<double>(mat.begin(), mat.end()));
    auto parsed_ref = tatami::DenseColumnMatrix<double, int>(ref.nrow(), ref.ncol(), std::vector<double>(ref.begin(), ref.end()));
    auto labels2 = setup_labels(labels);
    auto markers2 = setup_markers(markers);

    // Setting up the outputs.
    size_t mNC = parsed_mat.ncol();
    size_t nlabels = markers.size();
    Rcpp::IntegerVector output_best(mNC);
    Rcpp::NumericVector output_delta(mNC);
    Rcpp::NumericMatrix output_scores(mNC, nlabels);

    std::vector<double*> score_ptrs(nlabels);
    for (size_t l = 0; l < nlabels; ++l) {
        score_ptrs[l] = static_cast<double*>(output_scores.begin()) + l * mNC;
    }

    // Running everything.
    singlepp::Classifier runner;
    runner.set_top(top).set_quantile(quantile).set_fine_tune(fine_tune).set_fine_tune_threshold(tune_thresh);

    runner.run(
        &parsed_mat, 
        &parsed_ref,
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
