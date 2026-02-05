#include "Rcpp.h"
#include "singlepp/singlepp.hpp"
#include "utils.h"
#include <algorithm>

//' @useDynLib singlepp.tests
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export(rng=false)]]
Rcpp::List classify_single(
    Rcpp::NumericMatrix test, 
    Rcpp::NumericMatrix ref, 
    Rcpp::IntegerVector labels, 
    Rcpp::List markers, 
    double quantile = 0.8, 
    bool fine_tune = true, 
    double tune_thresh = 0.05,
    int top = 20) 
{
    // Setting up the inputs.
    auto parsed_test = tatami::DenseColumnMatrix<double, int>(test.nrow(), test.ncol(), std::vector<double>(test.begin(), test.end()));
    auto parsed_ref = tatami::DenseColumnMatrix<double, int>(ref.nrow(), ref.ncol(), std::vector<double>(ref.begin(), ref.end()));
    auto labels2 = setup_labels(labels);
    auto markers2 = setup_markers(markers);

    // Setting up the outputs.
    size_t mNC = parsed_test.ncol();
    size_t nlabels = markers2.size();
    Rcpp::IntegerVector output_best(mNC);
    Rcpp::NumericVector output_delta(mNC);
    Rcpp::NumericMatrix output_scores(mNC, nlabels);

    singlepp::ClassifySingleBuffers<int, double> buffers;
    buffers.best = static_cast<int*>(output_best.begin());
    buffers.delta = static_cast<double*>(output_delta.begin());
    buffers.scores.resize(nlabels);
    for (size_t l = 0; l < nlabels; ++l) {
        buffers.scores[l] = static_cast<double*>(output_scores.begin()) + l * mNC;
    }

    // Running everything.
    singlepp::TrainSingleOptions bopt;
    bopt.top = top;
    auto trained = singlepp::train_single(parsed_ref, labels2.data(), std::move(markers2), bopt);

    singlepp::ClassifySingleOptions<double> copt;
    copt.quantile = quantile;
    copt.fine_tune = fine_tune;
    copt.fine_tune_threshold = tune_thresh;
    singlepp::classify_single(parsed_test, trained, buffers, copt);

    for (auto& o : output_best) {
        ++o; // 1-based indexing.
    }

    return Rcpp::List::create(
        Rcpp::Named("best") = output_best,
        Rcpp::Named("scores") = output_scores, 
        Rcpp::Named("delta") = output_delta
    ); 
}
