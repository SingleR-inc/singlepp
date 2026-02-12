#include "Rcpp.h"
#include "singlepp/singlepp.hpp"
#include "utils.h"
#include <algorithm>
#include <vector>
#include <memory>

//' @export
// [[Rcpp::export(rng=false)]]
Rcpp::List classify_integrate(
    Rcpp::NumericMatrix test, 
    Rcpp::List results,
    Rcpp::List refs, 
    Rcpp::List labels,
    Rcpp::List markers,
    double quantile = 0.8,
    bool fine_tune = true, 
    double tune_thresh = 0.05)
{
    size_t nrefs = refs.size();
    if (nrefs != results.size()) {
        throw std::runtime_error("'refs' and 'results' should have the same length");
    }
    if (nrefs != labels.size()) {
        throw std::runtime_error("'refs' and 'labels' should have the same length");
    }
    if (nrefs != markers.size()) {
        throw std::runtime_error("'refs' and 'markers' should have the same length");
    }

    // Building the integrated classifier.
    std::vector<std::shared_ptr<tatami::Matrix<double, int> > > rematrices;
    std::vector<std::vector<int> > relabels;
    std::vector<std::vector<int> > reresults;

    for (size_t r = 0; r < nrefs; ++r) {
        Rcpp::NumericMatrix ref(refs[r]);
        rematrices.emplace_back(new tatami::DenseColumnMatrix<double, int>(ref.nrow(), ref.ncol(), std::vector<double>(ref.begin(), ref.end())));
        relabels.emplace_back(setup_labels(labels[r]));
        reresults.emplace_back(setup_labels(results[r]));
    }

    singlepp::TrainSingleOptions bopt;
    bopt.top = -1; // use all markers.
    std::vector<singlepp::TrainedSingle<int, double> > prebuilts;
    prebuilts.reserve(nrefs);
    std::vector<singlepp::TrainIntegratedInput<double, int, int> > inputs; 
    inputs.reserve(nrefs);

    for (size_t r = 0; r < nrefs; ++r) {
        prebuilts.push_back(singlepp::train_single(*(rematrices[r]), relabels[r].data(), setup_markers(markers[r]), bopt));
        inputs.push_back(singlepp::prepare_integrated_input(*(rematrices[r]), relabels[r].data(), prebuilts.back()));
    }
    singlepp::TrainIntegratedOptions iopt;
    auto itrained = singlepp::train_integrated(std::move(inputs), iopt);

    // Scoring.
    tatami::DenseColumnMatrix<double, int> parsed_test(test.nrow(), test.ncol(), std::vector<double>(test.begin(), test.end()));
    std::vector<const int*> resptrs;
    for (const auto& res : reresults) {
        resptrs.emplace_back(res.data());
    }

    size_t mNC = parsed_test.ncol();
    Rcpp::IntegerVector output_best(mNC);
    Rcpp::NumericVector output_delta(mNC);
    Rcpp::NumericMatrix output_scores(mNC, nrefs);

    singlepp::ClassifyIntegratedBuffers<int, double> buffers;
    buffers.best = static_cast<int*>(output_best.begin());
    buffers.delta = static_cast<double*>(output_delta.begin());
    buffers.scores.resize(nrefs);
    for (size_t r = 0; r < nrefs; ++r) {
        buffers.scores[r] = static_cast<double*>(output_scores.begin()) + r * mNC;
    }

    singlepp::ClassifyIntegratedOptions<double> copt;
    copt.quantile = quantile;
    copt.fine_tune = fine_tune;
    copt.fine_tune_threshold = tune_thresh;
    singlepp::classify_integrated(parsed_test, resptrs, itrained, buffers, copt);

    for (auto& o : output_best) {
        ++o; // 1-based indexing.
    }

    return Rcpp::List::create(
        Rcpp::Named("best") = output_best,
        Rcpp::Named("scores") = output_scores, 
        Rcpp::Named("delta") = output_delta
    ); 
}
