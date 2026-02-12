#include "Rcpp.h"
#include "singlepp/singlepp.hpp"
#include "utils.h"
#include <algorithm>
#include <vector>
#include <memory>

//' @export
// [[Rcpp::export(rng=false)]]
Rcpp::List intersect_integrate(
    Rcpp::NumericMatrix test, 
    std::vector<std::string> test_ids,
    Rcpp::List results,
    Rcpp::List refs, 
    Rcpp::List ref_ids,
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
    if (nrefs != ref_ids.size()) {
        throw std::runtime_error("'refs' and 'ref_ids' should have the same length");
    }
    if (nrefs != markers.size()) {
        throw std::runtime_error("'refs' and 'markers' should have the same length");
    }

    // Setting up the inputs.
    std::vector<std::shared_ptr<tatami::Matrix<double, int> > > rematrices;
    std::vector<std::vector<int> > reresults;
    std::vector<std::vector<std::string> > reids;
    std::vector<std::vector<int> > relabels;

    for (size_t r = 0; r < nrefs; ++r) {
        reresults.emplace_back(setup_labels(results[r]));
        Rcpp::NumericMatrix ref(refs[r]);
        rematrices.emplace_back(new tatami::DenseColumnMatrix<double, int>(ref.nrow(), ref.ncol(), std::vector<double>(ref.begin(), ref.end())));
        reids.emplace_back(ref_ids[r]);
        relabels.emplace_back(setup_labels(labels[r]));
    }

    tatami::DenseColumnMatrix<double, int> parsed_test(test.nrow(), test.ncol(), std::vector<double>(test.begin(), test.end()));

    // Building the integrated classifier.
    singlepp::TrainSingleOptions bopt;
    bopt.top = -1; // use all markers.
    std::vector<singlepp::TrainedSingleIntersect<int, double> > prebuilts; 
    prebuilts.reserve(nrefs);
    std::vector<singlepp::TrainIntegratedInput<double, int, int> > inputs; 
    inputs.reserve(nrefs);
    for (size_t r = 0; r < nrefs; ++r) {
        prebuilts.push_back(singlepp::train_single_intersect(test.nrow(), test_ids.data(), *(rematrices[r]), reids[r].data(), relabels[r].data(), setup_markers(markers[r]), bopt));
        inputs.push_back(singlepp::prepare_integrated_input_intersect(test.nrow(), test_ids.data(), *(rematrices[r]), reids[r].data(), relabels[r].data(), prebuilts.back()));
    }
    singlepp::TrainIntegratedOptions iopt;
    auto itrained = singlepp::train_integrated(std::move(inputs), iopt);

    // Scoring.
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
