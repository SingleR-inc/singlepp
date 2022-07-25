#include "Rcpp.h"
#include "singlepp/singlepp.hpp"
#include "utils.h"
#include <algorithm>
#include <vector>
#include <memory>

//' @export
// [[Rcpp::export(rng=false)]]
Rcpp::List integrate_singlepp(
    Rcpp::NumericMatrix mat, 
    Rcpp::List results,
    Rcpp::List refs, 
    Rcpp::List labels,
    Rcpp::List markers,
    double quantile = 0.8) 
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

    // Building.
    std::vector<std::shared_ptr<tatami::Matrix<double, int> > > rematrices;
    std::vector<std::vector<int> > relabels;
    std::vector<std::vector<int> > reresults;

    for (size_t r = 0; r < nrefs; ++r) {
        Rcpp::NumericMatrix rmat(refs[r]);
        rematrices.emplace_back(new tatami::DenseColumnMatrix<double, int>(rmat.nrow(), rmat.ncol(), std::vector<double>(rmat.begin(), rmat.end())));
        relabels.emplace_back(setup_labels(labels[r]));
        reresults.emplace_back(setup_labels(results[r]));
    }

    singlepp::IntegratedBuilder builder;
    for (size_t r = 0; r < nrefs; ++r) {
        builder.add(rematrices[r].get(), relabels[r].data(), setup_markers(markers[r]));
    }
    auto finished = builder.finish();

    // Scoring.
    finished.set_quantile(quantile);

    auto parsed_mat = tatami::DenseColumnMatrix<double, int>(mat.nrow(), mat.ncol(), std::vector<double>(mat.begin(), mat.end()));
    std::vector<const int*> resptrs;
    for (const auto& res : reresults) {
        resptrs.emplace_back(res.data());
    }

    size_t mNC = parsed_mat.ncol();
    Rcpp::IntegerVector output_best(mNC);
    Rcpp::NumericVector output_delta(mNC);
    Rcpp::NumericMatrix output_scores(mNC, nrefs);

    std::vector<double*> score_ptrs(nrefs);
    for (size_t r = 0; r < nrefs; ++r) {
        score_ptrs[r] = static_cast<double*>(output_scores.begin()) + r * mNC;
    }

    finished.run(
        &parsed_mat,
        resptrs,
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
