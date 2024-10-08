// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// classify_integrate
Rcpp::List classify_integrate(Rcpp::NumericMatrix test, Rcpp::List results, Rcpp::List refs, Rcpp::List labels, Rcpp::List markers, double quantile, bool fine_tune, double tune_thresh);
RcppExport SEXP _singlepp_tests_classify_integrate(SEXP testSEXP, SEXP resultsSEXP, SEXP refsSEXP, SEXP labelsSEXP, SEXP markersSEXP, SEXP quantileSEXP, SEXP fine_tuneSEXP, SEXP tune_threshSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type test(testSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type results(resultsSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type refs(refsSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type labels(labelsSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type markers(markersSEXP);
    Rcpp::traits::input_parameter< double >::type quantile(quantileSEXP);
    Rcpp::traits::input_parameter< bool >::type fine_tune(fine_tuneSEXP);
    Rcpp::traits::input_parameter< double >::type tune_thresh(tune_threshSEXP);
    rcpp_result_gen = Rcpp::wrap(classify_integrate(test, results, refs, labels, markers, quantile, fine_tune, tune_thresh));
    return rcpp_result_gen;
END_RCPP
}
// classify_single
Rcpp::List classify_single(Rcpp::NumericMatrix test, Rcpp::NumericMatrix ref, Rcpp::IntegerVector labels, Rcpp::List markers, double quantile, bool fine_tune, double tune_thresh, int top);
RcppExport SEXP _singlepp_tests_classify_single(SEXP testSEXP, SEXP refSEXP, SEXP labelsSEXP, SEXP markersSEXP, SEXP quantileSEXP, SEXP fine_tuneSEXP, SEXP tune_threshSEXP, SEXP topSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type test(testSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type ref(refSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type labels(labelsSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type markers(markersSEXP);
    Rcpp::traits::input_parameter< double >::type quantile(quantileSEXP);
    Rcpp::traits::input_parameter< bool >::type fine_tune(fine_tuneSEXP);
    Rcpp::traits::input_parameter< double >::type tune_thresh(tune_threshSEXP);
    Rcpp::traits::input_parameter< int >::type top(topSEXP);
    rcpp_result_gen = Rcpp::wrap(classify_single(test, ref, labels, markers, quantile, fine_tune, tune_thresh, top));
    return rcpp_result_gen;
END_RCPP
}
// intersect_integrate
Rcpp::List intersect_integrate(Rcpp::NumericMatrix test, std::vector<std::string> test_ids, Rcpp::List results, Rcpp::List refs, Rcpp::List ref_ids, Rcpp::List labels, Rcpp::List markers, double quantile, bool fine_tune, double tune_thresh);
RcppExport SEXP _singlepp_tests_intersect_integrate(SEXP testSEXP, SEXP test_idsSEXP, SEXP resultsSEXP, SEXP refsSEXP, SEXP ref_idsSEXP, SEXP labelsSEXP, SEXP markersSEXP, SEXP quantileSEXP, SEXP fine_tuneSEXP, SEXP tune_threshSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type test(testSEXP);
    Rcpp::traits::input_parameter< std::vector<std::string> >::type test_ids(test_idsSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type results(resultsSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type refs(refsSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type ref_ids(ref_idsSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type labels(labelsSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type markers(markersSEXP);
    Rcpp::traits::input_parameter< double >::type quantile(quantileSEXP);
    Rcpp::traits::input_parameter< bool >::type fine_tune(fine_tuneSEXP);
    Rcpp::traits::input_parameter< double >::type tune_thresh(tune_threshSEXP);
    rcpp_result_gen = Rcpp::wrap(intersect_integrate(test, test_ids, results, refs, ref_ids, labels, markers, quantile, fine_tune, tune_thresh));
    return rcpp_result_gen;
END_RCPP
}
// intersect_single
Rcpp::List intersect_single(Rcpp::NumericMatrix test, std::vector<std::string> test_ids, Rcpp::NumericMatrix ref, std::vector<std::string> ref_ids, Rcpp::IntegerVector labels, Rcpp::List markers, double quantile, bool fine_tune, double tune_thresh, int top);
RcppExport SEXP _singlepp_tests_intersect_single(SEXP testSEXP, SEXP test_idsSEXP, SEXP refSEXP, SEXP ref_idsSEXP, SEXP labelsSEXP, SEXP markersSEXP, SEXP quantileSEXP, SEXP fine_tuneSEXP, SEXP tune_threshSEXP, SEXP topSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type test(testSEXP);
    Rcpp::traits::input_parameter< std::vector<std::string> >::type test_ids(test_idsSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type ref(refSEXP);
    Rcpp::traits::input_parameter< std::vector<std::string> >::type ref_ids(ref_idsSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type labels(labelsSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type markers(markersSEXP);
    Rcpp::traits::input_parameter< double >::type quantile(quantileSEXP);
    Rcpp::traits::input_parameter< bool >::type fine_tune(fine_tuneSEXP);
    Rcpp::traits::input_parameter< double >::type tune_thresh(tune_threshSEXP);
    Rcpp::traits::input_parameter< int >::type top(topSEXP);
    rcpp_result_gen = Rcpp::wrap(intersect_single(test, test_ids, ref, ref_ids, labels, markers, quantile, fine_tune, tune_thresh, top));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_singlepp_tests_classify_integrate", (DL_FUNC) &_singlepp_tests_classify_integrate, 8},
    {"_singlepp_tests_classify_single", (DL_FUNC) &_singlepp_tests_classify_single, 8},
    {"_singlepp_tests_intersect_integrate", (DL_FUNC) &_singlepp_tests_intersect_integrate, 10},
    {"_singlepp_tests_intersect_single", (DL_FUNC) &_singlepp_tests_intersect_single, 10},
    {NULL, NULL, 0}
};

RcppExport void R_init_singlepp_tests(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
