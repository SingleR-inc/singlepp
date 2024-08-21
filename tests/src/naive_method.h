#ifndef NAIVE_METHOD_H
#define NAIVE_METHOD_H

#include <vector>
#include <algorithm>
#include "singlepp/classify_single.hpp"
#include "fill_ranks.h"

template<class Labels>
auto split_by_label(size_t nlabels, const Labels& labels) {
    std::vector<std::vector<int> > by_labels(nlabels);
    for (size_t c = 0; c < labels.size(); ++c) {
        by_labels[labels[c]].push_back(c);
    }
    return by_labels;
}

template<class RefMatrix>
double naive_score(const std::vector<double>& scaled_test, const std::vector<int>& in_label, const RefMatrix& refs, const std::vector<int>& subset, double quantile) {
    std::vector<double> correlations;
    auto wrk = refs->dense_column(subset);
    std::vector<double> buffer(subset.size());

    for (auto l : in_label) {
        auto col = wrk->fetch(l, buffer.data());
        tatami::copy_n(col, buffer.size(), buffer.data());
        const auto scaled_ref = quick_scaled_ranks(buffer);
        correlations.push_back(singlepp::internal::distance_to_correlation<double>(scaled_test, scaled_ref));
    }

    return singlepp::internal::correlations_to_scores(correlations, quantile);
}

template<class Labels, class Matrix, class RefMatrix>
auto naive_method(size_t nlabels, const Labels& labels, const RefMatrix& refs, const Matrix& mat, const std::vector<int>& subset, double quantile) {
    singlepp::ClassifySingleResults<int, double> output(mat->ncol(), nlabels);

    auto by_labels = split_by_label(nlabels, labels);
    auto wrk = mat->dense_column(subset);
    std::vector<double> buffer(subset.size());

    for (size_t c = 0; c < mat->ncol(); ++c) {
        auto col = wrk->fetch(c, buffer.data());
        tatami::copy_n(col, buffer.size(), buffer.data());
        auto scaled = quick_scaled_ranks(buffer);

        std::vector<std::pair<double, size_t> > my_scores;
        for (size_t r = 0; r < nlabels; ++r) {
            double score = naive_score(scaled, by_labels[r], refs, subset, quantile);
            output.scores[r][c] = score;
            my_scores.emplace_back(score, r);
        }

        // Double-check that the best and delta values are computed correctly.
        std::sort(my_scores.begin(), my_scores.end());
        output.best[c] = my_scores.back().second;

        double observed_delta = my_scores[nlabels-1].first - my_scores[nlabels-2].first;
        output.delta[c] = observed_delta;
    }

    return output;
}

#endif
