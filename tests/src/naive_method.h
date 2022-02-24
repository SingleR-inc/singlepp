#ifndef NAIVE_METHOD_H
#define NAIVE_METHOD_H

#include <vector>
#include <algorithm>
#include "singlepp/SinglePP.hpp"

template<class Labels, class Matrix, class RefMatrix>
auto naive_method(size_t nlabels, const Labels& labels, const RefMatrix& refs, const Matrix& mat, const std::vector<int>& subset, double quantile) {
    singlepp::SinglePP::Results output(mat->ncol(), nlabels);

    std::vector<std::vector<int> > by_labels(nlabels);
    for (size_t r = 0; r < refs->ncol(); ++r) {
        by_labels[labels[r]].push_back(r);
    }

    for (size_t c = 0; c < mat->ncol(); ++c) {
        auto col = mat->column(c);
        singlepp::RankedVector<double, int> vec(subset.size());
        singlepp::fill_ranks(subset, col.data(), vec);
        std::vector<double> scaled(subset.size());
        singlepp::scaled_ranks(vec, scaled.data());

        std::vector<std::pair<double, size_t> > my_scores;
        for (size_t r = 0; r < nlabels; ++r) {
            std::vector<double> correlations;
            for (auto l : by_labels[r]) {
                auto col2 = refs->column(l);
                singlepp::RankedVector<double, int> vec2(subset.size());
                singlepp::fill_ranks(subset, col2.data(), vec2);
                std::vector<double> scaled2(subset.size());
                singlepp::scaled_ranks(vec2, scaled2.data());
                correlations.push_back(singlepp::distance_to_correlation(scaled.size(), scaled.data(), scaled2.data()));
            }

            double score = singlepp::correlations_to_scores(correlations, quantile);
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
