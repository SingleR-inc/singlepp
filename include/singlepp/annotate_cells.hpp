#ifndef SINGLEPP_ANNOTATE_CELLS_HPP
#define SINGLEPP_ANNOTATE_CELLS_HPP

#include "tatami/tatami.h"
#include "knncolle/knncolle.hpp"

#include "scaled_ranks.hpp"
#include "process_features.hpp"
#include "build_indices.hpp"

#include <vector>
#include <algorithm>
#include <cmath>

namespace singlepp {

inline void annotate_cells_simple(
    const tatami::Matrix<double, int>* mat, 
    const std::vector<Reference>& ref,
    const Markers& markers,
    double quantile,
    bool fine_tune,
    double threshold,
    int* best, 
    std::vector<double*>& scores,
    double* delta) 
{
    const size_t NR = mat->nrow();
    const size_t NC = mat->ncol();

    // Figuring out how many neighbors to keep and how to compute the quantiles.
    const size_t NL = ref.size();
    std::vector<int> search_k(NL);
    std::vector<std::pair<double, double> > coeffs(NL);
    for (size_t r = 0; r < NL; ++r) {
        double denom = ref[r].index->nobs() - 1;
        search_k[r] = std::ceil(denom * quantile) + 1;
        coeffs[r].first = quantile - static_cast<double>(k - 2) / denom;
        coeffs[r].second = static_cast<double>(k - 1) / denom - quantile;
    }

    #pragma omp parallel
    {
        std::vector<double> buffer(NR);
        RankedVector vec;
        vec.reserve(NR);
        auto wrk = mat->new_workspace(false);

        #pragma omp for
        for (size_t c = 0; c < NC; ++c) {
            auto ptr = mat->column(c, buffer.data(), wrk.get());
            scaled_ranks(NR, ptr, vec, output);

            std::vector<double> curscores(NL);
            for (size_t r = 0; r < NL; ++r) {
                size_t k = search_k[r];
                auto current = ref[r]->query_nearest_neighbors(output.data(), k);

                double last = current[k - 1].second;
                last = 1 - 2 * last * last;
                if (k == 1) {
                    curscores[r] = last;
                } else {
                    double second = current[k - 2].second;
                    second = 1 - 2 * second * second;
                    curscores[r] = coeffs[r].first * second + coeffs[r].second * last;
                }

                if (scores[r]) {
                    scores[r][c] = curscores[r];
                }
            }

            if (!fine_tune) {
                auto top = std::max_element(curscores.begin(), curscores.end());
                best[c] = top - curscores.begin();
                if (curscores.size() > 1) {
                    double topscore = *top;
                    *top = -100;
                    delta[c] = topscore - *std::max_element(curscores.begin(), curscores.end());
                } else {
                    delta[c] = std::numeric_limits<double>::quiet_NaN();
                }
            } else {
                // enter fine-tuning logic here.
            }
        }
    }

    return;
}

template<class Mat, typename Id, class Builder>
void annotate_cells_simple(
    const tatami::Matrix<double, int>* mat, 
    const Id* mat_id,
    const std::vector<Mat*>& ref,
    const Id* ref_id,
    const Builder& build,
    Markers markers,
    double quantile,
    bool fine_tune,
    double threshold,
    int* best, 
    std::vector<double*>& scores,
    double* delta) 
{
    auto intersection = intersect_features(mat->nrow(), mat_id, ref->nrow(), ref_id);
    auto subset = subset_markers(intersection, markers, top);
    auto unzip = unzip(intersection);
    auto submat = tatami::make_DelayedSubset(tatami::wrap_subset_ptr(mat), std::move(unzip.first));
    auto subref = build_indices(unzip.second, ref, build);
    annotate_cells_simple(submat.get(), subref, markers, quantile, fine_tune, threshold, best, scores, delta);
    return;
}

template<class Mat, typename Id, class Builder>
void annotate_cells_simple(
    const tatami::Matrix<double, int>* mat, 
    const std::vector<Mat*>& ref,
    const Builder& build,
    const Markers& markers,
    double quantile,
    bool fine_tune,
    double threshold,
    int* best, 
    std::vector<double*>& scores,
    double* delta) 
{
    auto subset = subset_markers(markers, top);
    auto subref = build_indices(subset, ref, build);
    auto submat = tatami::make_DelayedSubset(tatami::wrap_subset_ptr(mat), std::move(subset));
    annotate_cells_simple(submat.get(), subref, markers, quantile, fine_tune, threshold, best, scores, delta);
    return;
}

}

#endif
