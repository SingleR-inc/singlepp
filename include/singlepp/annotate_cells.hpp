#ifndef SINGLEPP_ANNOTATE_CELLS_HPP
#define SINGLEPP_ANNOTATE_CELLS_HPP

#include "tatami/tatami.hpp"

#include "scaled_ranks.hpp"
#include "process_features.hpp"
#include "build_indices.hpp"
#include "fine_tune.hpp"

#include <vector>
#include <algorithm>
#include <cmath>

namespace singlepp {

inline void annotate_cells_simple(
    const tatami::Matrix<double, int>* mat,
    const std::vector<int>& subset,
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

    int first = 0, last = 0;
    if (subset.size()) {
        // Assumes that 'subset' is sorted.
        first = subset.front();
        last = subset.back();
    }

    // Figuring out how many neighbors to keep and how to compute the quantiles.
    const size_t NL = ref.size();
    std::vector<int> search_k(NL);
    std::vector<std::pair<double, double> > coeffs(NL);
    for (size_t r = 0; r < NL; ++r) {
        double denom = ref[r].index->nobs() - 1;
        auto k = std::ceil(denom * quantile) + 1;
        search_k[r] = k;
        coeffs[r].first = quantile - static_cast<double>(k - 2) / denom;
        coeffs[r].second = static_cast<double>(k - 1) / denom - quantile;
    }

    #pragma omp parallel
    {
        std::vector<double> buffer(NR);
        RankedVector vec;
        vec.reserve(NR);
        auto wrk = mat->new_workspace(false);
        std::vector<double> scaled(NR);

        #pragma omp for
        for (size_t c = 0; c < NC; ++c) {
            auto ptr = mat->column(c, buffer.data(), wrk.get());
            scaled_ranks(NR, ptr, vec, scaled.data());

            std::vector<double> curscores(NL);
            for (size_t r = 0; r < NL; ++r) {
                size_t k = search_k[r];
                auto current = ref[r].index->find_nearest_neighbors(scaled.data(), k);

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
                auto tuned = fine_tune_loop(scaled.data(), ref, markers, curscores, quantile, threshold);
                best[c] = tuned.first;
                best[c] = tuned.second;
            }
        }
    }

    return;
}

}

#endif
