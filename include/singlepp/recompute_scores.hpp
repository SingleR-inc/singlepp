#ifndef SINGLEPP_RECOMPUTE_SCORES_HPP
#define SINGLEPP_RECOMPUTE_SCORES_HPP

#include "compute_scores.hpp"
#include "scaled_ranks.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace singlepp {

template<class Matrix, class IntegratedReference, typename AssignPtr>
void recompute_scores(
    const Matrix* mat,
    const std::vector<AssignPtr>& assigned,
    const std::vector<IntegratedReference>& references,
    double quantile,
    std::vector<double*>& scores,
    AssignPtr* best)
{
    size_t NR = mat->nrow();
    size_t NC = mat->ncol();

    #pragma omp parallel
    {
        std::vector<typename Matrix::value_type> buffer(NR);
        auto wrk = mat->new_workspace(false);

        RankedVector<typename Matrix::value_type, int> data_ranked, data_ranked2, ref_ranked;
        data_ranked.reserve(NR);
        ref_ranked.reserve(NR);
        if constexpr(IntegratedReference::check_availability) {
            data_ranked2.reserve(NR);
        }

        std::vector<double> scaled_data(NR);
        std::vector<double> scaled_ref(NR);
        std::unordered_set<int> universe_tmp;
        std::vector<int> universe;
        std::unordered_map<int, int> mapping;
        std::vector<double> all_correlations;

        #pragma omp for
        for (size_t i = 0; i < NC; ++i) {
            // Building the universe.
            universe_tmp.clear();
            for (size_t r = 0; r < references.size(); ++r) {
                auto best = assigned[r][i];
                const auto& markers = references[r].markers[best];
                universe_tmp.insert(markers.begin(), markers.end());
            }

            universe.clear();
            universe.insert(universe.end(), universe_tmp.begin(), universe_tmp.end());
            std::sort(universe.begin(), universe.end());

            // Creating the rankings for the data.
            data_ranked.clear();
            if (universe.size()) {
                size_t first = universe.front();
                size_t last = universe.back() + 1;
                auto ptr = mat->column(i, buffer.data(), first, last, wrk.get());

                if constexpr(!IntegratedReference::check_availability) {
                    // If we don't need to check availability, the mapping can
                    // be constructed here, once per cell, rather than once per
                    // reference. We can also directly construct 'data_ranked'
                    // with the final indices, without going through 'mapping'.
                    mapping.clear();
                    for (size_t s = 0; s < universe.size(); ++s) {
                        auto u = universe[s];
                        data_ranked.emplace_back(ptr[u - first], s);
                        mapping[u] = s;
                    }

                    scaled_ref.resize(mapping.size());
                    scaled_data.resize(mapping.size());
                    scaled_ranks(data_ranked, scaled_data.data());
                } else {
                    for (auto u : universe) {
                        data_ranked.emplace_back(ptr[u - first], u);                    
                    }
                }

                std::sort(data_ranked.begin(), data_ranked.end());
            }

            // Scanning through each reference and computing the score for the best group.
            double best_score = -1000;
            int best_ref = 0;
            
            for (size_t r = 0; r < references.size(); ++r) {
                const auto& ref = references[r];

                // If we need to check availability, we reconstruct the mapping
                // for the intersection of features available in this reference.
                // We then calculate the scaled ranks for the data.
                if constexpr(IntegratedReference::check_availability) {
                    mapping.clear();
                    int counter = 0;
                    for (auto c : universe) {
                        if (ref.available.find(c) != ref.available.end()) {
                            mapping[c] = counter;
                            ++counter;
                        }
                    }
                    scaled_ref.resize(mapping.size());
                    scaled_data.resize(mapping.size());

                    data_ranked2.clear();
                    subset_ranks(data_ranked, data_ranked2, mapping);
                    scaled_ranks(data_ranked2, scaled_data.data());
                }

                // Now actually calculating the score for the best group for
                // this cell in this reference. This assumes that 'ref.ranked'
                // already contains sorted pairs where the indices refer to the
                // rows of the original data matrix.
                auto best = assigned[r][i];
                const auto& best_ranked = ref.ranked[best];
                all_correlations.clear();

                for (size_t s = 0; s < best_ranked.size(); ++s) {
                    ref_ranked.clear();
                    subset_ranks(best_ranked[s], ref_ranked, mapping);
                    scaled_ranks(ref_ranked, scaled_ref.data());
                    double cor = distance_to_correlation(scaled_ref.size(), scaled_data, scaled_ref);
                    all_correlations.push_back(cor);
                }

                double score = correlations_to_scores(all_correlations, quantile);
                if (scores[r]) {
                    scores[r][i] = score;
                }
                if (score > best_score) {
                    best_score = score;
                    best_ref = r;
                }
            }

            if (best) {
                best[i] = best_ref;
            }
        }
    }

    return;
}

}

#endif
