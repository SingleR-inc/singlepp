#ifndef SINGLEPP_RECOMPUTE_SCORES_HPP
#define SINGLEPP_RECOMPUTE_SCORES_HPP

#include "compute_scores.hpp"
#include "scaled_ranks.hpp"
#include "Integrator.hpp"

#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace singlepp {

class IntegratedScorer {
private:
    double quantile = SinglePP::Defaults::quantile;

public:
    IntegratedScorer& set_quantile(double q = SinglePP::Defaults::quantile) {
        quantile = q;
        return *this;
    }

private:
    /* Here, we've split out some of the functions for easier reading.
     * Otherwise everything would land in a single mega-function.
     */

    static void build_universe(int cell,
        const std::vector<const int*>& assigned,
        const std::vector<IntegratedReference>& references, 
        std::unordered_set<int>& uset, 
        std::vector<int>& uvec) 
    {
        uset.clear();
        for (size_t r = 0; r < references.size(); ++r) {
            auto best = assigned[r][cell];
            const auto& markers = references[r].markers[best];
            uset.insert(markers.begin(), markers.end());
        }

        uvec.clear();
        uvec.insert(uvec.end(), uset.begin(), uset.end());
        std::sort(uvec.begin(), uvec.end());
        return;
    }

    static void fill_ranks(
        const tatami::Matrix<double, int>* mat, 
        const std::vector<int>& universe, 
        int cell,
        std::vector<double>& buffer,
        tatami::Workspace* wrk,
        RankedVector<double, int>& data_ranked) 
    {
        data_ranked.clear();
        if (universe.empty()) {
            return;
        }

        size_t first = universe.front();
        size_t last = universe.back() + 1;
        auto ptr = mat->column(cell, buffer.data(), first, last, wrk);

        for (auto u : universe) {
            data_ranked.emplace_back(ptr[u - first], u);                    
        }
        std::sort(data_ranked.begin(), data_ranked.end());
        return;
    }

    void prepare_mapping(
        const IntegratedReference& ref, 
        const std::vector<int>& universe,
        std::unordered_map<int, int>& mapping)
    {
        mapping.clear();
        if (ref.check_availability) {
            // If we need to check availability, we reconstruct the mapping
            // for the intersection of features available in this reference.
            // We then calculate the scaled ranks for the data.
            int counter = 0;
            for (auto c : universe) {
                if (ref.available.find(c) != ref.available.end()) {
                    mapping[c] = counter;
                    ++counter;
                }
            }
        } else {
            // If we don't need to check availability, the mapping is
            // much simpler. Technically, we could do this in the outer
            // loop if none of the references required checks, but this
            // seems like a niche optimization in practical settings.
            for (size_t s = 0; s < universe.size(); ++s) {
                auto u = universe[s];
                mapping[u] = s;
            }
        } 
        return;
    }

public:
    void run(
        const tatami::Matrix<double, int>* mat,
        const std::vector<const int*>& assigned,
        const std::vector<IntegratedReference>& references,
        int* best,
        std::vector<double*>& scores,
        double* delta)
    {
        size_t NR = mat->nrow();
        size_t NC = mat->ncol();

        #pragma omp parallel
        {
            std::vector<double> buffer(NR);
            auto wrk = mat->new_workspace(false);

            RankedVector<double, int> data_ranked, data_ranked2;
            data_ranked.reserve(NR);
            data_ranked2.reserve(NR);
            RankedVector<int, int> ref_ranked;
            ref_ranked.reserve(NR);

            std::vector<double> scaled_data(NR);
            std::vector<double> scaled_ref(NR);
            std::unordered_set<int> universe_tmp;
            std::vector<int> universe;
            std::unordered_map<int, int> mapping;
            std::vector<double> all_correlations;

            #pragma omp for
            for (size_t i = 0; i < NC; ++i) {
                build_universe(i, assigned, references, universe_tmp, universe);
                fill_ranks(mat, universe, i, buffer, wrk.get(), data_ranked);

                // Scanning through each reference and computing the score for the best group.
                double best_score = -1000, next_best = -1000;
                int best_ref = 0;
                
                for (size_t r = 0; r < references.size(); ++r) {
                    const auto& ref = references[r];
                    prepare_mapping(ref, universe, mapping);

                    scaled_ref.resize(mapping.size());
                    scaled_data.resize(mapping.size());

                    data_ranked2.clear();
                    subset_ranks(data_ranked, data_ranked2, mapping);
                    scaled_ranks(data_ranked2, scaled_data.data());

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
                        next_best = best_score;
                        best_score = score;
                        best_ref = r;
                    } else if (score > next_best) {
                        next_best = score;
                    }
                }

                if (best) {
                    best[i] = best_ref;
                }
                if (delta && references.size() > 1) {
                    delta[i] = best_score - next_best;
                }
            }
        }
        return;
    }

public:
    SinglePP::Results run( 
        const tatami::Matrix<double, int>* mat,
        const std::vector<const int*>& assigned,
        const std::vector<IntegratedReference>& references)
    {
        SinglePP::Results output(mat->ncol(), references.size());
        auto scores = output.scores_to_pointers();
        run(mat, assigned, references, output.best.data(), scores, output.delta.data());
        return output;
    }
};

}

#endif
