#ifndef TATAMI_FINE_TUNE_HPP
#define TATAMI_FINE_TUNE_HPP

#include <vector>
#include <algorithm>
#include <unordered_set>

#include "scaled_ranks.hpp"
#include "process_features.hpp"

namespace singlepp {

inline std::pair<int, double> fill_labels_in_use(const std::vector<double>& scores, std::vector<int>& in_use) {
    in_use.clear();
    auto it = std::max_element(scores.begin(), scores.end());
    int best_label = it - scores.begin();
    double max_score = *it;

    constexpr double DUMMY = -1000;
    double next_score = DUMMY;

    double threshold = max_score - tune_thresh;
    for (size_t i = 0; i < scores.size(); ++i) {
        const auto& val = scores[i];
        if (val >= threshold) {
            labels_in_use.push_back(i);
        }
        if (i != best_label && next_score < val) {
            next_score = val;
        }
    }

    return std::make_pair(best_label, max_score - next_score); 
}

std::pair<int, double> fine_tune(
    const double* ptr, 
    const std::vector<Reference>& ref,
    const Markers& markers,
    std::vector<double> scores,
    double quantile,
    double threshold)
{
    if (scores.size() <= 1) {
        return std::make_pair(0, std::numeric_limits<double>::quiet_NaN());
    } 

    std::vector<int> labels_in_use;
    auto candidate = fill_labels_in_use(scores, labels_in_use);
    if (labels_in_use.size() == 1) {
        return candidate;
    }

    std::unordered_set<int> genes_in_use;
    RankedVector ranked;
    std::vector<double> scaled_left, scaled_right;
    std::vector<double> all_correlations;

    while (labels_in_use.size() > 1) {
        genes_in_use.clear();
        for (auto l : labels_in_use) {
            for (auto l2 : labels_in_use){ 
                if (l != l2) {
                    const auto& current = markers[l][l2];
                    genes_in_use.insert(current.begin(), current.end());
                }
            }
        }

        ranked.reserve(genes_in_use.size());
        scaled_left.resize(genes_in_use.size());
        scaled_ranks(ptr, genes_in_use, ranked, scaled_left.data());

        scaled_right.resize(genes_in_use.size());
        scores.clear();

        for (size_t i = 0; i < labels_in_use.size(); ++i) {
            all_correlations.clear();
            const auto& curref = ref[labels_in_use[i]];
            size_t NR = curref.index->nobs();

            for (size_t c = 0; c < ncells; ++c) {
                auto rightptr = ref.data.data() + c * NR;
                scaled_ranks(rightptr, genes_in_use, ranked, scaled_right.data());
                double cor = distance_to_correlation(scaled_left.size(), scaled_left, scaled_right);
                all_correlations.push_back(cor);
            }

            double score = correlations_to_scores(all_correlations, quantile);
            scores.push_back(score);
        }

        candidate = fill_labels_in_use(scores, labels_in_use); 
        if (labels_in_use.size() == scores.size()) { // i.e., unchanged.
            break;
        }
    }

    return candidate;
}

}

#endif
