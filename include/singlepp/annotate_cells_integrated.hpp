#ifndef SINGLEPP_ANNOTATE_CELLS_INTEGRATED_HPP
#define SINGLEPP_ANNOTATE_CELLS_INTEGRATED_HPP

#include "macros.hpp"

#include "tatami/tatami.hpp"

#include "scaled_ranks.hpp"
#include "find_best_and_delta.hpp"
#include "fill_labels_in_use.hpp"
#include "correlations_to_score.hpp"

#include <vector>
#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace singlepp {

namespace internal {

template<typename Index_, typename Value_, typename Float_>
struct PerReferenceIntegratedWorkspace {
    RankRemapper<Index_> intersect_mapping;
    bool direct_mapping_filled;
    RankRemapper<Index_> direct_mapping;

    RankedVector<Value_, Index_> test_ranked_full;
    RankedVector<Value_, Index_> test_ranked;
    RankedVector<Index_, Index_> ref_ranked;

    std::vector<Float_> test_scaled;
    std::vector<Float_> ref_scaled;
    std::vector<Float_> all_correlations;
};

template<typename Label_, typename Index_, typename Value_, typename Float_>
Float_ compute_single_reference_score_integrated(
    size_t ref_i,
    Label_ best,
    const std::vector<uint8_t>& check_availability,
    const std::vector<std::unordered_set<Index_> >& available,
    const std::vector<std::vector<std::vector<RankedVector<Index_, Index_> > > >& ranked,
    const std::vector<Index_>& miniverse,
    PerReferenceIntegratedWorkspace<Index_, Value_, Float_>& workspace,
    Float_ quantile)
{
    // Further subsetting to the intersection of markers that are
    // actual present in this particular reference.
    const RankRemapper<Index_>* mapping;
    if (check_availability[ref_i]) {
        const auto& cur_available = available[ref_i];
        workspace.intersect_mapping.clear();
        workspace.intersect_mapping.reserve(miniverse.size());
        for (auto c : miniverse) {
            if (cur_available.find(c) != cur_available.end()) {
                workspace.intersect_mapping.add(c);
            }
        }
        mapping = &(workspace.intersect_mapping);

    } else {
        // If we don't need to check availability of genes, we only need to
        // populate the direct mapping once per reference, because it will be
        // the same for all references that don't need an availability check.
        if (!workspace.direct_mapping_filled) {
            workspace.direct_mapping.clear();
            workspace.direct_mapping.reserve(miniverse.size());
            for (auto c : miniverse) {
                workspace.direct_mapping.add(c);
            }
            workspace.direct_mapping_filled = true;
        }
        mapping = &(workspace.direct_mapping);
    } 

    mapping->remap(workspace.test_ranked_full, workspace.test_ranked);
    workspace.test_scaled.resize(workspace.test_ranked.size());
    scaled_ranks(workspace.test_ranked, workspace.test_scaled.data());

    // Now actually calculating the score for the best group for
    // this cell in this reference. This assumes that
    // 'ranked' already contains sorted pairs where the
    // indices refer to the rows of the original data matrix.
    const auto& best_ranked = ranked[ref_i][best];
    size_t nranked = best_ranked.size();
    workspace.all_correlations.clear();
    workspace.ref_scaled.resize(workspace.test_scaled.size());

    for (size_t s = 0; s < nranked; ++s) {
        workspace.ref_ranked.clear();
        mapping->remap(best_ranked[s], workspace.ref_ranked);
        scaled_ranks(workspace.ref_ranked, workspace.ref_scaled.data());
        Float_ cor = distance_to_correlation<Float_>(workspace.test_scaled, workspace.ref_scaled);
        workspace.all_correlations.push_back(cor);
    }

    return correlations_to_score(workspace.all_correlations, quantile);
}

template<typename Value_, typename Index_, typename Label_, typename Float_, typename RefLabel_>
void annotate_cells_integrated(
    const tatami::Matrix<Value_, Index_>& test,
    const std::vector<Index_>& universe,
    const std::vector<uint8_t>& check_availability,
    const std::vector<std::unordered_set<Index_> >& available,
    const std::vector<std::vector<std::vector<Index_> > >& markers,
    const std::vector<std::vector<std::vector<RankedVector<Index_, Index_> > > >& ranked,
    const std::vector<const Label_*>& assigned,
    Float_ quantile,
    bool fine_tune,
    Float_ threshold,
    RefLabel_* best, 
    const std::vector<Float_*>& scores,
    Float_* delta,
    int num_threads)
{
    auto NR = test.nrow();
    auto nref = markers.size();

    tatami::parallelize([&](size_t, Index_ start, Index_ len) -> void {
        // We perform an indexed extraction, so all subsequent indices
        // will refer to indices into this subset (i.e., 'universe').
        tatami::VectorPtr<Index_> universe_ptr(tatami::VectorPtr<Index_>{}, &(universe));
        auto wrk = tatami::consecutive_extractor<false>(&test, false, start, len, std::move(universe_ptr));
        std::vector<Value_> buffer(universe.size());

        PerReferenceIntegratedWorkspace<Index_, Value_, Float_> workspace;
        workspace.test_ranked_full.reserve(NR);
        workspace.test_ranked.reserve(NR);
        workspace.ref_ranked.reserve(NR);

        std::unordered_set<Index_> miniverse_tmp;
        std::vector<Index_> miniverse;
        std::vector<Float_> curscores;
        std::vector<RefLabel_> reflabels_in_use;

        for (Index_ i = start, end = start + len; i < end; ++i) {
            // Extracting only the markers of the best labels for this cell.
            miniverse_tmp.clear();
            for (size_t r = 0; r < nref; ++r) {
                auto curassigned = assigned[r][i];
                const auto& curmarkers = markers[r][curassigned];
                miniverse_tmp.insert(curmarkers.begin(), curmarkers.end());
            }

            miniverse.clear();
            miniverse.insert(miniverse.end(), miniverse_tmp.begin(), miniverse_tmp.end());
            std::sort(miniverse.begin(), miniverse.end()); // sorting for consistency in floating-point summation within scaled_ranks().

            workspace.test_ranked_full.clear();
            auto ptr = wrk->fetch(buffer.data());
            for (auto u : miniverse) {
                workspace.test_ranked_full.emplace_back(ptr[u], u);
            }
            std::sort(workspace.test_ranked_full.begin(), workspace.test_ranked_full.end());

            // Scanning through each reference and computing the score for the best group.
            curscores.clear();
            workspace.direct_mapping_filled = false;
            for (size_t r = 0; r < nref; ++r) {
                auto score = compute_single_reference_score_integrated(r, assigned[r][i], check_availability, available, ranked, miniverse, workspace, quantile);
                curscores.push_back(score);
                if (scores[r]) {
                    scores[r][i] = score;
                }
            }

            std::pair<Label_, Float_> candidate;
            if (!fine_tune) {
                candidate = find_best_and_delta<Label_>(curscores);

            } else {
                candidate = fill_labels_in_use(curscores, threshold, reflabels_in_use);
                while (reflabels_in_use.size() > 1) {
                    miniverse_tmp.clear();
                    for (auto r : reflabels_in_use) {
                        auto curassigned = assigned[r][i];
                        const auto& curmarkers = markers[r][curassigned];
                        miniverse_tmp.insert(curmarkers.begin(), curmarkers.end());
                    }

                    miniverse.clear();
                    miniverse.insert(miniverse.end(), miniverse_tmp.begin(), miniverse_tmp.end());
                    std::sort(miniverse.begin(), miniverse.end()); // sorting for consistency in floating-point summation within scaled_ranks().

                    curscores.clear();
                    workspace.direct_mapping_filled = false;
                    for (auto r : reflabels_in_use) {
                        auto score = compute_single_reference_score_integrated(r, assigned[r][i], check_availability, available, ranked, miniverse, workspace, quantile);
                        curscores.push_back(score);
                    }

                    candidate = update_labels_in_use(curscores, threshold, reflabels_in_use); 
                    if (reflabels_in_use.size() == curscores.size()) { // i.e., unchanged.
                        break;
                    } 
                }
            }

            best[i] = candidate.first;
            if (delta) {
                delta[i] = candidate.second;
            }
        }
    }, test.ncol(), num_threads);
}

}

}

#endif