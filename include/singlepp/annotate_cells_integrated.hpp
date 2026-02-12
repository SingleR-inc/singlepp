#ifndef SINGLEPP_ANNOTATE_CELLS_INTEGRATED_HPP
#define SINGLEPP_ANNOTATE_CELLS_INTEGRATED_HPP

#include "defs.hpp"

#include "tatami/tatami.hpp"

#include "scaled_ranks.hpp"
#include "SubsetRemapper.hpp"
#include "train_integrated.hpp"
#include "find_best_and_delta.hpp"
#include "fill_labels_in_use.hpp"
#include "correlations_to_score.hpp"
#include "build_reference.hpp"

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstddef>
#include <optional>

namespace singlepp {

template<bool query_sparse_, typename Value_, typename Index_, typename Label_, typename Float_, typename RefLabel_>
void annotate_cells_integrated_raw(
    const tatami::Matrix<Value_, Index_>& test,
    const TrainedIntegrated<Index_>& trained,
    const std::vector<const Label_*>& assigned,
    Float_ quantile,
    bool fine_tune,
    Float_ threshold,
    RefLabel_* best, 
    const std::vector<Float_*>& scores,
    Float_* delta,
    int num_threads
) {
    const auto NR = test.nrow();
    if (!sanisizer::is_equal(NR, trained.test_nrow)) {
        throw std::runtime_error("number of rows in 'test' do not match up with those expected by 'trained'");
    }

    bool any_ref_sparse = false, any_ref_dense = false;
    for (const auto& ref : trained.references) {
        if (ref.sparse.has_value()) {
            any_ref_sparse = true;
        } else {
            any_ref_dense = true;
        }
    }

    const auto num_universe = trained.universe.size();
    auto remap_test_to_universe = [&]() {
        if constexpr(query_sparse_) {
            auto remapping = sanisizer::create<std::vector<Index_> >(NR);
            for (I<decltype(num_universe)> u = 0; u < num_universe; ++u) {
                remapping[trained.universe[u]] = u;
            }
            return remapping;
        } else {
            return false;
        }
    }();

    tatami::parallelize([&](int, Index_ start, Index_ len) {
        SubsetRemapper<Index_> remapper(num_universe);

        RankedVector<Value_, Index_> test_ranked_full, test_ranked_sub;
        test_ranked_full.reserve(num_universe);
        test_ranked_sub.reserve(num_universe);

        RankedVector<Index_, Index_> ref_ranked;
        ref_ranked.reserve(num_universe);
        std::optional<RankedVector<Index_, Index_> > ref_ranked_alt;
        if (any_ref_sparse) {
            ref_ranked_alt.emplace();
            ref_ranked_alt->reserve(num_universe);
        }

        auto vbuffer = sanisizer::create<std::vector<Value_> >(num_universe);
        auto ibuffer = [&](){
            if constexpr(query_sparse_) {
                return sanisizer::create<std::vector<Index_> >(num_universe);
            } else {
                return false;
            }
        }();

        auto test_scaled = [&]() {
            if constexpr(query_sparse_) {
                SparseScaled<Index_, Float_> output;
                output.nonzero.reserve(num_universe);
                return output;
            } else {
                return sanisizer::create<std::vector<Float_> >(num_universe);
            }
        }();

        std::optional<SparseScaled<Index_, Float_> > sparse_ref_scaled;
        if (any_ref_sparse) {
            sparse_ref_scaled.emplace();
            sanisizer::reserve(sparse_ref_scaled->nonzero, num_universe);
        }
        std::optional<std::vector<Float_> > dense_ref_scaled;
        if (any_ref_dense) {
            dense_ref_scaled.emplace();
            sanisizer::reserve(*dense_ref_scaled, num_universe);
        }

        std::vector<Float_> all_correlations;
        const auto nref = trained.references.size();
        std::vector<RefLabel_> reflabels_in_use;
        sanisizer::reserve(reflabels_in_use, nref);
        std::vector<Float_> all_scores;
        sanisizer::reserve(all_scores, nref);

        // We perform an indexed extraction, so all subsequent indices
        // will refer to indices into this subset (i.e., 'universe').
        tatami::VectorPtr<Index_> universe_ptr(tatami::VectorPtr<Index_>{}, &(trained.universe));
        auto mat_work = tatami::consecutive_extractor<query_sparse_>(&test, false, start, len, universe_ptr);

        for (Index_ i = start, end = start + len; i < end; ++i) {
            test_ranked_full.clear();
            if constexpr(query_sparse_) {
                auto info = mat_work->fetch(vbuffer.data(), ibuffer.data());
                for (I<decltype(info.number)> i = 0; i < info.number; ++i) {
                    test_ranked_full.emplace_back(info.value[i], remap_test_to_universe[info.index[i]]);
                }
            } else {
                auto ptr = mat_work->fetch(vbuffer.data());
                for (I<decltype(num_universe)> i = 0; i < num_universe; ++i) {
                    test_ranked_full.emplace_back(ptr[i],  i);
                }
            }
            std::sort(test_ranked_full.begin(), test_ranked_full.end());

            reflabels_in_use.resize(nref);
            std::iota(reflabels_in_use.begin(), reflabels_in_use.end(), static_cast<RefLabel_>(0));
            std::pair<Label_, Float_> candidate{ 0, 1.0 };

            // For completeness, we want to compute the scores even if there is only one reference,
            // so we ask for a single iteration of the fine-tuning loop.
            bool first_iteration = true;

            while (reflabels_in_use.size() > 1 || first_iteration) {
                remapper.clear();
                for (const auto r : reflabels_in_use) {
                    const auto curassigned = assigned[r][i];
                    const auto& curref = trained.references[r];

                    if (curref.sparse.has_value()) {
                        const auto& curmarkers = (*(curref.sparse))[curassigned].markers;
                        for (const auto& x : curmarkers) {
                            remapper.add(x);
                        }
                    } else {
                        const auto& curmarkers = (*(curref.dense))[curassigned].markers;
                        for (const auto& x : curmarkers) {
                            remapper.add(x);
                        }
                    }
                }

                const auto num_markers = remapper.size();
                remapper.remap(test_ranked_full, test_ranked_sub);
                if constexpr(query_sparse_) {
                    const auto tStart = test_ranked_sub.begin(), tEnd = test_ranked_sub.end();
                    auto zero_ranges = find_zero_ranges<Value_, Index_>(tStart, tEnd);
                    scaled_ranks<Value_, Index_>(num_markers, tStart, zero_ranges.first, zero_ranges.second, tEnd, test_scaled);
                } else {
                    test_scaled.resize(num_markers);
                    scaled_ranks(num_markers, test_ranked_sub, test_scaled);
                }

                if (any_ref_dense) {
                    dense_ref_scaled->resize(num_markers);
                }

                all_scores.clear();
                for (auto r : reflabels_in_use) {
                    const auto& curref = trained.references[r];
                    const auto curassigned = assigned[r][i];
                    all_correlations.clear();

                    if (curref.sparse.has_value()) {
                        const auto& curlab = (*(curref.sparse))[curassigned];
                        const auto nsamples = curlab.num_samples;
                        all_correlations.reserve(nsamples);

                        for (I<decltype(nsamples)> s = 0; s < nsamples; ++s) {
                            ref_ranked.clear();
                            auto nStart = curlab.negative_ranked.begin();
                            remapper.remap(nStart + curlab.negative_indptrs[s], nStart + curlab.negative_indptrs[s + 1], ref_ranked);

                            ref_ranked_alt->clear();
                            auto pStart = curlab.positive_ranked.begin();
                            remapper.remap(pStart + curlab.positive_indptrs[s], pStart + curlab.positive_indptrs[s + 1], *ref_ranked_alt);

                            scaled_ranks(num_markers, ref_ranked, *ref_ranked_alt, *sparse_ref_scaled);
                            const Float_ cor = l2_to_correlation(compute_l2(num_markers, test_scaled, *sparse_ref_scaled));
                            all_correlations.push_back(cor);
                        }

                    } else {
                        const auto& curlab = (*(curref.dense))[curassigned];
                        const auto nsamples = curlab.num_samples;
                        all_correlations.reserve(nsamples);

                        for (I<decltype(nsamples)> s = 0; s < nsamples; ++s) {
                            ref_ranked.clear();
                            auto refstart = curlab.all_ranked.begin() + sanisizer::product_unsafe<std::size_t>(s, num_universe);
                            auto refend = refstart + num_universe;
                            remapper.remap(refstart, refend, ref_ranked);
                            scaled_ranks(num_markers, ref_ranked, *dense_ref_scaled);
                            const Float_ cor = l2_to_correlation(compute_l2(num_markers, test_scaled, *dense_ref_scaled));
                            all_correlations.push_back(cor);
                        }
                    }

                    const auto score = correlations_to_score(all_correlations, quantile);
                    all_scores.push_back(score);
                }

                candidate = update_labels_in_use(all_scores, threshold, reflabels_in_use);
                for (I<decltype(nref)> r = 0; r < nref; ++r) {
                    scores[r][i] = all_scores[r];
                }

                if (!fine_tune || reflabels_in_use.size() == all_scores.size()) {
                    break;
                }
                first_iteration = false;
            }

            best[i] = candidate.first;
            if (delta) {
                delta[i] = candidate.second;
            }
        }
    }, test.ncol(), num_threads);
}

template<typename Value_, typename Index_, typename Label_, typename Float_, typename RefLabel_>
void annotate_cells_integrated(
    const tatami::Matrix<Value_, Index_>& test,
    const TrainedIntegrated<Index_>& trained,
    const std::vector<const Label_*>& assigned,
    Float_ quantile,
    bool fine_tune,
    Float_ threshold,
    RefLabel_* best, 
    const std::vector<Float_*>& scores,
    Float_* delta,
    int num_threads
) {
    if (test.is_sparse()) {
        annotate_cells_integrated_raw<true>(test, trained, assigned, quantile, fine_tune, threshold, best, scores, delta, num_threads);
    } else {
        annotate_cells_integrated_raw<false>(test, trained, assigned, quantile, fine_tune, threshold, best, scores, delta, num_threads);
    }
}

}

#endif
