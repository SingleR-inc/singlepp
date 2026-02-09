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

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace singlepp {

template<bool query_sparse_, bool ref_sparse_, typename Value_, typename Index_, typename Label_, typename Float_, typename RefLabel_>
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

    tatami::parallelize([&](int, Index_ start, Index_ len) {
        SubsetRemapper<Index_> remapper(NR);
        RankedVector<Value_, Index_> test_ranked_full;
        test_ranked_full.reserve(NR);
        RankedVector<Index_, Index_> test_ranked, ref_ranked;
        test_ranked.reserve(NR);
        ref_ranked.reserve(NR);

        auto vbuffer = sanisizer::create<std::vector<Value_> >(NR);
        auto ibuffer = [&](){
            if constexpr(query_sparse_) {
                return sanisizer::create<std::vector<Index_> >(NR);
            } else {
                return false;
            }
        }();

        auto test_scaled = [&]() {
            if constexpr(query_sparse_) {
                SparseScaled<Index_> output;
                output.nonzero.reserve(NR);
                return output;
            } else {
                return sanisizer::create<std::vector<Index_> >(NR);
            }
        }();

        auto ref_scaled = [&]() {
            if constexpr(ref_sparse_) {
                SparseScaled<Index_> output;
                output.nonzero.reserve(NR);
                return output;
            } else {
                return sanisizer::create<std::vector<Index_> >(NR);
            }
        }();

        std::vector<Float_> all_scores;
        std::vector<RefLabel_> reflabels_in_use;
        const auto nref = trained.my_references.size();
        sanisizer::reserve(reflabels_in_use, nref);

        // We perform an indexed extraction, so all subsequent indices
        // will refer to indices into this subset (i.e., 'universe').
        tatami::VectorPtr<Index_> universe_ptr(tatami::VectorPtr<Index_>{}, &(trained.universe));
        auto mat_work = tatami::consecutive_extractor<query_sparse_>(&test, false, start, len, universe_ptr);
        const auto num_universe = trained.universe.size();

        for (Index_ i = start, end = start + len; i < end; ++i) {
            test_ranked_full.clear();
            if constexpr(query_sparse_) {
                auto info = mat_work->fetch(vbuffer.data(), ibuffer.data());
                for (I<decltype(info.number)> i = 0; i < info.number; ++i) {
                    test_ranked_full.emplace_back(info.value[i], info.index[i]);
                }
            } else {
                auto ptr = mat_work->fetch(vbuffer.data());
                for (I<decltype(num_universe)> i = 0; i < num_universe; ++i) {
                    test_ranked_full.emplace_back(ptr[i],  i);
                }
            }
            std::sort(test_ranked_full.begin(), test_ranked_full.end());

            reflabels_in_use.resize(nref);
            std::itoa(reflabels_in_use.begin(), reflabels_in_use.end(), static_cast<RefLabel_>(0));
            std::pair<Label_, Float_> candidate{ 0, 1.0 };

            while (reflabels_in_use.size() > 1) {
                remapper.clear();
                for (const auto r : reflabels_in_use) {
                    const auto& curref = trained.my_references[r];
                    auto curassigned = assigned[r][i];
                    const auto& curmarkers = curref.markers[curassigned];
                    for (const auto& x : curmarkers) {
                        remapper.add(x);
                    }
                }

                const auto num_markers = remapper.size();
                if constexpr(query_sparse_) {
                    mapping->remap(test_ranked_full, test_ranked);
                } else {
                    mapping->remap(test_ranked_full, test_ranked);
                    test_scaled.resize(num_markers);
                }
                scaled_ranks(num_markers, test_ranked, test_scaled);

                all_scores.clear();
                for (auto r : reflabels_in_use) {
                    const auto& curref = trained.my_references[r];
                    const auto nsamples = curref.num_samples;

                    for (I<decltype(nsamples)> s = 0; s < nsamples ; ++s) {
                        auto refstart = curref.all_ranked.begin(), refend = refstart;
                        if constexpr(ref_sparse_) {
                            refstart += curref.all_ranked.indptrs[s];
                            refend += curref.all_ranked.indptrs[s + 1];
                        } else {
                            const auto num = sanisizer::product_unsafe<std::size_t>(s, trained.test_nrow);
                            refstart += num;
                            refend = refstart + trained.test_nrow;
                            ref_scaled.resize(num_markers);
                        }

                        ref_ranked.clear();
                        mapping->remap(refstart, refend, ref_ranked);
                        scaled_ranks(num_markers, ref_ranked, ref_scaled);
                        const Float_ cor = distance_to_correlation(compute_l2(num_markers, test_scaled, ref_scaled));
                        workspace.all_correlations.push_back(cor);
                    }

                    const auto score = correlations_to_score(workspace.all_correlations, quantile);
                    all_scores.push_back(score);
                }

                candidate = update_labels_in_use(all_scores, threshold, reflabels_in_use); 
                if (!fine_tune || reflabels_in_use.size() == all_scores.size()) {
                    break;
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

#endif
