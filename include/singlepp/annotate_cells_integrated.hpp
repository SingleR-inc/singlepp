#ifndef SINGLEPP_ANNOTATE_CELLS_INTEGRATED_HPP
#define SINGLEPP_ANNOTATE_CELLS_INTEGRATED_HPP

#include "defs.hpp"

#include "tatami/tatami.hpp"

#include "scaled_ranks.hpp"
#include "l2.hpp"
#include "SubsetRemapper.hpp"
#include "SubsetSanitizer.hpp"
#include "train_integrated.hpp"
#include "find_best_and_delta.hpp"
#include "fill_labels_in_use.hpp"
#include "correlations_to_score.hpp"

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstddef>
#include <optional>

namespace singlepp {

template<typename Index_>
struct TrainedIntegratedDetails {
    Index_ num_universe;
    Index_ max_num_samples = 0;
    bool any_dense = false, any_sparse = false;
};

template<typename Index_>
TrainedIntegratedDetails<Index_> interrogate_trained_integrated(const TrainedIntegrated<Index_>& trained) {
    TrainedIntegratedDetails<Index_> output;
    output.num_universe = trained.subset().size(); // safety of cast is implicit as universe is a subset of all rows in the various tatami::Matrix objects.

    for (const auto& ref : trained.references()) {
        if (ref.sparse.has_value()) {
            output.any_sparse = true;
            for (const auto& lab : *(ref.sparse)) {
                output.max_num_samples = std::max(output.max_num_samples, lab.num_samples);
            }
        } else {
            output.any_dense = true;
            for (const auto& lab : *(ref.dense)) {
                output.max_num_samples = std::max(output.max_num_samples, lab.num_samples);
            }
        }
    }

    return output;
}

template<bool query_sparse_, typename Index_, typename Value_, typename Float_>
class AnnotateIntegrated {
private:
    Index_ my_num_universe; 

    SubsetRemapper<Index_> my_remapper;
    RankedVector<Value_, Index_> my_subset_query;
    RankedVector<Index_, Index_> my_subset_ref;
    std::optional<RankedVector<Index_, Index_> > my_subset_ref_positive;

    std::vector<Float_> my_scaled_query;
    std::optional<std::vector<std::pair<Index_, Float_> > > my_scaled_ref_sparse;
    std::optional<std::vector<Float_> > my_scaled_ref_dense;
    typename std::conditional<query_sparse_, std::vector<std::pair<Index_, Float_> >, bool>::type my_scaled_query_sparse_buffer;

    std::vector<Float_> my_all_correlations;

public:
    AnnotateIntegrated(const TrainedIntegratedDetails<Index_>& details) : my_num_universe(details.num_universe), my_remapper(my_num_universe) {
        sanisizer::reserve(my_subset_query, my_num_universe);
        sanisizer::reserve(my_subset_ref, my_num_universe);
        if (details.any_sparse) {
            my_subset_ref_positive.emplace();
            my_subset_ref_positive->reserve(my_num_universe);
        }

        sanisizer::resize(my_scaled_query, my_num_universe);
        if (details.any_sparse) {
            my_scaled_ref_sparse.emplace();
            sanisizer::reserve(*my_scaled_ref_sparse, my_num_universe);
        }
        if (details.any_dense) {
            my_scaled_ref_dense.emplace();
            sanisizer::reserve(*my_scaled_ref_dense, my_num_universe);
        }
        if constexpr(query_sparse_) {
            my_scaled_query_sparse_buffer.reserve(my_num_universe);
        }

        sanisizer::reserve(my_all_correlations, details.max_num_samples);
    }

    // For testing only.
    AnnotateIntegrated(const TrainedIntegrated<Index_>& trained) :
        AnnotateIntegrated(interrogate_trained_integrated(trained))
    {}

private:
    template<bool first_, typename Label_, class RefLabelsInUse_>
    void run_internal(
        const Index_ query_index,
        const RankedVector<Value_, Index_>& query_ranked, 
        const TrainedIntegrated<Index_>& trained,
        const std::vector<const Label_*>& assigned,
        const RefLabelsInUse_& reflabels_in_use,
        const Float_ quantile,
        std::vector<Float_>& scores
    ) {
        const auto& references = trained.references();
        const auto num_refs = [&](){
            if constexpr(first_) {
                return references.size();
            } else {
                return reflabels_in_use.size();
            }
        }();

        my_remapper.clear();
        for (I<decltype(num_refs)> r = 0; r < num_refs; ++r) {
            const auto ref_index = [&](){
                if constexpr(first_) {
                    return r;
                } else {
                    return reflabels_in_use[r];
                }
            }();

            const auto curassigned = assigned[ref_index][query_index];
            const auto& curref = references[ref_index];

            if (curref.sparse.has_value()) {
                const auto& curmarkers = (*(curref.sparse))[curassigned].markers;
                for (const auto& x : curmarkers) {
                    my_remapper.add(x);
                }
            } else {
                const auto& curmarkers = (*(curref.dense))[curassigned].markers;
                for (const auto& x : curmarkers) {
                    my_remapper.add(x);
                }
            }
        }

        const auto num_markers = my_remapper.size();
        my_remapper.remap(query_ranked, my_subset_query);
        bool query_has_nonzero = false;
        if constexpr(query_sparse_) {
            const auto sStart = my_subset_query.begin(), sEnd = my_subset_query.end();
            auto zero_ranges = find_zero_ranges<Value_, Index_>(sStart, sEnd);
            query_has_nonzero = scaled_ranks_sparse<Index_, Value_, Float_>(
                num_markers,
                sStart,
                zero_ranges.first,
                zero_ranges.second,
                sEnd,
                my_scaled_query_sparse_buffer,
                my_scaled_query.data()
            );
        } else {
            query_has_nonzero = scaled_ranks_dense(
                num_markers,
                my_subset_query,
                my_scaled_query.data()
            );
        }

        if (my_scaled_ref_dense.has_value()) {
            my_scaled_ref_dense->resize(num_markers);
        }

        scores.clear();
        for (I<decltype(num_refs)> r = 0; r < num_refs; ++r) {
            const auto ref_index = [&](){
                if constexpr(first_) {
                    return r;
                } else {
                    return reflabels_in_use[r];
                }
            }();

            const auto& curref = references[ref_index];
            const auto curassigned = assigned[ref_index][query_index];
            my_all_correlations.clear();

            if (curref.sparse.has_value()) {
                const auto& curlab = (*(curref.sparse))[curassigned];
                for (I<decltype(curlab.num_samples)> s = 0; s < curlab.num_samples; ++s) {
                    my_subset_ref.clear();
                    auto nStart = curlab.negative_ranked.begin();
                    my_remapper.remap(nStart + curlab.negative_indptrs[s], nStart + curlab.negative_indptrs[s + 1], my_subset_ref);

                    my_subset_ref_positive->clear();
                    auto pStart = curlab.positive_ranked.begin();
                    my_remapper.remap(pStart + curlab.positive_indptrs[s], pStart + curlab.positive_indptrs[s + 1], *my_subset_ref_positive);

                    const auto l2 = scaled_ranks_sparse_l2(
                        num_markers,
                        my_scaled_query.data(),
                        query_has_nonzero,
                        my_subset_ref,
                        *my_subset_ref_positive,
                        *my_scaled_ref_sparse
                    );

                    const Float_ cor = l2_to_correlation(l2);
                    my_all_correlations.push_back(cor);
                }

            } else {
                const auto& curlab = (*(curref.dense))[curassigned];
                for (I<decltype(curlab.num_samples)> s = 0; s < curlab.num_samples; ++s) {
                    my_subset_ref.clear();
                    auto refstart = curlab.all_ranked.begin() + sanisizer::product_unsafe<std::size_t>(s, my_num_universe);
                    auto refend = refstart + my_num_universe;
                    my_remapper.remap(refstart, refend, my_subset_ref);

                    const auto l2 = scaled_ranks_dense_l2(
                        num_markers,
                        my_scaled_query.data(),
                        my_subset_ref,
                        my_scaled_ref_dense->data()
                    );

                    const Float_ cor = l2_to_correlation(l2);
                    my_all_correlations.push_back(cor);
                }
            }

            const auto score = correlations_to_score(my_all_correlations, quantile);
            scores.push_back(score);
        }
    }

public:
    template<typename Label_>
    void run_first(
        const Index_ query_index,
        const RankedVector<Value_, Index_>& query_ranked, 
        const TrainedIntegrated<Index_>& trained,
        const std::vector<const Label_*>& assigned,
        const Float_ quantile,
        std::vector<Float_>& scores
    ) {
        run_internal<true>(query_index, query_ranked, trained, assigned, false, quantile, scores);
    }

    template<typename Label_, typename RefLabel_>
    std::pair<RefLabel_, Float_> run_fine(
        const Index_ query_index,
        const RankedVector<Value_, Index_>& query_ranked, 
        const TrainedIntegrated<Index_>& trained,
        const std::vector<const Label_*>& assigned,
        const Float_ quantile,
        const Float_ threshold,
        std::vector<Float_>& scores,
        std::vector<RefLabel_>& reflabels_in_use
    ) {
        auto candidate = fill_labels_in_use(scores, threshold, reflabels_in_use);
        while (reflabels_in_use.size() > 1 && reflabels_in_use.size() < scores.size()) {
            run_internal<false>(query_index, query_ranked, trained, assigned, reflabels_in_use, quantile, scores);
            candidate = update_labels_in_use(scores, threshold, reflabels_in_use);
        }
        return candidate;
    }
};

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
    if (!sanisizer::is_equal(NR, trained.test_nrow())) {
        throw std::runtime_error("number of rows in 'test' do not match up with those expected by 'trained'");
    }

    const auto& subset = trained.subset();
    SubsetNoop<query_sparse_, Index_> subsorted(subset);

    const auto details = interrogate_trained_integrated(trained);
    const auto num_universe = details.num_universe;

    const auto nref = trained.references().size();
    sanisizer::cast<RefLabel_>(nref); // checking that it'll fit in the output.

    tatami::parallelize([&](int, Index_ start, Index_ len) {
        AnnotateIntegrated<query_sparse_, Index_, Value_, Float_> ft(details);
        RankedVector<Value_, Index_> test_ranked_full;
        test_ranked_full.reserve(num_universe);

        auto vbuffer = sanisizer::create<std::vector<Value_> >(num_universe);
        auto ibuffer = [&](){
            if constexpr(query_sparse_) {
                return sanisizer::create<std::vector<Index_> >(num_universe);
            } else {
                return false;
            }
        }();

        std::vector<Float_> all_scores;
        sanisizer::reserve(all_scores, nref);
        std::optional<std::vector<RefLabel_> > reflabels_in_use;
        if (fine_tune) {
            reflabels_in_use.emplace();
            sanisizer::reserve(*reflabels_in_use, nref);
        }

        // We perform an indexed extraction, so all subsequent indices
        // will refer to indices into this subset (i.e., 'universe').
        tatami::VectorPtr<Index_> universe_ptr(tatami::VectorPtr<Index_>{}, &subset);
        auto mat_work = tatami::consecutive_extractor<query_sparse_>(&test, false, start, len, std::move(universe_ptr));

        for (Index_ i = start, end = start + len; i < end; ++i) {
            const auto info = [&](){
                if constexpr(query_sparse_) {
                    return mat_work->fetch(vbuffer.data(), ibuffer.data());
                } else {
                    return mat_work->fetch(vbuffer.data());
                }
            }();
            subsorted.fill_ranks(info, test_ranked_full);

            ft.run_first(i, test_ranked_full, trained, assigned, quantile, all_scores);
            for (I<decltype(nref)> r = 0; r < nref; ++r) {
                scores[r][i] = all_scores[r];
            }

            std::pair<RefLabel_, Float_> candidate;
            if (fine_tune) {
                candidate = ft.run_fine(i, test_ranked_full, trained, assigned, quantile, threshold, all_scores, *reflabels_in_use);
            } else {
                candidate = find_best_and_delta<RefLabel_>(all_scores);
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
