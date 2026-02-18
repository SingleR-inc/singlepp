#ifndef SINGLEPP_ANNOTATE_CELLS_INTEGRATED_HPP
#define SINGLEPP_ANNOTATE_CELLS_INTEGRATED_HPP

#include "defs.hpp"

#include "tatami/tatami.hpp"

#include "scaled_ranks.hpp"
#include "SubsetRemapper.hpp"
#include "SubsetSanitizer.hpp"
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

template<typename Index_>
struct TrainedIntegratedDetails {
    Index_ num_universe;
    std::vector<std::vector<Index_> > num_samples;
    bool any_dense = false, any_sparse = false;
};

template<typename Index_>
TrainedIntegratedDetails<Index_> interrogate_trained_integrated(const TrainedIntegrated<Index_>& trained) {
    TrainedIntegratedDetails<Index_> output;
    output.num_universe = trained.subset().size(); // safety of cast is implicit as universe is a subset of all rows in the various tatami::Matrix objects.
    output.num_samples.reserve(trained.references().size());

    for (const auto& ref : trained.references()) {
        output.num_samples.emplace_back();
        auto& cur_num_samples = output.num_samples.back();

        if (ref.sparse.has_value()) {
            output.any_sparse = true;
            sanisizer::reserve(cur_num_samples, ref.sparse->size());
            for (const auto& lab : *(ref.sparse)) {
                cur_num_samples.push_back(lab.num_samples);
            }

        } else {
            output.any_dense = true;
            sanisizer::reserve(cur_num_samples, ref.dense->size());
            for (const auto& lab : *(ref.dense)) {
                cur_num_samples.push_back(lab.num_samples);
            }
        }
    }

    return output;
}

template<bool query_sparse_, bool reuse_, typename Index_, typename Value_, typename Float_, typename RefLabel_>
class AnnotateIntegratedCore {
private:
    Index_ my_num_universe; 
    SubsetRemapper<Index_> my_remapper;

    RankedVector<Value_, Index_> my_subset_query;
    RankedVector<Index_, Index_> my_subset_ref;
    std::optional<RankedVector<Index_, Index_> > my_subset_ref_positive;

    typename std::conditional<query_sparse_, SparseScaled<Index_, Float_>, std::vector<Float_> >::type my_scaled_query;
    std::optional<SparseScaled<Index_, Float_> > my_scaled_ref_sparse;
    std::optional<std::vector<Float_> > my_scaled_ref_dense;

    std::optional<std::vector<Float_> > my_sparse_remapping, my_densified_buffer;

    std::vector<RefLabel_> my_references_in_use;
    std::vector<std::vector<QuantileDetails<Index_, Float_> > > my_quantile_details;

    std::vector<Float_> my_all_correlations;
    typename std::conditional<reuse_, std::vector<std::pair<Float_, Index_> >, bool>::type my_all_correlations_indexed;

public:
    AnnotateIntegratedCore(const TrainedIntegratedDetails<Index_>& details, Float_ quantile) : 
        my_num_universe(details.num_universe),
        my_remapper(my_num_universe)
    {
        sanisizer::reserve(my_subset_query, my_num_universe);
        sanisizer::reserve(my_subset_ref, my_num_universe);

        if (details.any_sparse) {
            my_subset_ref_positive.emplace();
            my_subset_ref_positive->reserve(my_num_universe);
            my_scaled_ref_sparse.emplace();
            sanisizer::reserve(my_scaled_ref_sparse->nonzero, my_num_universe);
        }

        if (details.any_dense) {
            my_scaled_ref_dense.emplace();
            sanisizer::reserve(*my_scaled_ref_dense, my_num_universe);
        }

        if constexpr(query_sparse_) {
            my_scaled_query.nonzero.reserve(my_num_universe);
        } else {
            sanisizer::resize(my_scaled_query, my_num_universe);
        }

        if constexpr(query_sparse_) {
            if (details.any_dense) {
                my_densified_buffer.emplace();
                sanisizer::resize(*my_densified_buffer, my_num_universe);
            }
            if (details.any_sparse) {
                my_sparse_remapping.emplace();
                sanisizer::resize(*my_sparse_remapping, my_num_universe);
            }
        } else {
            if (details.any_sparse) {
                my_densified_buffer.emplace();
                sanisizer::resize(*my_densified_buffer, my_num_universe);
            }
        }

        const auto num_ref = details.num_samples.size();
        sanisizer::cast<RefLabel_>(num_ref); // Make sure implicit casts are safe in run_rest().
        sanisizer::reserve(my_references_in_use, num_ref);
        sanisizer::reserve(my_quantile_details, num_ref);

        Index_ max_num_samples = 0;
        for (const auto& ref_num_samples : details.num_samples) {
            my_quantile_details.emplace_back();
            auto& ref_quantile_details = my_quantile_details.back();
            sanisizer::reserve(ref_quantile_details, ref_num_samples.size());
            for (auto lab_num_samples : ref_num_samples) {
                max_num_samples = std::max(max_num_samples, lab_num_samples);
                ref_quantile_details.push_back(prepare_quantile_details(lab_num_samples, quantile));
            }
        }

        sanisizer::reserve(my_all_correlations, max_num_samples);
        if constexpr(reuse_) {
            my_all_correlations_indexed.emplace();
            sanisizer::reserve(my_all_correlations_indexed, max_num_samples);
        }
    }

public:
    void reserve_reuse_neighbors(std::vector<std::vector<Index_> >& reuse_neighbors) const {
        const auto num_ref = my_quantile_details.size();
        sanisizer::reserve(reuse_neighbors, num_ref);
        for (const auto& refquant : my_quantile_details) {
            reuse_neighbors.emplace_back();
            auto& curneighbors = reuse_neighbors.back();
            Index_ max_index = 0;
            for (auto labquant : refquant) {
                // Increment is safe as upper_index < number of samples, and the latter is known to fit in an Index_.
                max_index = std::max(max_index, static_cast<Index_>(labquant.upper_index + 1));
            }
            sanisizer::reserve(curneighbors, max_index);
        }
    }

public:
    template<bool first_> 
    using ReuseNeighborsArg = typename std::conditional<
        reuse_, 
        typename std::conditional<
            first_,
            std::vector<std::vector<Index_> >&,
            const std::vector<std::vector<Index_> >&
        >::type,
        bool
    >::type;

    template<bool first_, typename Label_>
    void run(
        const Index_ query_index,
        const RankedVector<Value_, Index_>& query_ranked, 
        const TrainedIntegrated<Index_>& trained,
        const std::vector<const Label_*>& assigned,
        ReuseNeighborsArg<first_> reuse_neighbors,
        std::vector<Float_>& scores
    ) {
        /******* Setting up the marker remapping *******/

        const auto& references = trained.references();
        const auto num_refs = [&](){
            if constexpr(first_) {
                return references.size();
            } else {
                return my_reflabels_in_use.size();
            }
        }();

        for (I<decltype(num_refs)> r = 0; r < num_refs; ++r) {
            const auto ref_index = [&](){
                if constexpr(first_) {
                    return r;
                } else {
                    return my_reflabels_in_use[r];
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

        /******* Computing scaled ranks for the query *******/

        const auto num_markers = my_remapper.size();
        my_remapper.remap(query_ranked, my_subset_query);
        if constexpr(query_sparse_) {
            const auto sStart = my_subset_query.begin(), sEnd = my_subset_query.end();
            auto zero_ranges = find_zero_ranges<Value_, Index_>(sStart, sEnd);
            scaled_ranks<Value_, Index_>(num_markers, sStart, zero_ranges.first, zero_ranges.second, sEnd, my_scaled_query);

            // Sorting for a better chance of accessing contiguous memory during iterations.
            // Indices are unique so we should not need to consider the second element of each pair.
            sort_by_first(my_scaled_query.nonzero);

            if (my_scaled_ref_dense.has_value()) {
                densify_sparse_vector(num_markers, my_scaled_query, *my_densified_buffer);
            } else {
                setup_sparse_l2_remapping(num_markers, my_scaled_query, *my_sparse_remapping);
            }
        } else {
            my_scaled_query.resize(num_markers);
            scaled_ranks(num_markers, my_subset_query, my_scaled_query);
        }

        if (my_scaled_ref_dense.has_value()) {
            // No need to be safe with the resize here, as num_markers < num_universe.
            my_scaled_ref_dense->resize(num_markers);
        }

        /******* Iterating over references *******/

        scores.clear();
        for (I<decltype(num_refs)> r = 0; r < num_refs; ++r) {
            const auto ref_index = [&](){
                if constexpr(first_) {
                    return r;
                } else {
                    return my_reflabels_in_use[r];
                }
            }();

            const auto& curref = references[ref_index];
            const auto curassigned = assigned[ref_index][query_index];
            const auto& curqdeets = my_quantile_details[ref_index][curassigned];

            if constexpr(first_ && reuse_) {
                my_all_correlations_indexed.clear();
            } else {
                my_all_correlations.clear();
            }

            if (curref.sparse.has_value()) {
                const auto& curlab = (*(curref.sparse))[curassigned];
                const auto num_samples = [&](){
                    if constexpr(!first_ && reuse_) {
                        return reuse_neighbors[ref_index].size();
                    } else  {
                        return curref.num_samples;
                    }
                }();

                for (I<decltype(num_samples)> s = 0; s < num_samples; ++s) {
                    const auto sample_index = [&](){
                        if constexpr(!first_ && !reuse_) {
                            return reuse_neighbors[ref_index][s];
                        } else {
                            return s;
                        }
                    }();

                    my_subset_ref.clear();
                    auto nStart = curlab.negative_ranked.begin();
                    my_remapper.remap(nStart + curlab.negative_indptrs[s], nStart + curlab.negative_indptrs[s + 1], my_subset_ref);

                    my_subset_ref_positive->clear();
                    auto pStart = curlab.positive_ranked.begin();
                    my_remapper.remap(pStart + curlab.positive_indptrs[s], pStart + curlab.positive_indptrs[s + 1], *my_subset_ref_positive);

                    scaled_ranks(num_markers, my_subset_ref, *my_subset_ref_positive, *my_scaled_ref_sparse);
                    const Float_ l2 = [&](){
                        if constexpr(query_sparse_) {
                            return sparse_l2(num_markers, my_scaled_query, *my_sparse_remapping, *my_scaled_ref_sparse);
                        } else {
                            densify_sparse_vector(num_markers, *my_scaled_ref_sparse, *my_densified_buffer);
                            return dense_l2(num_markers, my_scaled_query.data(), my_densified_buffer->data());
                        }
                    }();

                    const Float_ cor = l2_to_correlation(l2);
                    if constexpr(first_ && reuse_) {
                        my_all_correlations_indexed->push_back(cor, s);
                    } else {
                        my_all_correlations->push_back(cor);
                    }
                }

            } else {
                const auto& curlab = (*(curref.dense))[curassigned];
                const auto num_samples = [&](){
                    if constexpr(!first_ && reuse_) {
                        return reuse_neighbors[ref_index].size();
                    } else  {
                        return curref.num_samples;
                    }
                }();

                for (const auto s : samples_of_interest) {
                    const auto sample_index = [&](){
                        if constexpr(!first_ && !reuse_) {
                            return reuse_neighbors[ref_index][s];
                        } else {
                            return s;
                        }
                    }();

                    my_subset_ref.clear();
                    auto refstart = curlab.all_ranked.begin() + sanisizer::product_unsafe<std::size_t>(s, my_num_universe);
                    auto refend = refstart + my_num_universe;
                    my_remapper.remap(refstart, refend, my_subset_ref);

                    scaled_ranks(num_markers, my_subset_ref, *my_scaled_ref_dense);
                    const Float_ l2 = [&](){
                        if constexpr(query_sparse_) {
                            return dense_l2(num_markers, my_densified_buffer->data(), my_scaled_ref_dense->data());
                        } else {
                            return dense_l2(num_markers, my_scaled_query.data(), my_scaled_ref_dense->data());
                        }
                    }();

                    const Float_ cor = l2_to_correlation(l2);
                    if constexpr(first_ && reuse_) {
                        my_all_correlations_indexed->push_back(cor, s);
                    } else {
                        my_all_correlations->push_back(cor);
                    }
                }
            }

            if constexpr(first_ && reuse_) {
                const Float_ score = correlations_to_score(my_all_correlations_indexed, curqdeets, reuse_neighbors[ref_index]);
                scores.push_back(score);
            } else if constexpr(reuse_) {
                const Float_ score = truncated_correlations_to_score(my_all_correlations, curqdeets);
                scores.push_back(score);
            } else {
                const Float_ score = correlations_to_score(my_all_correlations, curqdeets);
                scores.push_back(score);
            }
        }
    }

public:
    template<typename Label_>
    std::pair<RefLabel_, Float_> run_rest(
        const Index_ query_index,
        const RankedVector<Value_, Index_>& query_ranked, 
        const TrainedIntegrated<Index_>& trained,
        const std::vector<const Label_*>& assigned,
        ReuseNeighborsArg<false> reuse_neighbors,
        std::vector<Float_>& scores,
        Float_ threshold
    ) {
        assert(sanisizer::is_equal(scores.size(), trained.references().size()));
        auto candidate = fill_labels_in_use(scores, threshold, my_reflabels_in_use);
        while (reflabels_in_use.size() > 1 && reflabels_in_use.size() < scores.size()) {
            run_internal<false>(query_index, query_ranked, trained, assigned, reuse_neighbors, scores);
            candidate = update_labels_in_use(scores, threshold, my_reflabels_in_use);
        }
        return candidate;
    }
};

template<bool query_sparse_, typename Index_, typename Value_, typename Float_, typename RefLabel_>
class AnnotateIntegratedSimple {
public:
    AnnotateIntegratedSimple(const TrainedIntegratedDetails<Index_>& details, Float_ quantile) : my_core(details, quantile) {}

private:
    AnnotateIntegratedCore<query_sparse_, false, Index_, Value_, Float_, RefLabel_> my_core;

public:  
    template<typename Label_>
    void run_first(
        const Index_ query_index,
        const RankedVector<Value_, Index_>& query_ranked, 
        const TrainedIntegrated<Index_>& trained,
        const std::vector<const Label_*>& assigned,
        std::vector<Float_>& scores
    ) {
        my_core.template run<true>(query_index, query_ranked, trained, assigned, false, scores);
    }

    template<typename Label_>
    std::pair<RefLabel_, Float_> run_rest(
        const Index_ query_index,
        const RankedVector<Value_, Index_>& query_ranked, 
        const TrainedIntegrated<Index_>& trained,
        const std::vector<const Label_*>& assigned,
        std::vector<Float_>& scores,
        Float_ threshold
    ) {
        return my_core.run_rest(query_index, query_ranked, trained, assigned, false, scores, threshold);
    }
};

template<bool query_sparse_, typename Index_, typename Value_, typename Float_, typename RefLabel_>
class AnnotateIntegratedReuse {
public:
    AnnotateIntegratedReuse(const TrainedIntegratedDetails<Index_>& details, Float_ quantile) : my_core(details, quantile) {}

private:
    AnnotateIntegratedCore<query_sparse_, true, Index_, Value_, Float_, RefLabel_> my_core;

public:
    void reserve_reuse_neighbors(std::vector<std::vector<Index_> >& reuse_neighbors) const {
        my_core.reserve_reuse_neighbors(reuse_neighbors);
    }

    template<typename Label_>
    void run_first(
        const Index_ query_index,
        const RankedVector<Value_, Index_>& query_ranked, 
        const TrainedIntegrated<Index_>& trained,
        const std::vector<const Label_*>& assigned,
        std::vector<std::vector<Index_> >& reuse_neighbors,
        std::vector<Float_>& scores
    ) {
        my_core.template run<true>(query_index, query_ranked, trained, assigned, reuse_neighbors, scores);
    }

    // Only implemented for testing purposes.
    // The idea is to call run_first_again() after run_first() and check that the scores are unchanged.
    template<typename Label_>
    void run_first_again(
        const Index_ query_index,
        const RankedVector<Value_, Index_>& query_ranked, 
        const TrainedIntegrated<Index_>& trained,
        const std::vector<const Label_*>& assigned,
        const std::vector<std::vector<Index_> >& reuse_neighbors,
        std::vector<Float_>& scores
    ) {
        my_core.template run<false>(query_index, query_ranked, trained, assigned, reuse_neighbors, scores);
    }

    template<typename Label_>
    std::pair<RefLabel_, Float_> run_rest(
        const Index_ query_index,
        const RankedVector<Value_, Index_>& query_ranked, 
        const TrainedIntegrated<Index_>& trained,
        const std::vector<const Label_*>& assigned,
        const std::vector<std::vector<Index_> >& reuse_neighbors,
        std::vector<Float_>& scores,
        Float_ threshold
    ) {
        return my_core.run_rest(query_index, query_ranked, trained, assigned, reuse_neighbors, scores, threshold);
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
    bool reuse_neighbors,
    RefLabel_* best, 
    const std::vector<Float_*>& scores,
    Float_* delta,
    int num_threads
) {
    const auto NR = test.nrow();
    if (!sanisizer::is_equal(NR, trained.test_nrow())) {
        throw std::runtime_error("number of rows in 'test' do not match up with those expected by 'trained'");
    }

    const auto details = interrogate_trained_integrated(trained);
    const auto& subset = trained.subset();
    SubsetNoop<query_sparse_, Index_> subsorted(subset);
    const auto num_universe = details.num_universe;

    tatami::parallelize([&](int, Index_ start, Index_ len) {
        std::optional<AnnotateIntegratedSimple<query_sparse_, Index_, Value_, Float_> > ft_simple;
        std::optional<AnnotateIntegratedReuse<query_sparse_, Index_, Value_, Float_> > ft_reuse;
        std::optional<std::vector<std::vector<Index_> > > reuse_neighbors;
        if (fine_tune && reuse_neighbors) {
            ft_reuse.emplace(details, quantile);
            reuse_neighbors.emplace();
            ft_reuse->reserve_reuse_neighbors(*reuse_neighbors);
        } else {
            ft_simple.emplace(details, quantile);
        }

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

        const auto nref = trained.references().size();
        std::vector<Float_> all_scores;
        sanisizer::reserve(all_scores, nref);

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

            if (ft_simple.has_value()) {
                ft_simple->run_first(i, test_ranked_full, trained, assigned, all_scores);
            } else {
                ft_reuse->run_first(i, test_ranked_full, trained, assigned, *reuse_neighbors, all_scores);
            }
            for (I<decltype(nref)> r = 0; r < nref; ++r) {
                scores[r][i] = all_scores[r];
            }

            std::pair<RefLabel_, Float_> candidate;
            if (!fine_tune) {
                candidate = find_best_and_delta<RefLabel_>(all_scores);
            } else if (ft_simple.has_value()) {
                candidate = ft_simple->run_rest(i, test_ranked_full, trained, assigned, all_scores, threshold);
            } else {
                candidate = ft_reuse->run_rest(i, test_ranked_full, trained, assigned, all_scores, *reuse_neighbors, threshold);
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
