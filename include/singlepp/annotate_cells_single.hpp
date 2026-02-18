#ifndef SINGLEPP_ANNOTATE_CELLS_SINGLE_HPP
#define SINGLEPP_ANNOTATE_CELLS_SINGLE_HPP

#include "defs.hpp"

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "Markers.hpp"
#include "train_single.hpp"
#include "SubsetSanitizer.hpp"
#include "SubsetRemapper.hpp"
#include "find_best_and_delta.hpp"
#include "scaled_ranks.hpp"
#include "correlations_to_score.hpp"
#include "fill_labels_in_use.hpp"
#include "utils.hpp"

#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <type_traits>

namespace singlepp {

template<bool ref_sparse_, typename Index_, typename Float_>
const auto& get_per_label_references(const BuiltReference<Index_, Float_>& built) {
    if constexpr(ref_sparse_) {
        assert(built.sparse.has_value());
        assert(!built.dense.has_value());
        return *(built.sparse);
    } else {
        assert(!built.sparse.has_value());
        assert(built.dense.has_value());
        return *(built.dense);
    }
}

template<bool query_sparse_, bool ref_sparse_, typename Label_, typename Index_, typename Float_, typename Value_>
class FineTuneSingle {
private:
    std::vector<Label_> my_labels_in_use;

    SubsetRemapper<Index_> my_gene_subset;

    typename std::conditional<query_sparse_, SparseScaled<Index_, Float_>, std::vector<Float_> >::type my_scaled_query;
    typename std::conditional<ref_sparse_, SparseScaled<Index_, Float_>, std::vector<Float_> >::type my_scaled_ref;
    typename std::conditional<query_sparse_ || ref_sparse_, std::vector<Float_>, bool>::type my_dense_buffer;

    RankedVector<Value_, Index_> my_subset_query;
    RankedVector<Index_, Index_> my_subset_ref;
    typename std::conditional<ref_sparse_, RankedVector<Index_, Index_>, bool>::type my_subset_ref_alt;

    std::vector<Float_> my_all_correlations;

public:
    typedef typename std::conditional<ref_sparse_, SparsePerLabel<Index_, Float_>, DensePerLabel<Index_, Float_> >::type PerLabel;

    FineTuneSingle(const Index_ full_num_markers, const std::vector<PerLabel>& ref) : my_gene_subset(full_num_markers) {
        sanisizer::reserve(my_labels_in_use, ref.size());

        sanisizer::reserve(my_subset_query, full_num_markers); 
        if constexpr(query_sparse_) {
            sanisizer::reserve(my_scaled_query.nonzero, full_num_markers);
        } else {
            sanisizer::reserve(my_scaled_query, full_num_markers);
        }

        sanisizer::reserve(my_subset_ref, full_num_markers); 
        if constexpr(ref_sparse_) {
            sanisizer::reserve(my_subset_ref_alt, full_num_markers); 
            sanisizer::reserve(my_scaled_ref.nonzero, full_num_markers);
        } else {
            sanisizer::reserve(my_scaled_ref, full_num_markers);
        }

        I<decltype(get_num_samples(ref.front()))> max_labels = 0;
        for (const auto& curref : ref) {
            max_labels = std::max(max_labels, get_num_samples(curref));
        }
        sanisizer::reserve(my_all_correlations, max_labels);

        if constexpr(query_sparse_ || ref_sparse_) {
            sanisizer::resize(my_dense_buffer, full_num_markers);
        }
    }

    // For testing only.
    FineTuneSingle(const TrainedSingle<Index_, Float_>& trained) : 
        FineTuneSingle(trained.subset().size(), get_per_label_references<ref_sparse_>(trained.built()))
    {}

private:
    template<bool reuse_>
    void run_internal(
        const RankedVector<Value_, Index_>& input, 
        const TrainedSingle<Index_, Float_>& trained,
        const std::vector<Label_>& labels_in_use,
        typename std::conditional<reuse_, const std::vector<std::vector<Index_> >&, bool>::type reuse_neighbors,
        const std::vector<QuantileDetails<Index_, Float_> >& quantile_details,
        std::vector<Float_>& scores
    ) {
        const auto& ref = get_per_label_references<ref_sparse_>(trained.built());
        const auto& markers = trained.markers();

        my_gene_subset.clear();
        for (auto l : labels_in_use) {
            for (auto l2 : labels_in_use){ 
                for (auto c : markers[l][l2]) {
                    my_gene_subset.add(c);
                }
            }
        }
        my_gene_subset.remap(input, my_subset_query);
        const auto current_num_markers = my_gene_subset.size();

        if constexpr(query_sparse_) {
            const auto substart = my_subset_query.begin();
            const auto subend = my_subset_query.end();
            auto zero_ranges = find_zero_ranges<Value_, Index_>(substart, subend);
            scaled_ranks<Value_, Index_, Float_>(current_num_markers, substart, zero_ranges.first, zero_ranges.second, subend, my_scaled_query);

            // Sorting for a better chance of accessing contiguous memory during iterations.
            // Indices are unique so we should not need to consider the second element of each pair.
            sort_by_first(my_scaled_query.nonzero);

            if constexpr(ref_sparse_) {
                setup_sparse_l2_remapping(current_num_markers, my_scaled_query, my_dense_buffer);
            } else {
                densify_sparse_vector(current_num_markers, my_scaled_query, my_dense_buffer);
            }
        } else {
            my_scaled_query.resize(current_num_markers);
            scaled_ranks(current_num_markers, my_subset_query, my_scaled_query);
        }

        if constexpr(!ref_sparse_) {
            // No need to be safe here, as current_num_markers < full_num_markers.
            my_scaled_ref.resize(current_num_markers);
        }

        scores.clear();
        const auto nlabels_used = labels_in_use.size();
        for (I<decltype(nlabels_used)> i = 0; i < nlabels_used; ++i) {
            auto curlab = labels_in_use[i];
            const auto& curref = ref[curlab];
            const auto& curqdeets = quantile_details[curlab];
            my_all_correlations.clear();

            const auto nsamples = [&](){
                if constexpr(reuse_) {
                    return reuse_neighbors[curlab].size();
                } else {
                    return curref.num_samples;
                }
            }();

            for (I<decltype(range)> s = 0; s < nsamples; ++s) {
                const auto sample_index = [&](){
                    if constexpr(reuse_) {
                        return reuse_neighbors[curlab][s];
                    } else {
                        return s;
                    }
                }();

                // Technically we could be faster if we remembered the subset of markers from the previous fine-tuning iteration,
                // but this requires us to (possibly) make a copy of the entire reference set; we can't afford to do this in each thread.
                if constexpr(ref_sparse_) {
                    const auto nStart = current_reference.negative_ranked.begin();
                    my_gene_subset.remap(nStart + current_reference.negative_indptrs[sample_index], nStart + current_reference.negative_indptrs[sample_index + 1], my_subset_ref);
                    const auto pStart = current_reference.positive_ranked.begin();
                    my_gene_subset.remap(pStart + current_reference.positive_indptrs[sample_index], pStart + current_reference.positive_indptrs[sample_index + 1], my_subset_ref_alt);
                    scaled_ranks(current_num_markers, my_subset_ref, my_subset_ref_alt, my_scaled_ref);
                } else {
                    const auto full_num_markers = my_gene_subset.capacity();
                    const auto refstart = current_reference.all_ranked.begin() + sanisizer::product_unsafe<std::size_t>(full_num_markers, sample_index);
                    const auto refend = refstart + full_num_markers;
                    my_gene_subset.remap(refstart, refend, my_subset_ref);
                    scaled_ranks(current_num_markers, my_subset_ref, my_scaled_ref);
                }

                const Float_ l2 = [&](){
                    if constexpr(query_sparse_) {
                        if constexpr(ref_sparse_) {
                            return sparse_l2(current_num_markers, my_scaled_query, my_dense_buffer, my_scaled_ref);
                        } else {
                            return dense_l2(current_num_markers, my_dense_buffer.data(), my_scaled_ref.data());
                        }
                    } else {
                        if constexpr(ref_sparse_) {
                            densify_sparse_vector(current_num_markers, my_scaled_ref, my_dense_buffer);
                            return dense_l2(current_num_markers, my_scaled_query.data(), my_dense_buffer.data());
                        } else {
                            return dense_l2(current_num_markers, my_scaled_query.data(), my_scaled_ref.data());
                        }
                    }
                }();

                const Float_ cor = l2_to_correlation(l2);
                my_all_correlations.push_back(cor);
            };

            if constexpr(reuse_) {
                const Float_ score = truncated_correlations_to_score(my_all_correlations, curqdeets);
                scores.push_back(score);
            } else {
                const Float_ score = correlations_to_score(my_all_correlations, curqdeets);
                scores.push_back(score);
            }
        }
    }

private:
    // Provided for testing purposes only.
    template<bool reuse_>
    void run_once_internal(
        const RankedVector<Value_, Index_>& input, 
        const TrainedSingle<Index_, Float_>& trained,
        typename std::conditional<reuse_, const std::vector<std::vector<Index_> >&, bool>::type reuse_neighbors,
        const std::vector<QuantileDetails<Index_, Float_> >& quantile_details,
        std::vector<Float_>& scores
    ) {
        const auto& ref = get_per_label_references<ref_sparse_>(trained.built());
        sanisizer::resize(my_labels_in_use, ref.size());
        std::iota(my_labels_in_use.begin(), my_labels_in_use.end());
        run_internal<reuse_>(input, trained, my_labels_in_use, reuse_neighbors, quantile_details, scores);
    }

public:
    // Provided for testing purposes only.
    void run_once(
        const RankedVector<Value_, Index_>& input, 
        const TrainedSingle<Index_, Float_>& trained,
        const std::vector<QuantileDetails<Index_, Float_> >& quantile_details,
        std::vector<Float_>& scores
    ) {
        run_once_internal<false>(input, trained, false, quantile_details, scores);
    }

    // Provided for testing purposes only.
    void run_once(
        const RankedVector<Value_, Index_>& input, 
        const TrainedSingle<Index_, Float_>& trained,
        const std::vector<std::vector<Index_> >& reuse_neighbors,
        const std::vector<QuantileDetails<Index_, Float_> >& quantile_details,
        std::vector<Float_>& scores
    ) {
        run_once_internal<true>(input, trained, reuse_neighbors, quantile_details, scores);
    }

public:
    template<bool reuse_>
    std::pair<Label_, Float_> run_all_internal(
        const RankedVector<Value_, Index_>& input, 
        const TrainedSingle<Index_, Float_>& trained,
        typename std::conditional<reuse_, const std::vector<std::vector<Index_> >&, bool>::type reuse_neighbors,
        const std::vector<QuantileDetails<Index_, Float_> >& quantile_details,
        const Float_ threshold,
        std::vector<Float_>& scores
    ) {
        auto candidate = fill_labels_in_use(scores, threshold, my_labels_in_use);

        // If there's only one top label, we don't need to do anything else.
        // We also give up if every label is in range, because any subsequent
        // calculations would use all markers and just give the same result.
        while (my_labels_in_use.size() > 1 && my_labels_in_use.size() < scores.size()) {
            run_once_internal<reuse_>(input, trained, my_labels_in_use, reuse_neighbors, quantile_details, scores);
            candidate = update_labels_in_use(scores, threshold, my_labels_in_use); 
        }

        return candidate;
    }

public:
    std::pair<Label_, Float_> run_all(
        const RankedVector<Value_, Index_>& input, 
        const TrainedSingle<Index_, Float_>& trained,
        const std::vector<QuantileDetails<Index_, Float_> >& quantile_details,
        const Float_ threshold,
        std::vector<Float_>& scores
    ) {
        return run_all_internal<false>(intput, trained, false, quantile_details, threshold, scores);
    }

    std::pair<Label_, Float_> run_all(
        const RankedVector<Value_, Index_>& input, 
        const TrainedSingle<Index_, Float_>& trained,
        const std::vector<std::vector<Index_> >& reuse_neighbors,
        const std::vector<QuantileDetails<Index_, Float_> >& quantile_details,
        const Float_ threshold,
        std::vector<Float_>& scores
    ) {
        return run_all_internal<true>(intput, trained, reuse_neighbors, quantile_details, threshold, scores);
    }
};

template<typename Index_, typename Float_>
Index_ define_k_for_quantile(const Index_ num_samples, const QuantileDetails<Index_, Float_>& details) {
    // lower_index < num_samples, so we'd always get something in [1, num_samples].
    return num_samples - details.lower_index;
}

template<bool query_sparse_, bool ref_sparse_, typename Value_, typename Index_, typename Float_, typename Label_>
void annotate_cells_single_raw(
    const tatami::Matrix<Value_, Index_>& test,
    const TrainedSingle<Index_, Float_>& trained,
    Float_ quantile,
    bool fine_tune,
    Float_ threshold,
    bool reuse_neighbors,
    Label_* best, 
    const std::vector<Float_*>& scores,
    Float_* delta,
    int num_threads
) {
    const auto& subset = trained.subset();
    const Index_ num_markers = subset.size(); // cast is safe as 'subset' is a unique subset of the rows of the reference matrix.
    SubsetNoop<query_sparse_, Index_> subsorted(subset);

    const auto& built = trained.built();
    const auto& ref = get_per_label_references<ref_sparse_>(built);

    // Figuring out how many neighbors to keep and how to compute the quantiles.
    const auto num_labels = ref.size();
    std::vector<QuantileDetails<Index_, Float_> > quantile_deets;
    quantile_deets.reserve(num_labels);
    for (I<decltype(num_labels)> r = 0; r < num_labels; ++r) {
        quantile_deets[r] = prepare_quantile_details(get_num_samples(ref[r]), quantile);
    }

    std::optional<std::vector<std::vector<Index_> > > neighbors;
    if (fine_tune && reuse_neighbors) {
        neighbors.emplace();
        sanisizer::resize(*neighbors, num_labels);
        for (I<decltype(num_labels)> r = 0; r < num_labels; ++r) {
            sanisizer::resize((*neighbors)[r], define_k_for_quantile(get_num_samples(ref[r]), quantile_deets[r]));
        }
    }

    tatami::parallelize([&](int, Index_ start, Index_ length) {
        tatami::VectorPtr<Index_> subset_ptr(tatami::VectorPtr<Index_>{}, &subset);
        auto ext = tatami::consecutive_extractor<query_sparse_>(&test, false, start, length, std::move(subset_ptr));

        auto vbuffer = sanisizer::create<std::vector<Value_> >(num_markers);
        auto ibuffer = [&](){
            if constexpr(query_sparse_) {
                return sanisizer::create<std::vector<Index_> >(num_markers);
            } else {
                return false;
            }
        }();

        RankedVector<Value_, Index_> vec;
        vec.reserve(num_markers);

        std::vector<FindClosestNeighborsWorkspace<query_sparse_, ref_sparse_, Index_, Float_> > workspaces;
        workspaces.reserve(num_labels);
        for (I<decltype(num_labels)> r = 0; r < num_labels; ++r) {
            workspaces.emplace_back(num_markers, get_num_samples(ref[r]));
        }

        auto query_scaled = [&](){
            if constexpr(query_sparse_) {
                SparseScaled<Index_, Float_> output;
                sanisizer::reserve(output.nonzero, num_markers);
                return output;
            } else {
                return sanisizer::create<std::vector<Float_> >(num_markers);
            }
        }();

        std::optional<FineTuneSingle<query_sparse_, ref_sparse_, Label_, Index_, Float_, Value_> > ft;
        if (fine_tune) {
            ft.emplace(num_markers, ref);
        }
        std::vector<Float_> curscores(num_labels);

        for (Index_ c = start, end = start + length; c < end; ++c) {
            if constexpr(query_sparse_) {
                auto info = ext->fetch(vbuffer.data(), ibuffer.data());
                subsorted.fill_ranks(info, vec);
                const auto vStart = vec.begin(), vEnd = vec.end();
                auto zero_ranges = find_zero_ranges<Value_, Index_>(vStart, vEnd);
                scaled_ranks<Value_, Index_, Float_>(num_markers, vStart, zero_ranges.first, zero_ranges.second, vEnd, query_scaled);
            } else {
                auto info = ext->fetch(vbuffer.data());
                subsorted.fill_ranks(info, vec);
                scaled_ranks(num_markers, vec, query_scaled);
            }

            curscores.resize(num_labels);
            for (I<decltype(num_labels)> r = 0; r < num_labels; ++r) {
                const auto& qdeets = quantile_deets[r];
                const Index_ k = NC - qdeets.lower_index;
                auto& work = workspaces[r]; 
                find_closest_neighbors<query_sparse_, ref_sparse_>(num_markers, query_scaled, k, ref[r], work);

                // If we need to save the neighbor indices, do it before we do any popping of the queue.
                if (neighbors.has_value()) {
                    save_all_neighbors(work, (*neighbors)[r]);
                }

                const Float_ last_squared = get_furthest_neighbor(work).first;
                const Float_ last = l2_to_correlation(last_squared);
                if (qdeets.lower_index == qdeets.upper_index) {
                    curscores[r] = last;
                } else {
                    pop_furthest_neighbor(work);
                    const Float_ next_squared = get_furthest_neighbor(work).first;
                    const Float_ next = l2_to_correlation(next_squared);
                    curscores[r] = qdeets.upper_weight * next + qdeets.lower_weight * last;
                }

                if (scores[r]) {
                    scores[r][c] = curscores[r];
                }
            }

            std::pair<Label_, Float_> chosen;
            if (neighbors.has_value()) {
                chosen = ft->run_all(vec, trained, curscores, *neighbors, quantile, threshold);
            } else if (fine_tune) {
                chosen = ft->run_all(vec, trained, curscores, quantile, threshold);
            } else {
                chosen = find_best_and_delta<Label_>(curscores);
            }

            best[c] = chosen.first;
            if (delta) {
                delta[c] = chosen.second;
            }
        }
    }, test.ncol(), num_threads);

    return;
}

template<typename Value_, typename Index_, typename Float_, typename Label_>
void annotate_cells_single(
    const tatami::Matrix<Value_, Index_>& test,
    const TrainedSingle<Index_, Float_>& trained,
    Float_ quantile,
    bool fine_tune,
    Float_ threshold,
    Label_* best, 
    const std::vector<Float_*>& scores,
    Float_* delta,
    int num_threads
) {
    const auto ref_sparse = trained.built().sparse.has_value();
    if (test.is_sparse()) {
        if (ref_sparse) {
            annotate_cells_single_raw<true, true>(test, trained, quantile, fine_tune, threshold, best, scores, delta, num_threads);
        } else {
            annotate_cells_single_raw<true, false>(test, trained, quantile, fine_tune, threshold, best, scores, delta, num_threads);
        }
    } else {
        if (ref_sparse) {
            annotate_cells_single_raw<false, true>(test, trained, quantile, fine_tune, threshold, best, scores, delta, num_threads);
        } else {
            annotate_cells_single_raw<false, false>(test, trained, quantile, fine_tune, threshold, best, scores, delta, num_threads);
        }
    }
}

}

#endif
