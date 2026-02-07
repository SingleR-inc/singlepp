#ifndef SINGLEPP_ANNOTATE_CELLS_SINGLE_HPP
#define SINGLEPP_ANNOTATE_CELLS_SINGLE_HPP

#include "defs.hpp"

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "Markers.hpp"
#include "build_reference.hpp"
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

template<bool query_sparse_, bool ref_sparse_, typename Label_, typename Index_, typename Float_, typename Value_>
class FineTuneSingle {
private:
    std::vector<Label_> my_labels_in_use;

    SubsetRemapper<Index_> my_gene_subset;

    typename std::conditional<query_sparse_, SparseScaled<Index_, Float_>, std::vector<Float_> >::type my_scaled_query;
    typename std::conditional<ref_sparse_, SparseScaled<Index_, Float_>, std::vector<Float_> >::type my_scaled_ref;

    RankedVector<Value_, Index_> my_subset_query;
    RankedVector<Index_, Index_> my_subset_ref;

    std::vector<Float_> my_all_correlations;

public:
    typedef typename std::conditional<ref_sparse_, SparsePerLabelReference<Index_, Float_>, DensePerLabelReference<Index_, Float_> >::type PerLabelReference;

    FineTuneSingle(const Index_ max_markers, const std::vector<PerLabelReference>& ref) : my_gene_subset(max_markers) {
        sanisizer::reserve(my_labels_in_use, ref.size());

        if constexpr(query_sparse_) {
            sanisizer::reserve(my_scaled_query.nonzero, max_markers);
        } else {
            sanisizer::reserve(my_scaled_query, max_markers);
        }

        if constexpr(ref_sparse_) {
            sanisizer::reserve(my_scaled_ref.nonzero, max_markers);
        } else {
            sanisizer::reserve(my_scaled_ref, max_markers);
        }

        sanisizer::reserve(my_subset_query, max_markers); 
        sanisizer::reserve(my_subset_ref, max_markers); 

        I<decltype(get_num_samples(ref.front()))> max_labels = 0;
        for (const auto& curref : ref) {
            max_labels = std::max(max_labels, get_num_samples(curref));
        }
        sanisizer::reserve(my_all_correlations, max_labels);
    }

public:
    std::pair<Label_, Float_> run(
        const RankedVector<Value_, Index_>& input, 
        const std::vector<PerLabelReference>& ref,
        const Markers<Index_>& markers,
        std::vector<Float_>& scores,
        Float_ quantile,
        Float_ threshold
    ) {
        auto candidate = fill_labels_in_use(scores, threshold, my_labels_in_use);

        // If there's only one top label, we don't need to do anything else.
        // We also give up if every label is in range, because any subsequent
        // calculations would use all markers and just give the same result.
        while (my_labels_in_use.size() > 1 && my_labels_in_use.size() < scores.size()) {
            my_gene_subset.clear();
            for (auto l : my_labels_in_use) {
                for (auto l2 : my_labels_in_use){ 
                    for (auto c : markers[l][l2]) {
                        my_gene_subset.add(c);
                    }
                }
            }
            my_gene_subset.remap(input.begin(), input.end(), my_subset_query);
            const auto num_markers = my_gene_subset.size();

            if constexpr(!query_sparse_) {
                my_scaled_query.resize(num_markers);
            }
            if constexpr(!ref_sparse_) {
                my_scaled_ref.resize(num_markers);
            }
            scaled_ranks(num_markers, my_subset_query, my_scaled_query);

            scores.clear();
            auto nlabels_used = my_labels_in_use.size();
            for (I<decltype(nlabels_used)> i = 0; i < nlabels_used; ++i) {
                auto curlab = my_labels_in_use[i];

                my_all_correlations.clear();
                const auto& curref = ref[curlab];
                const auto NC = get_num_samples(curref);

                for (I<decltype(NC)> c = 0; c < NC; ++c) {
                    // Technically we could be faster if we remembered the
                    // subset from the previous fine-tuning iteration, but this
                    // requires us to (possibly) make a copy of the entire
                    // reference set; we can't afford to do this in each thread.

                    std::size_t refstart, refend;
                    if constexpr(ref_sparse_) {
                        refstart = curref.indptrs[c];
                        refend = curref.indptrs[c + 1];
                    } else {
                        refstart = sanisizer::product_unsafe<std::size_t>(num_markers, c);
                        refend = refstart + num_markers;
                    }
                    my_gene_subset.remap(curref.all_ranked.begin() + refstart, curref.all_ranked.begin() + refend, my_subset_ref);
                    scaled_ranks(num_markers, my_subset_ref, my_scaled_ref);

                    const Float_ r2 = compute_l2(num_markers, my_scaled_query, my_scaled_ref);
                    const Float_ cor = 1 - 2 * r2;
                    my_all_correlations.push_back(cor);
                }

                Float_ score = correlations_to_score(my_all_correlations, quantile);
                scores.push_back(score);
            }

            candidate = update_labels_in_use(scores, threshold, my_labels_in_use); 
        }

        return candidate;
    }
};

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

template<bool query_sparse_, bool ref_sparse_, typename Value_, typename Index_, typename Float_, typename Label_>
void annotate_cells_single_raw(
    const tatami::Matrix<Value_, Index_>& test,
    const std::vector<Index_>& subset,
    const BuiltReference<Index_, Float_>& built,
    const Markers<Index_>& markers,
    Float_ quantile,
    bool fine_tune,
    Float_ threshold,
    Label_* best, 
    const std::vector<Float_*>& scores,
    Float_* delta,
    int num_threads
) {
    assert(sanisizer::is_equal(built.num_markers, subset.size()));
    const auto num_markers = built.num_markers;
    const auto& ref = get_per_label_references<ref_sparse_>(built);

    // Figuring out how many neighbors to keep and how to compute the quantiles.
    const auto num_labels = ref.size();
    std::vector<Index_> search_k(num_labels);
    std::vector<std::pair<Float_, Float_> > coeffs(num_labels);

    for (I<decltype(num_labels)> r = 0; r < num_labels; ++r) {
        const Float_ denom = get_num_samples(ref[r]) - 1;
        const Float_  prod = denom * (1 - quantile);
        auto k = std::ceil(prod) + 1;
        search_k[r] = k;

        // `(1 - quantile) - (k - 2) / denom` represents the gap to the smaller quantile,
        // while `(k - 1) / denom - (1 - quantile)` represents the gap from the larger quantile.
        // The size of the gap is used as the weight for the _other_ quantile, i.e., 
        // the closer you are to a quantile, the higher the weight.
        // We convert these into proportions by dividing by their sum, i.e., `1/denom`.
        coeffs[r].first = static_cast<Float_>(k - 1) - prod;
        coeffs[r].second = prod - static_cast<Float_>(k - 2);
    }

    SubsetSanitizer<query_sparse_, Index_> subsorted(subset);
    tatami::VectorPtr<Index_> subset_ptr(tatami::VectorPtr<Index_>{}, &(subsorted.extraction_subset()));

    tatami::parallelize([&](int, Index_ start, Index_ length) {
        auto ext = tatami::consecutive_extractor<query_sparse_>(&test, false, start, length, subset_ptr);

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

        FineTuneSingle<query_sparse_, ref_sparse_, Label_, Index_, Float_, Value_> ft(num_markers, ref);
        std::vector<Float_> curscores(num_labels);

        for (Index_ c = start, end = start + length; c < end; ++c) {
            const auto info = [&](){
                if constexpr(query_sparse_) {
                    return ext->fetch(vbuffer.data(), ibuffer.data());
                } else {
                    return ext->fetch(vbuffer.data());
                }
            }();
            subsorted.fill_ranks(info, vec);
            scaled_ranks(num_markers, vec, query_scaled); 

            curscores.resize(num_labels);
            for (I<decltype(num_labels)> r = 0; r < num_labels; ++r) {
                const auto k = search_k[r];
                find_closest_neighbors<query_sparse_, ref_sparse_>(num_markers, query_scaled, k, ref[r], workspaces[r]);

                const Float_ last_squared = pop_furthest_neighbor(workspaces[r]).first;
                const Float_ last = 1 - 2 * last_squared;
                if (k == 1) {
                    curscores[r] = last;
                } else {
                    const Float_ next_squared = pop_furthest_neighbor(workspaces[r]).first;
                    const Float_ next = 1 - 2 * next_squared;
                    curscores[r] = coeffs[r].first * next + coeffs[r].second * last;
                }

                if (scores[r]) {
                    scores[r][c] = curscores[r];
                }
            }

            std::pair<Label_, Float_> chosen;
            if (!fine_tune) {
                chosen = find_best_and_delta<Label_>(curscores);
            } else {
                chosen = ft.run(vec, ref, markers, curscores, quantile, threshold);
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
    const std::vector<Index_>& subset,
    const BuiltReference<Index_, Float_>& ref,
    const Markers<Index_>& markers,
    Float_ quantile,
    bool fine_tune,
    Float_ threshold,
    Label_* best, 
    const std::vector<Float_*>& scores,
    Float_* delta,
    int num_threads
) {
    const auto ref_sparse = ref.sparse.has_value();
    if (test.is_sparse()) {
        if (ref_sparse) {
            annotate_cells_single_raw<true, true>(test, subset, ref, markers, quantile, fine_tune, threshold, best, scores, delta, num_threads);
        } else {
            annotate_cells_single_raw<true, false>(test, subset, ref, markers, quantile, fine_tune, threshold, best, scores, delta, num_threads);
        }
    } else {
        if (ref_sparse) {
            annotate_cells_single_raw<false, true>(test, subset, ref, markers, quantile, fine_tune, threshold, best, scores, delta, num_threads);
        } else {
            annotate_cells_single_raw<false, false>(test, subset, ref, markers, quantile, fine_tune, threshold, best, scores, delta, num_threads);
        }
    }
}

}

#endif
