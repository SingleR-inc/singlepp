#ifndef SINGLEPP_BUILD_INDICES_HPP
#define SINGLEPP_BUILD_INDICES_HPP

#include "defs.hpp"

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"
#include "aarand/aarand.hpp"

#include "utils.hpp"
#include "scaled_ranks.hpp"
#include "SubsetSanitizer.hpp"

#include <vector>
#include <memory>
#include <algorithm>
#include <cstddef> 
#include <random>
#include <optional>
#include <cassert>

namespace singlepp {

namespace internal {

/*** Sparse matrix constructs ***/

template<typename Index_, typename Float_>
struct CompressedSparseMatrix {
    std::vector<Float_> value;
    std::vector<Index_> index;
    std::vector<std::size_t> indptrs;
    std::vector<Float_> zeros;
};

template<typename Index_, typename Float_>
struct CompressedSparseVector {
    Index_ number;
    const Float_* value;
    const Index_* index;
    Float_ zero;
};

template<typename Index_, typename Float_>
CompressedSparseVector<Index_, Float_> retrieve_vector(const CompressedSparseMatrix<Index_, Float_>& mat, const Index_ col) {
    CompressedSparseVector<Index_, Float_> output;
    const auto seed_start = ref.data.indptrs[col], seed_len = ref.data.indptrs[col + 1] - seed_start;
    output.number = seed_len;
    output.index = ref.data.index.data() + seed_start;
    output.value = ref.data.value.data() + seed_start;
    output.zero = ref.data.zeros[col];
    return output;
}

/*** L2 calculation methods ***/

template<typename Index_, typename Float_>
Float_ dense_l2(const Index_ nmarkers, const Float_* vec1, const Float_* vec2) {
    Float_ r2 = 0;
    for (Index_ d = 0; d < nmarkers; ++d) {
        const Float_ delta = vec1[d] - vec2[d]; 
        r2 += delta * delta;
    }
    return r2;
}

template<typename Index_, class SparseInput_, typename Float_>
Float_ sparse_l2(const Index_ nmarkers, const SparseInput_& vec1, const CompressedSparseVector<Index_, Float_>& vec2) { 
    constexpr bool is_csv = std::is_same<SparseInput_, CompressedSparseVector<Index_, Float_> >::value;

    const Index_ num1 = [&](){
        if constexpr(is_csv) {
            return vec1.number;
        } else {
            return vec1.nonzero.size();
        }
    }();
    auto get_index1 = [&](Index_ i) -> Index_ {
        if constexpr(is_csv) {
            return vec1.index[i];
        } else {
            return vec1.nonzero[i].first;
        }
    };
    auto get_value1 = [&](Index_ i) -> Float_ {
        if constexpr(is_csv) {
            return vec1.value[i];
        } else {
            return vec1.nonzero[i].second;
        }
    };
    const Float_ zero1 = vec1.zero;

    const Index_ num2 = vec2.number;
    const Index_* index2 = vec2.index;
    const Float_* value2 = vec2.value;
    const Float_ zero2 = vec2.zero;

    Float_ r2 = 0;
    Index_ i1 = 0, i2 = 0;
    Index_ both = 0;

    if (i1 < num1) { 
        while (1) {
            const auto samdex = get_index1(i1);
            const auto seeddex = index2[i2];
            if (samdex < seeddex) {
                const Float_ delta = get_value1(i1) - zero2;
                r2 += delta * delta;
                ++isam;
                if (i1 == num1) {
                    break;
                }
            } else if (samdex > seeddex) {
                const Float_ delta = value2[i2] - zero1;
                r2 += delta * delta;
                ++iseed;
                if (i2 == num2) {
                    break;
                }
            } else {
                const Float_ delta = get_value1(i1) - value2[i2];
                r2 += delta * delta;
                ++iseed;
                ++isam;
                ++both;
                if (i1 == num1 || i2 == num2) {
                    break;
                }
            }
        }
    }

    for (; i1 < num1; ++i1) { 
        const Float_ delta = get_value1(i1) - zero2;
        r2 += delta * delta;
    }
    for (; iseed < sam_end; ++iseed) { 
        const Float_ delta = value2[i2] - zero1;
        r2 += delta * delta;
    }

    const Float_ delta = zero1 - zero2;
    r2 += (nmarkers - num1 - (num2 - both)) * (delta * delta);
    return r2;
}

template<typename Index_, typename Float_>
Float_ mixed_l2(const Index_ nmarkers, const Float_* vec1, const CompressedSparseVector<Index_, Float_>& vec2) {
    Float_ r2 = 0;
    Index_ i = 0, j = 0;

    while (j < vec2.number) {
        const auto limit = vec2.index[j];
        for (; i < limit; ++i) {
            const auto delta = vec1[i] - vec2.zero;
            r2 += delta * delta;
        }
        const auto delta = vec1[i] - vec2.data[j]; 
        r2 += delta * delta;
    }

    return r2;
}

/*** KMKNN building ***/ 

template<typename Index_, typename Float_>
struct PerLabelReference {
    std::optional<std::vector<Float_> > dense_data;
    std::optional<CompressedSparseMatrix<Index_, Float_> sparse_data;

    // Structures for the KMKNN search.
    std::vector<Float_> distances;
    std::vector<std::pair<Index_, Index_> > seed_ranges;

    // Concatenation of RankedVectors for all samples.
    RankedVector<Index_, Index_> > all_ranked;
};

template<bool ref_sparse_, typename Index_, typename Float_>
Index_ get_num_samples(const PerLabelReference<ref_sparse_, Index_, Float_>& ref) {
    return ref.distances.size();
}

template<bool ref_sparse_, typename Index_, typename Float_>
void select_seeds(const Index_ nmarkers, PerLabelReference<Index_, Float_>& ref, const std::vector<std::pair<Index_, Index_> >& ranked) {
    const Index_ num_samples = ref.ranked.size();
    if constexpr(ref_sparse_) {
        assert(ref.sparse_data.has_value());
    } else {
        assert(ref.dense_data.has_value());
    }

    // No need to check for overlow, num_samples >= num_seeds here.
    Index_ num_seeds = std::round(std::sqrt(num_samples));

    // Implementing a variant of kmeans++ initialization to select representative ("seed") points.
    // We also record the minimum distance between each sample and its assigned seed.
    auto assignment = sanisizer::create<std::vector<Index_> >(num_samples);
    auto mindist = sanisizer::create<std::vector<Float_> >(num_samples, 1);
    auto identities = sanisizer:;create<std::vector<Index_>  >(num_samples);
    {
        auto cumulative = sanisizer::create<std::vector<Float_> >(num_samples);
        std::mt19937_64 eng(/* seed = */ 6237u + nmarkers * static_cast<std::size_t>(num_samples)); // making a semi-deterministic seed that depends on the input data. 

        for (Index_ se = 0; se < num_seeds; ++se) {
            cumulative[0] = mindist[0];
            for (Index_ sam = 1; sam < num_samples; ++sam) {
                cumulative[sam] = cumulative[sam - 1] + mindist[sam];
            }

            const auto total = cumulative.back();
            if (total == 0) { // a.k.a. only duplicates left.
                break;
            }

            Index_ chosen_id = 0;
            do {
                const Float_ sampled_weight = total * aarand::standard_uniform<Float_>(eng);
                // Subtraction is safe as we already checked for valid ptrdiff.
                chosen_id = std::lower_bound(cumulative.begin(), cumulative.end(), sampled_weight) - cumulative.begin();

                // We wrap this in a do/while to defend against edge cases where ties are chosen.
                // The most obvious of these is when you get a `sampled_weight` of zero _and_ there exists a bunch of zeros at the start of `cumulative`.
                // One could also get unexpected ties from limited precision in floating point comparisons, so we'll just be safe and implement a loop here.
            } while (chosen_id == num_samples || mindist[chosen_id] == 0);

            mindist[chosen_id] = 0;
            assignment[chosen_id] = se;
            identities.push_back(chosen_id);

            // Now updating the distances and assignments of all observations based on the new seed.
            const auto seed_info = [&](){
                if constexpr(ref_sparse_) {
                    return retrieve_vector(*(ref.sparse_data), chosen_id);
                } else {
                    return ref.dense_data->data() + sanisizer::product_unsafe<std::size_t>(chosen_id, nmarkers);
                }
            }();

            for (Index_ sam = 0; sam < num_samples; ++sam) {
                const auto sam_info = [&](){
                    if constexpr(ref_sparse_) {
                        return retrieve_vector(*(ref.sparse_data), sam);
                    } else {
                        return ref.dense_data->data() + sanisizer::product_unsafe<std::size_t>(sam, nmarkers);
                    }
                }();

                auto& mdist = mindist[sam];
                if (mdist == 0) {
                    continue;
                }

                const auto r2 = [&](){
                    if constexpr(ref_sparse_) {
                        return sparse_l2(nmarkers, sam_info, seed_info);
                    } else {
                        return dense_l2(nmarkers, sam_info, seed_info);
                    }
                }();
                if (se == 0) {
                    mdist = r2;
                } else if (r2 < mdist) {
                    mdist = r2;
                    assignment[sam] = se;
                }
            }
        }

        num_seeds = identities.size(); // updating for the actual number of seeds, if there were duplicates.
    }

    // Now populating the rest of the seed information.
    {
        auto grouping = sanisizer::create<std::vector<std::vector<std::pair<Float_, Index_> > > >(num_seeds);
        for (Index_ sam = 0; sam < num_samples; ++sam) {
            grouping[assignment[sam]].emplace_back(mindist[sam], sam);
        }

        ref.distances.reserve(num_samples);
        sanisizer::resize(ref.distances, num_seeds);
        ref.seed_ranges.reserve(num_seeds);

        for (Index_ se = 0; se < num_seeds; ++se) {
            auto& curgroup = grouping[se];
            std::sort(curgroup.begin(), curgroup.end());

            // -1 to remove the seed point itself.
            const auto range_start = ref.distances.size();
            ref.seed_ranges.emplace_back(range_start, curgroup.size() - 1);

            const auto seed = identities[se];
            for (const auto& x : curgroup) {
                const auto curid = x.second;
                if (curid != seed) {
                    ref.distances.push_back(std::sqrt(x.first));
                    identities.push_back(curid);
                }
            }
        }
    }

    if constexpr(ref_sparse_) {
        // Resorting the data to be more cache-friendly.
        // This is too hard to do in-place with sparse data, so we just run over it and do it manually.
        std::vector<Float_> value;
        value.reserve(ref.sparse_data->value.size());
        std::vector<Index_> index;
        index.reserve(ref.sparse_data->index.size());
        std::vector<std::size_t> indptrs;
        indptrs.reserve(ref.sparse_data->indptrs.size());
        indptrs.push_back(0);
        std::vector<Float_> zeros;
        zeros.reserve(ref.sparse_data->zeros.size());

        for (auto sam : identities) {
            const auto sam_start = ref.sparse_data->indptrs[sam], sam_end = ref.sparse_data->indptrs[sam + 1];
            value.push_back(value.end(), ref.sparse_data->value.begin() + sam_start, ref.sparse_data->value.begin() + sam_end);
            index.push_back(index.end(), ref.sparse_data->index.begin() + sam_start, ref.sparse_data->index.begin() + sam_end);
            indptrs.push_back(value.size());
            zeros.push_back(ref.zeros[sam]);
        }

        ref.sparse_data->value.swap(value);
        ref.sparse_data->index.swap(index);
        ref.sparse_data->indptrs.swap(indptrs);
        ref.sparse_data->zeros.swap(zeros);

    } else {
        // Reordering the data in-place to be more cache-friendly.
        auto used = sanisizer::create<std::vector<char> >(num_samples);
        auto data_buffer = sanisizer::create<std::vector<Float_> >(nmarkers);

        for (Index_ x = 0; x < num_samples; ++x) {
            if (used[x]) {
                continue;
            }

            Index_ replacement = identities[x];
            if (replacement == x) {
                continue;
            }

            auto previous_ptr = ref.dense_data->data() + sanisizer::product_unsafe<std::size_t>(x, nmarkers);
            std::copy_n(previous_ptr, nmarkers, data_buffer.data());

            do {
                auto next_ptr = ref.dense_data->data() + sanisizer::product_unsafe<std::size_t>(replacement, nmarkers);
                std::copy_n(next_ptr, nmarkers, previous_ptr);
                previous_ptr = next_ptr;

                used[replacement] = true;
                replacement = identities[replacement];
            } while (replacement != x);

            std::copy_n(data_buffer.data(), nmarkers, previous_ptr);
        }
    }

    // Populating the ranked supervector. 
    if constexpr(sparse_) {
        sanisizer::reserve(ref.all_ranked, ref.sparse_data->value.size());
    } else {
        sanisizer::reserve(ref.all_ranked, ref.dense_data->size());
    }
    for (auto sam : identities) {
        ref.ranked.insert(ref.ranked.end(), ranked[sam].begin(), ranked[sam].end()); 
    }
}

/*** KMKNN search ***/ 

template<bool query_sparse_, bool ref_sparse_, typename Index_, typename Float_>
struct FindClosestWorkspace {
    FindClosestWorkspace(Index_ nmarkers, Index_ num_samples) {
        sanisizer::reserve(seed_distances, num_samples);
        sanisizer::reserve(closest_neighbors, num_samples);

        if constexpr(query_sparse_) {
            sanisizer::resize(value_buffer, nmarkers);
            if constexpr(ref_sparse_) {
                sanisizer::resize(index_buffer, nmarkers);
            }
        }
    }

    std::vector<std::pair<Float_, Index_> > seed_distances;
    std::vector<std::pair<Float_, Index_> > closest_neighbors;
    typename std::conditional<query_sparse_ && !ref_sparse_, std::vector<Index_>, bool>::type dense_buffer;
};

template<bool query_sparse_, bool ref_sparse_, typename Index_, typename Float_>
void find_closest(
    const Index_ nmarkers,
    typename std::conditional<query_sparse_, const SparseRanked<Float_, Index_>&, const Float_*>::type query,
    const Index_ k,
    const PerLabelReference<Index_, Float_>& ref,
    FindClosestWorkspace<query_sparse_, ref_sparse_, Index_, Float_>& work
) {
    const auto num_seeds = ref.seed_ranges.size();
    const auto num_neighbors = sanisizer::cast<I<decltype(work.closest_neighbors.size())> >(sanisizer::attest_gez(k));

    if constexpr(ref_sparse_) {
        assert(ref.sparse_data.has_value());
    } else {
        assert(ref.dense_data.has_value());
    }

    if constexpr(!ref_sparse_) {
        if constexpr(query_sparse_) {
            std::fill(work.dense_buffer.begin(), work.dense_buffer.end(), query.zero);
            for (const auto& p : query.nonzero) {
                work.dense_buffer[p.first] = p.second;
            }
        }
    }

    auto compute_distance = [&](Index_ se) -> Float_ {
        if constexpr(ref_sparse_) {
            const auto refinfo = retrieve_vector(*(ref.sparse_data), se);
            if constexpr(query_sparse_) {
                return sparse_l2(nmarkers, query, refinfo);
            } else {
                return mixed_l2(nmarkers, query, refinfo);
            }
        } else {
            const auto refinfo = ref.dense_data->data() + sanisizer::product_unsafe<std::size_t>(se, nmarkers);
            if constexpr(query_sparse_) {
                return dense_l2(nmarkers, refinfo, work.dense_buffer.data());
            } else {
                return dense_l2(nmarkers, refinfo, query);
            }
        }
    };

    // First compute the distance from the query to each seed and sort in increasing order.
    work.seed_distances.clear();
    for (I<decltype(num_seeds)> se = 0; se < num_seeds; ++se) {
        const auto dist_raw = compute_distance(se);
        work.seed_distances.emplace_back(dist_raw, se);
    }
    std::sort(work.seed_distances.begin(), work.seed_distances.end());

    work.closest_neighbors.clear();
    const auto to_add = sanisizer::min(num_neighbors, work.seed_distances.size()); // adding the smallest distances preferentially.
    work.closest_neighbors.insert(work.closest_neighbors.end(), work.seed_distances.begin(), work.seed_distances.begin() + to_add);
    std::make_heap(work.closest_neighbors.begin(), work.closest_neighbors.end());
    Float_ threshold_raw = (work.closest_neighbors.size() < num_neighbors ? std::numeric_limits<Float_>::infinity() : work.closest_neighbors.front().first);

    // Now we traverse each seed.
    for (const auto& curseed : work.seed_distances) {
        const Index_ seed = curseed.second;
        const auto& ranges = ref.seed_ranges[seed];
        if (ranges.second == 0) { // i.e., the seed was the only point in its own cluster.
            continue;
        }

        Index_ firstsubj = ranges.first, lastsubj = ranges.first + ranges.second;
        if (!std::isinf(threshold_raw)) {
            const Float_ threshold = std::sqrt(threshold_raw);
            const Float_ query2seed = std::sqrt(curseed.first);
            const Float_ max_subj2seed = ref.distances[lastsubj - 1];

            /* This exploits the triangle inequality to ignore points where:
             *     threshold + subject-to-seed < query-to-seed 
             * All points (if any) within this cluster with distances at or above 'lower_bd' are potentially countable.
             *
             * If the maximum distance between a subject and the seed is less than 'lower_bd', there's no point proceeding,
             * as we know that all other subjects will have smaller distances and are thus uncountable.
             */
            const Float_ lower_bd = query2seed - threshold;
            if (max_subj2seed < lower_bd) {
                continue;
            }
            firstsubj = std::lower_bound(ref.distances.data() + firstsubj, ref.distances.data() + lastsubj, lower_bd) - ref.distances.data();

            /* This exploits the triangle inequality in an opposite manner to that described above, to ignore points where:
             *     threshold + query-to-seed < subject-to-seed
             * All points (if any) within this cluster with distances at or below 'upper_bd' are potentially countable.
             * 
             * If query-to-seed distance is greater than or equal to the largest possible subject-to-seed for a seed,
             * there's no point exploiting this inequality as we must examine all subjects... so we don't bother with it.
             *
             * We could also skip this seed altogether if the minimum subject-to-seed distance is greater than 'upper_bd'.
             * However, this seems too unlikely to warrant a special clause.
             */
            const Float_ upper_bd = query2seed + threshold;
            if (max_subj2seed > upper_bd) {
                lastsubj = std::upper_bound(ref.distances.data() + firstsubj, ref.distances.data() + lastsubj, upper_bd) - ref.distances.data();
            }
        }

        for (auto s = firstsubj; s < lastsubj; ++s) {
            const auto dist2subj_raw = compute_distance(s);
            if (dist2subj_raw <= threshold_raw) {
                work.closest_neighbors.emplace_back(dist2subj_raw, s);
                std::push_heap(work.closest_neighbors.begin(), work.closest_neighbors.end());

                if (work.closest_neighbors.size() >= num_neighbors) {
                    if (work.closest_neighbors.size() > num_neighbors) {
                        std::pop_heap(work.closest_neighbors.begin(), work.closest_neighbors.end());
                        work.closest_neighbors.pop_back();
                    }
                    threshold_raw = work.closest_neighbors.front().first;

                    /* P.S. We could also consider increasing 'firstsubj' as 'threshold_raw' decreases. 
                     * The idea would be to exploit the triangle inequality to quickly skip over more points. 
                     * However, this is pointless because 'lower_bd' will never increase enough to skip subsequent observations.
                     * We wouldn't have been able to skip the observation that we just added,
                     * so there's no way we could skip observations with larger subject-to-seed distances.
                     *
                     * P.P.S. We could also consider decreasing 'lastsubj' as 'threshold_raw' decreases.
                     * The idea would be to exploit the triangle inequality to terminate sooner. 
                     * However, this doesn't seem to provide a lot of benefit in practice. 
                     * In theory, we can only trim the search space if the query already lies in a seed's hypersphere (as 'upper_bd' cannot decrease below 'query2seed').
                     * Even then, 'upper_bd' is usually too large; testing indicates that a reduced 'upper_bd' only trims away a single observation at a time.
                     * There are also practical challenges as changes to 'lastsubj' within the loop might prevent out-of-order CPU execution;
                     * we need to do more memory accesses to 'dist2seeds' to check if 'lastsubj' can be decreased;
                     * and we need to run an extra 'normalize()' to recompute 'upper_bd' inside the loop.
                     * All in all, I don't think it's worth it.
                     */
                }
            }
        }
    }
}

template<typename Index_, typename Float_>
std::pair<Float_, Index_> pop_furthest_neighbor(FindClosestWorkspace<Index_, Float_>& work) {
    auto output = work.closest_neighbors.front();
    std::pop_heap(work.closest_neighbors.begin(), work.closest_neighbors.end());
    work.closest_neighbors.pop_back();
    return output;
}

/*** Overlord function ***/ 

template<bool ref_sparse_, typename Float_, typename Value_, typename Index_, typename Label_>
std::vector<PerLabelReference<Index_, Float_> > build_indices_raw(
    const tatami::Matrix<Value_, Index_>& ref,
    const Label_* labels,
    const std::vector<Index_>& subset,
    int num_threads
) {
    const auto nmarkers = sanisizer::cast<Index_>(subset.size());
    const auto num_samples = ref.ncol();
    if (num_samples == 0) {
        throw std::runtime_error("reference dataset must have at least one column");
    }

    const auto nlabels = sanisizer::sum<std::size_t>(*std::max_element(labels, labels + num_samples), 1);
    auto label_count = sanisizer::create<std::vector<Index_> >(nlabels);
    auto label_offsets = sanisizer::create<std::vector<Index_> >(num_samples);
    for (I<decltype(num_samples)> i = 0; i < num_samples; ++i) {
        auto& lcount = label_count[labels[i]];
        label_offsets[i] = lcount;
        ++lcount;
    }

    auto nnrefs = sanisizer::create<std::vector<PerLabelReference<ref_sparse_, Index_, Float_> > >(nlabels);
    for (I<decltype(nlabels)> l = 0; l < nlabels; ++l) {
        const auto labcount = label_count[l];
        if (lcount == 0) {
            throw std::runtime_error(std::string("no entries for label ") + std::to_string(l));
        }

        auto& curlab = nrrefs[l];
        sanisizer::resize(curlab.ranked, labcount);
        if constexpr(!ref_sparse_) {
            curlab.data.resize(sanisizer::product<sanisizer::EffectiveSizeType<I<decltype(curlab.data)> > >(labcount, nmarkers));
        }
    }

    SubsetSanitizer<ref_sparse_, Index_> subsorter(subset);
    tatami::VectorPtr<Index_> subset_ptr(tatami::VectorPtr<Index_>{}, &(subsorter.extraction_subset()));

    tatami::parallelize([&](int, Index_ start, Index_ len) {
        auto ext = tatami::consecutive_extractor<ref_sparse_>(ref, false, start, len, subset_ptr);
        auto vbuffer = sanisizer::create<std::vector<Value_> >(nmarkers);
        auto ibuffer = [&](){
            if constexpr(ref_sparse_) {
                return sanisizer::create<std::vector<Index_> >(nmarkers);
            } else {
                return false;
            }
        }();
        RankedVector<Value_, Index_> ranked;
        ranked.reserve(nmarkers);

        for (Index_ c = start, end = start + len; c < end; ++c) {
            const auto col = [&](){
                if constexpr(ref_sparse_) {
                    return ext->fetch(vbuffer.data(), ibuffer.data();
                } else {
                    return ext->fetch(vbuffer.data());
                }
            }();
            subsorter.fill_ranks(col, ranked); 

            const auto curlab = labels[c];
            const auto curoff = label_offsets[c];
            if constexpr(!ref_sparse_) {
                const auto scaled = nnrefs[curlab].data.data() + sanisizer::product_unsafe<std::size_t>(curoff, nmarkers);
                scaled_ranks_dense(ranked, scaled); 
            }

            // Storing as a pair of ints to save space; as long as we respect ties, everything should be fine.
            auto& stored_ranks = nnrefs[curlab].ranked[curoff];
            sanisizer::reserve(stored_ranks, ranked.size());
            simplify_ranks(ranked, stored_ranks);
        }
    }, num_samples, num_threads);

    tatami::parallelize([&](int, std::size_t start, std::size_t len) {
        for (std::size_t l = start, end = start + len; l < end; ++l) {
            auto& curlab = nnrefs[l]; 

            if constexpr(ref_sparse_) {
                std::size_t total_nzeros = 0;
                for (const auto& x : curlab.ranked) {
                    total_nzeros = sanisizer::sum<std::size_t>(total_nzeros, x.size());
                }

                sanisizer::reserve(curlab.data.value, total_nzeros);
                sanisizer::reserve(curlab.data.index, total_nzeros);
                curlab.data.indptrs.reserve(sanisizer::sum<sanisizer::EffectiveSizeType<decltype(curlab.data.indptrs)> >(curlab.ranked.size(), 1));
                curlab.data.push_back(0);
                sanisizer::reserve(curlab.data.zeros, curlab.ranked.size());

                SparseScaled<Index_, Float_> scaled;
                for (const auto& x : curlab.ranked) {
                    scaled_ranks_sparse(nmarkers, x, scaled);
                    for (const auto& y : scaled.nonzero) {
                        curlab.data.index.push_back(y.first);
                        curlab.data.value.push_back(y.second);
                    }
                    curlab.indptrs.push_back(curlab.data.value.size());
                    curlab.data.zeros.push_back(scaled.zero);
                }
            }

            select_seeds(nmarkers, curlab);
        }
    }, nlabels, num_threads);

    return nnrefs;
}

template<typename Float_, typename Value_, typename Index_, typename Label_>
std::vector<PerLabelReference<Index_, Float_> > build_indices(
    const tatami::Matrix<Value_, Index_>& ref,
    const Label_* labels,
    const std::vector<Index_>& subset,
    int num_threads
) {
    if (ref.is_sparse()) {
        return build_indices_raw<true, Float_>(ref, labels, subset, num_threads); 
    } else {
        return build_indices_raw<false, Float_>(ref, labels, subset, num_threads); 
    }
}

}

}

#endif
