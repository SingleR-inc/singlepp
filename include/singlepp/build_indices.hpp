#ifndef SINGLEPP_BUILD_INDICES_HPP
#define SINGLEPP_BUILD_INDICES_HPP

#include "defs.hpp"

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"
#include "aarand/aarand.hpp"

#include "scaled_ranks.hpp"
#include "SubsetSanitizer.hpp"

#include <vector>
#include <memory>
#include <algorithm>
#include <cstddef> 
#include <random>
#include <optional>

namespace singlepp {

namespace internal {

template<typename Input_>
using I = std::remove_cv_t<std::remove_reference_t<Input_> >;

template<typename Index_, typename Float_>
struct PerLabelReference {
    std::size_t num_dim;
    std::vector<Float_> data;
    std::vector<Float_> distances;
    std::vector<std::pair<Index_, Index_> > seed_ranges;
    std::vector<RankedVector<Index_, Index_> > ranked;
};

template<typename Index_, typename Float_>
Index_ get_num_samples(const PerLabelReference<Index_, Float_>& ref) {
    return ref.distances.size();
}

template<typename Index_, typename Float_>
void select_seeds(const std::size_t num_dim, const Index_ num_samples, PerLabelReference<Index_, Float_>& ref) {
    // No need to check for overlow, num_samples >= num_seeds here.
    Index_ num_seeds = std::round(std::sqrt(num_samples));

    // Implementing a variant of kmeans++ initialization to select representative ("seed") points.
    // We also record the minimum distance between each sample and its assigned seed.
    auto assignment = sanisizer::create<std::vector<Index_> >(num_samples);
    auto mindist = sanisizer::create<std::vector<Float_> >(num_samples, 1);
    std::vector<Index_> identities;
    {
        auto cumulative = sanisizer::create<std::vector<Float_> >(num_samples);
        sanisizer::can_ptrdiff<I<decltype(cumulative.begin())> >(num_samples); // check that we can compute a ptrdiff for computing a weighted sample.
 
        std::mt19937_64 eng(/* seed = */ 6237u + ref.data.size()); // making a semi-deterministic seed that depends on the input data. 
        identities.reserve(num_samples);

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
            const auto seed_ptr = ref.data.data() + sanisizer::product_unsafe<std::size_t>(identities.back(), num_dim);
            auto compute_d2 = [&](const Float_* current) -> Float_ {
                Float_ r2 = 0;
                for (std::size_t d = 0; d < num_dim; ++d) {
                    const Float_ delta = current[d] - seed_ptr[d]; 
                    r2 += delta * delta;
                }
                return r2;
            };

            if (se == 0) {
                // Always compute the minimum distance in the first round, except for the point that was selected as a seed.
                for (Index_ sam = 0; sam < num_samples; ++sam) {
                    const auto current = ref.data.data() + sanisizer::product_unsafe<std::size_t>(sam, num_dim);
                    if (mindist[sam] == 0) {
                        continue;
                    }
                    mindist[sam] = compute_d2(current);
                }
            } else {
                // See if we can get a lower minimum distance to the latest seed.
                for (Index_ sam = 0; sam < num_samples; ++sam) {
                    const auto current = ref.data.data() + sanisizer::product_unsafe<std::size_t>(sam, num_dim);
                    if (mindist[sam] == 0) {
                        continue;
                    }
                    const auto r2 = compute_d2(current);
                    if (r2 < mindist[sam]) {
                        mindist[sam] = r2;
                        assignment[sam] = se;
                    }
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

    // Reordering the data in-place to be more cache-friendly. 
    {
        auto data_buffer = sanisizer::create<std::vector<Float_> >(num_dim);
        auto used = sanisizer::create<std::vector<char> >(num_samples);

        for (Index_ x = 0; x < num_samples; ++x) {
            if (used[x]) {
                continue;
            }

            Index_ replacement = identities[x];
            if (replacement == x) {
                continue;
            }

            auto previous_ptr = ref.data.data() + sanisizer::product_unsafe<std::size_t>(x, num_dim);
            std::copy_n(previous_ptr, num_dim, data_buffer.data());
            auto previous_ranked = &(ref.ranked[x]);

            do {
                auto next_ptr = ref.data.data() + sanisizer::product_unsafe<std::size_t>(replacement, num_dim);
                std::copy_n(next_ptr, num_dim, previous_ptr);
                previous_ptr = next_ptr;

                auto& next_ranked = ref.ranked[replacement];
                previous_ranked->swap(next_ranked);
                previous_ranked = &next_ranked;

                used[replacement] = true;
                replacement = identities[replacement];
            } while (replacement != x);

            std::copy_n(data_buffer.data(), num_dim, previous_ptr);
        }
    }
}

template<typename Index_, typename Float_>
struct FindClosestWorkspace {
    FindClosestWorkspace(Index_ num_samples) {
        seed_distances.reserve(num_samples);
        closest_neighbors.reserve(num_samples);
    }

    std::vector<std::pair<Float_, Index_> > seed_distances;
    std::vector<std::pair<Float_, Index_> > closest_neighbors;
};

template<bool can_truncate_upper_bound_, typename Index_, typename Float_, typename NumNeighbors_>
void find_closest_for_seed(
    const Float_* const query,
    const NumNeighbors_ num_neighbors,
    typename std::conditional<can_truncate_upper_bound_, Float_, void*>::type query2seed,
    const Index_ firstsubj,
    Index_ lastsubj,
    const PerLabelReference<Index_, Float_>& ref,
    FindClosestWorkspace<Index_, Float_>& work,
    Float_& threshold_raw
) {
    const auto num_dim = ref.num_dim;

}

template<typename Index_, typename Float_>
void find_closest(
    const Float_* query,
    const Index_ k,
    const PerLabelReference<Index_, Float_>& ref,
    FindClosestWorkspace<Index_, Float_>& work
) {
    const auto num_dim = ref.num_dim;
    const auto num_seeds = ref.seed_ranges.size();
    const auto num_neighbors = sanisizer::cast<I<decltype(work.closest_neighbors.size())> >(sanisizer::attest_gez(k));

    // First compute the distance from the query to each seed and sort in increasing order.
    work.seed_distances.clear();
    for (I<decltype(num_seeds)> se = 0; se < num_seeds; ++se) {
        const auto seed_ptr = ref.data.data() + sanisizer::product_unsafe<std::size_t>(se, num_dim);
        Float_ dist_raw = 0;
        for (std::size_t d = 0; d < num_dim; ++d) {
            const Float_ delta = query[d] - seed_ptr[d]; 
            dist_raw += delta * delta;
        }
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
            const Float_ lower_bd = *query2seed - threshold;
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
            const Float_ upper_bd = *query2seed + threshold;
            if (max_subj2seed > upper_bd) {
                lastsubj = std::upper_bound(ref.distances.data() + firstsubj, ref.distances.data() + lastsubj, upper_bd) - ref.distances.data();
            }
        }

        for (auto s = firstsubj; s < lastsubj; ++s) {
            const auto subject = ref.data.data() + sanisizer::product_unsafe<std::size_t>(s, num_dim);
            Float_ dist2subj_raw = 0;
            for (std::size_t d = 0; d < num_dim; ++d) {
                const Float_ delta = query[d] - subject[d]; 
                dist2subj_raw += delta * delta;
            }

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

template<typename Float_, typename Value_, typename Index_, typename Label_>
std::vector<PerLabelReference<Index_, Float_> > build_indices(
    const tatami::Matrix<Value_, Index_>& ref,
    const Label_* labels,
    const std::vector<Index_>& subset,
    int num_threads
) {
    const auto num_dim = sanisizer::cast<std::size_t>(subset.size());
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

    auto nnrefs = sanisizer::create<std::vector<PerLabelReference<Index_, Float_> > >(nlabels);
    for (I<decltype(nlabels)> l = 0; l < nlabels; ++l) {
        if (label_count[l] == 0) {
            throw std::runtime_error(std::string("no entries for label ") + std::to_string(l));
        }
        sanisizer::resize(nnrefs[l].ranked, label_count[l]);
        nnrefs[l].num_dim = num_dim;
        nnrefs[l].data.resize(sanisizer::product<typename std::vector<Float_>::size_type>(label_count[l], num_dim));
    }

    SubsetSanitizer<Index_> subsorter(subset);
    tatami::VectorPtr<Index_> subset_ptr(tatami::VectorPtr<Index_>{}, &(subsorter.extraction_subset()));

    tatami::parallelize([&](int, Index_ start, Index_ len) {
        auto ext = tatami::consecutive_extractor<false>(ref, false, start, len, subset_ptr);
        auto buffer = sanisizer::create<std::vector<Value_> >(num_dim);
        RankedVector<Value_, Index_> ranked;
        ranked.reserve(num_dim);

        for (Index_ c = start, end = start + len; c < end; ++c) {
            const auto ptr = ext->fetch(buffer.data());
            subsorter.fill_ranks(ptr, ranked); 

            const auto curlab = labels[c];
            const auto curoff = label_offsets[c];
            const auto scaled = nnrefs[curlab].data.data() + sanisizer::product_unsafe<std::size_t>(curoff, num_dim);
            scaled_ranks(ranked, scaled); 

            // Storing as a pair of ints to save space; as long
            // as we respect ties, everything should be fine.
            auto& stored_ranks = nnrefs[curlab].ranked[curoff];
            stored_ranks.reserve(ranked.size());
            simplify_ranks(ranked, stored_ranks);
        }
    }, num_samples, num_threads);

    tatami::parallelize([&](int, decltype(nlabels) start, decltype(nlabels) len) {
        for (I<decltype(nlabels)> l = start, end = start + len; l < end; ++l) {
            select_seeds(num_dim, label_count[l], nnrefs[l]);
        }
    }, nlabels, num_threads);

    return nnrefs;
}

}

}

#endif
