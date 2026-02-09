#ifndef SINGLEPP_TRAIN_INTEGRATED_HPP
#define SINGLEPP_TRAIN_INTEGRATED_HPP

#include "defs.hpp"

#include "scaled_ranks.hpp"
#include "train_single.hpp"
#include "Intersection.hpp"

#include <vector>
#include <algorithm>
#include <cstdint>
#include <memory>

/**
 * @file train_integrated.hpp
 * @brief Prepare for integrated classification across references.
 */

namespace singlepp {

/**
 * @brief Input to `train_integrated()`.
 *
 * Instances of this class should not be manually created, but instead returned by `prepare_integrated_input()` and `prepare_integrated_input_intersect()`.
 *
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam RefLabel_ Integer type for the reference labels.
 */
template<typename Value_, typename Index_, typename RefLabel_>
struct TrainIntegratedInput {
    /**
     * @cond
     */
    const tatami::Matrix<Value_, Index_>* ref;
    const RefLabel_* labels;
    const Markers<Index_>* ref_markers;
    const std::vector<Index_>* test_subset;
    Index_ test_nrow;
    std::shared_ptr<const Intersection<Index_> > intersection;
    /**
     * @endcond
     */
};

/**
 * Prepare a reference dataset for `train_integrated()`.
 * This overload assumes that the reference and test datasets have the same genes.
 * All inputs are expected to remain valid until `train_integrated()` is called.
 *
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam RefLabel_ Integer type for the reference labels.
 * @tparam Float_ Floating-point type for the correlations and scores.
 *
 * @param ref Matrix containing the reference expression values, where rows are genes and columns are reference profiles.
 * The number and identity of genes should be identical to the test dataset to be classified in `classify_integrated()`.
 * @param[in] labels Pointer to an array of label assignments.
 * Values should be integers in \f$[0, L)\f$ where \f$L\f$ is the number of unique labels.
 * @param trained Classifier created by calling `train_single()` on `ref` and `labels`.
 *
 * @return An opaque input object for `train_integrated()`.
 */
template<typename Value_, typename Index_, typename RefLabel_, typename Float_>
TrainIntegratedInput<Value_, Index_, RefLabel_> prepare_integrated_input(
    const tatami::Matrix<Value_, Index_>& ref,
    const RefLabel_* labels, 
    const TrainedSingle<Index_, Float_>& trained
) {
    TrainIntegratedInput<Value_, Index_, RefLabel_> output;
    output.ref = &ref;
    output.labels = labels;

    output.ref_markers = &(trained.get_markers());
    output.test_subset = &(trained.get_subset());

    output.test_nrow = ref.nrow(); // remember, test and ref are assumed to have the same features.
    return output;
}

/**
 * Prepare a reference dataset for `train_integrated()`.
 * This overload requires an existing intersection between the test and reference datasets. 
 * All inputs are expected to remain valid until `train_integrated()` is called.
 *
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam RefLabel_ Integer type for the reference labels.
 * @tparam Float_ Floating-point type for the correlations and scores.
 *
 * @param test_nrow Number of features in the test dataset.
 * @param intersection Vector defining the intersection of genes between the test and reference datasets. 
 * Each pair corresponds to a gene where the first and second elements represent the row indices of that gene in the test and reference matrices, respectively.
 * The first element of each pair should be non-negative and less than `test_nrow`, while the second element should be non-negative and less than `ref->nrow()`.
 * See `intersect_genes()` for more details.
 * @param ref Matrix containing the reference expression values, where rows are genes and columns are reference profiles.
 * The number and identity of genes should be consistent with `intersection`.
 * @param[in] labels An array of length equal to the number of columns of `ref`, containing the label for each sample.
 * Values should be integers in \f$[0, L)\f$ where \f$L\f$ is the number of unique labels.
 * @param trained Classifier created by calling `train_single_intersect()` on `test_nrow`, `intersection`, `ref` and `labels`.
 *
 * @return An opaque input object for `train_integrated()`.
 */
template<typename Index_, typename Value_, typename RefLabel_, typename Float_>
TrainIntegratedInput<Value_, Index_, RefLabel_> prepare_integrated_input_intersect(
    Index_ test_nrow,
    const Intersection<Index_>& intersection,
    const tatami::Matrix<Value_, Index_>& ref, 
    const RefLabel_* labels, 
    const TrainedSingleIntersect<Index_, Float_>& trained) 
{
    TrainIntegratedInput<Value_, Index_, RefLabel_> output;
    output.ref = &ref;
    output.labels = labels;

    output.ref_markers = &(trained.get_markers());
    output.test_subset = &(trained.get_test_subset());

    output.test_nrow = test_nrow;
    output.intersection = std::shared_ptr<const Intersection<Index_> >(std::shared_ptr<Intersection<Index_> >{}, &intersection);
    return output;
}

/**
 * Prepare a reference dataset for `train_integrated()`.
 * This overload automatically identifies the intersection of genes between the test and reference datasets.
 * All inputs are expected to remain valid until `train_integrated()` is called.
 *
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam Id_ Type of the gene identifier. 
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam RefLabel_ Integer type for the reference labels.
 * @tparam Float_ Floating-point type for the correlations and scores.
 *
 * @param test_nrow Number of rows (i.e., genes) in the test dataset.
 * @param[in] test_id Pointer to an array of length equal to `test_nrow`.
 * This should contain a unique identifier for each row of `mat`, typically a gene name or index.
 * If any duplicate IDs are present, only the first occurrence is used.
 * @param ref An expression matrix for the reference expression profiles, where rows are genes and columns are cells.
 * This should have non-zero columns.
 * @param[in] ref_id Pointer to an array equal to the number of rows of `ref`.
 * This should contain a unique identifier for each row in `ref`, comparable to those in `test_id`.
 * If any duplicate IDs are present, only the first occurrence is used.
 * @param[in] labels An array of length equal to the number of columns of `ref`, containing the label for each sample.
 * Values should be integers in \f$[0, L)\f$ where \f$L\f$ is the number of unique labels.
 * @param trained Classifier created by calling `train_single_intersect()` on `test_nrow`, `test_id`, `ref`, `ref_id` and `labels`.
 *
 * @return An opaque input object for `train_integrated()`.
 */
template<typename Index_, typename Id_, typename Value_, typename RefLabel_, typename Float_>
TrainIntegratedInput<Value_, Index_, RefLabel_> prepare_integrated_input_intersect(
    Index_ test_nrow,
    const Id_* test_id, 
    const tatami::Matrix<Value_, Index_>& ref, 
    const Id_* ref_id, 
    const RefLabel_* labels,
    const TrainedSingleIntersect<Index_, Float_>& trained
) {
    TrainIntegratedInput<Value_, Index_, RefLabel_> output;
    output.ref = &ref;
    output.labels = labels;

    output.ref_markers = &(trained.get_markers());
    output.test_subset = &(trained.get_test_subset());

    output.test_nrow = test_nrow;
    auto intersection = intersect_genes(test_nrow, test_id, ref.nrow(), ref_id);
    output.intersection = std::shared_ptr<const Intersection<Index_> >(new Intersection<Index_>(std::move(intersection)));
    return output;
}

/**
 * @brief Classifier that integrates multiple reference datasets.
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 */
template<typename Index_>
class TrainedIntegrated {
public:
    /**
     * @return Number of reference datasets.
     */
    std::size_t num_references() const {
        return my_references.size();
    }

    /**
     * @param r Reference dataset of interest.
     * @return Number of labels in this reference.
     */
    std::size_t num_labels(std::size_t r) const {
        return my_references[r].markers.size();
    }

    /**
     * @param r Reference dataset of interest.
     * @return Number of profiles in this reference.
     */
    Index_ num_profiles(std::size_t r) const {
        return my_references[r].num_samples;
    }

public:
    /**
     * @cond
     */
    // Technically this should be private, but it's a pain to add
    // templated friend functions, so I can't be bothered.
    Index_ test_nrow;
    std::vector<Index_> universe; // To be used by classify_integrated() for indexed extraction.

    struct PerReference {
        Index_ num_samples;
        std::vector<std::vector<Index_> > markers; // indices to 'universe'
        RankedVector<Index_, Index_> all_ranked; // .second contains indices to 'universe'
        std::optional<std::vector<std::size_t> > all_ranked_indptrs; // only required if sparse.
    };
    std::vector<PerReference> my_references;
    /**
     * @endcond
     */
};

/**
 * @brief Options for `train_integrated()`. 
 */
struct TrainIntegratedOptions {
    /**
     * Number of threads to use.
     * The parallelization scheme is determined by `tatami::parallelize()`.
     */
    int num_threads = 1;
};

/**
 * @cond
 */
template<bool ref_sparse_, typename Value_, typename Index_, typename RefLabel_>
void train_integrated_per_reference_simple(
    const TrainIntegratedInput<Value_, RefRefLabel_, Index_>& input,
    const std::vector<Index_>& universe,
    const std::vector<Index_>& remap_to_universe,
    const TrainIntegratedOptions& options,
    const std::vector<Index_>& positions,
    std::vector<RankedVector<Index_, Index_> >& out_ranked 
) {
    // 'universe' technically refers to the row indices of the test matrix,
    // but in simple mode, the rows of the test and reference are the same, so we can use it directly here.
    tatami::VectorPtr<Index_> universe_ptr(tatami::VectorPtr<Index_>{}, &universe);

    tatami::parallelize([&](int, Index_ start, Index_ len) {
        const auto num_universe = output.universe.size();
        auto vbuffer = sanisizer::create<std::vector<Value_> >(num_universe);
        auto ibuffer = [&]() {
            if constexpr(ref_sparse_) {
                return sanisizer::create<std::vector<Index_> >(num_universe);
            } else {
                return false;
            }
        }();

        RankedVector<Value_, Index_> tmp_ranked;
        tmp_ranked.reserve(num_universe);

        auto ext = tatami::consecutive_extractor<ref_sparse_>(&ref, false, start, len, universe_ptr); 
        for (Index_ c = start, end = start + len; c < end; ++c) {
            tmp_ranked.clear();

            if constexpr(ref_sparse_) {
                auto info = ext->fetch(vbuffer.data(), ibuffer.data());
                for (I<decltype(info.number)> i = 0; i < info.number; ++i) {
                    tmp_ranked.emplace_back(info.value[i], remap_to_universe[info.index[i]]);
                }
            } else {
                auto ptr = ext->fetch(vbuffer.data());
                for (I<decltype(num_universe)> i = 0; i < num_universe; ++i) {
                    tmp_ranked.emplace_back(ptr[i], i); // a.k.a. remap_to_universe[universe[i]]. 
                }
            }

            std::sort(tmp_ranked.begin(), tmp_ranked.end());
            auto& final_ranked = out_ranked[curlab[c]][positions[c]];
            simplify_ranks(tmp_ranked, final_ranked);
        }
    }, NC, options.num_threads);
}

template<bool ref_sparse_, typename Value_, typename Index_, typename RefLabel_>
void train_integrated_per_reference_intersect(
    const TrainIntegratedInput<Value_, RefRefLabel_, Index_>& input,
    const std::vector<Index_>& universe,
    const std::vector<Index_>& remap_to_universe,
    const Index_ test_nrow,
    const TrainIntegratedOptions& options,
    const std::vector<Index_>& positions,
    typename TrainedIntegrated<Index_>::PerReference& output
) {
    std::vector<Index_> ref_subset;
    auto remap_ref_subset_to_universe = sanisizer::create<std::vector<Index_> >(ref->nrow());
    for (const auto& pair : *(input.intersection)) {
        const auto rdex = remap_to_universe[pair.first];
        if (rdex != test_nrow) {
            ref_subset.push_back(pair.second);
            remap_ref_subset_to_universe[pair.second] = rdex;
        }
    }
    std::sort(ref_subset.begin(), ref_subset.end());

    typename std::conditional<ref_sparse_, bool, std::vector<Index_> >::type remap_dense_to_universe;
    if constexpr(!ref_sparse_) {
        remap_dense_to_universe.reserve(ref_subset.size());
        for (auto r : ref_subset) {
            remap_dense_to_universe.push_back(remap_ref_subset_to_universe[r]);
        }
    }

    tatami::VectorPtr<Index_> to_extract_ptr(tatami::VectorPtr<Index_>{}, &ref_subset);
    tatami::parallelize([&](int, Index_ start, Index_ len) {
        const auto ref_subset_size = ref_subset.size();
        auto vbuffer = sanisizer::create<std::vector<Value_> >(ref_subset_size);
        auto ibuffer = [&]() {
            if constexpr(ref_sparse_) {
                return sanisizer::create<std::vector<Index_> >(ref_subset_size);
            } else {
                return false;
            }
        }();

        internal::RankedVector<Value_, Index_> tmp_ranked;
        tmp_ranked.reserve(ref_subset_size);
        auto ext = tatami::consecutive_extractor<false>(&ref, false, start, len, to_extract_ptr);

        for (Index_ c = start, end = start + len; c < end; ++c) {
            tmp_ranked.clear();

            if constexpr(ref_sparse_) {
                auto info = ext->fetch(vbuffer.data(), ibuffer.data());
                for (I<decltype(info.number)> i = 0; i < info.number; ++i) {
                    tmp_ranked.emplace_back(info.value[i], remap_ref_to_universe[info.index[i]]);
                }
            } else {
                auto ptr = ext->fetch(vbuffer.data());
                for (I<decltype(ref_subset_size)> i = 0; i < ref_subset_size; ++i) {
                    tmp_ranked.emplace_back(ptr[i], remap_dense_to_universe[i]);
                }
            }

            std::sort(tmp_ranked.begin(), tmp_ranked.end());
            auto& final_ranked = cur_ranked[curlab[c]][positions[c]];
            simplify_ranks(tmp_ranked, final_ranked);
        }
    }, NC, options.num_threads);
}
/**
 * @endcond
 */

/**
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam RefLabel_ Integer type for the reference labels.
 *
 * @param inputs Vector of references, typically constructed with `prepare_integrated_input()` or `prepare_integrated_input_intersect()`.
 * @param options Further options.
 *
 * @return A pre-built classifier that integrates multiple references, for use in `classify_integrated()`.
 */
template<typename Value_, typename Index_, typename RefLabel_>
TrainedIntegrated<Index_> train_integrated(const std::vector<TrainIntegratedInput<Value_, Index_, RefLabel_> >& inputs, const TrainIntegratedOptions& options) {
    TrainedIntegrated<Index_> output;
    const auto nrefs = inputs.size();
    sanisizer::resize(output.my_references, nrefs);

    // Checking that the number of genes in the test dataset are consistent.
    output.test_nrow = 0;
    if (inputs.size()) {
        output.test_nrow = inputs.front().test_nrow;
        for (const auto& in : inputs) {
            if (sanisizer::is_equal(in.test_nrow, output.test_nrow)) {
                throw std::runtime_error("inconsistent number of rows in the test dataset across entries of 'inputs'");
            }
        }
    }

    // Identify the intersection of all marker genes as the universe.
    // Note that 'output.universe' contains sorted and unique row indices for the test matrix, where 'remap_to_universe[universe[i]] == i'.
    auto remap_to_universe = sanisizer::create<std::vector<Index_> >(output.test_nrow, output.test_nrow);
    {
        auto num_hits = sanisizer::create<std::vector<I<decltype(nrefs)> > >(output.test_nrow);
        auto current_used = sanisizer::create<std::vector<char> >(output.test_nrow);
        for (const auto& in : inputs) {
            std::fill(current_used.begin(), current_used.end(), false);

            for (const auto& labmrk : in.markers) {
                for (const auto& mrk : labmrk) {
                    for (auto y : mrk) {
                        if (!current_used[y]) {
                            num_hits[y] += 1;
                            current_used[y] = true;
                        }
                    }
                }
            }
        }

        output.universe.reserve(output.test_nrow);
        for (Index_ t = 0; t < output.test_nrow; ++t) {
            if (num_hits[t] == nrefs) {
                remap_to_universe[t] = output.universe.size();
                output.universe.push_back(t);
            }
        }

        output.universe.shrink_to_fit();
    }

    auto is_active = sanisizer::create<std::vector<char> >(output.test_nrow);
    std::vector<Index_> active_genes;
    active_genes.reserve(output.test_nrow);

    for (I<decltype(nrefs)> r = 0; r < nrefs; ++r) {
        auto& curinput = inputs[r];
        auto& curoutput = output.my_references[r];
        const auto nlabels = curinput.markers.size();
        const Index_ NC = curinput.ref->ncol();
        curoutput.num_samples = NC;

        // Assembling the per-label markers for this reference.
        for (I<decltype(nlabels)> i = 0; i < nlabels; ++i) {
            for (const auto& labmark : curinput.markers[i]) {
                for (const auto a : labmark) {
                    if (!is_active[a]) {
                        is_active[a] = true;
                        active_genes.push_back(a);
                    }
                }
            }

            curoutput.markers.emplace_back();
            auto& last_markers = curoutput.markers.back();
            last_markers.reserve(active_genes.size());

            for (const auto a : active_genes) {
                const auto universe_index = remap_to_universe[curinput.test_subset[a]];
                if (universe_index != test_nrow) {
                    last_markers.push_back(universe_index);
                }
                is_active[a] = false;
            }
        }

        // Pre-allocating the ranked vectors.
        auto out_ranked = sanisizer::create<std::vector<RankedVector<Index_, Index_> > >(nlabels);
        std::vector<Index_> positions;
        positions.reserve(NC);
        {
            auto samples_per_label = sanisizer::create<std::vector<Index_> >(nlabels);
            for (Index_ c = 0; c < NC; ++c) {
                auto& pos = samples_per_label[curlab[c]];
                positions.push_back(pos);
                ++pos;
            }

            for (I<decltype(nlabels)> l = 0; l < nlabels; ++l) {
                out_ranked[l].resize(samples_per_label[l]);
            }
        }

        if (curinput.ref->is_sparse()) {
            if (curinput.intersection) {
                train_integrated_per_reference_direct<true, Value_>(r, curinput, universe, remap_to_universe, output.test_nrow, options, positions, out_ranked);
            } else {
                train_integrated_per_reference_direct<true, Value_>(r, curinput, universe, remap_to_universe, options, positions, out_ranked);
            }

            I<decltype(curoutput.all_ranked.size())> num_total = 0;
            for (const auto& x : out_ranked) {
                num_total = sanisizer::sum<I<decltype(num_total)> >(num_total, x.size());
            }

            curoutput.all_ranked.reserve(num_total);
            curoutput.all_ranked_indptrs->emplace();
            curoutput.all_ranked_indptrs->reserve(out_ranked.size());
            curoutput.all_ranked_indptrs->push_back(0);

            for (const auto& x : out_ranked) {
                curoutput.all_ranked.insert(output.all_ranked.end(), x.begin(), x.end());
                curoutput.all_ranked_indptrs->push_back(output.all_ranked.size());
            }

        } else {
            if (curinput.intersection) {
                train_integrated_per_reference_direct<false, Value_>(r, curinput, universe, remap_to_universe, output.test_nrow, options, positions, out_ranked);
            } else {
                train_integrated_per_reference_direct<false, Value_>(r, curinput, universe, remap_to_universe, options, positions, out_ranked);
            }

            curoutput.all_ranked.reserve(sanisizer::product<I<decltype(output.all_ranked.size())> >(output.universe.size(), curoutput.num_samples));
            for (const auto& x : out_ranked) {
                curoutput.all_ranked.insert(curoutput.all_ranked.end(), x.begin(), x.end());
            }
        }
    }

    return output;
}

}

#endif
