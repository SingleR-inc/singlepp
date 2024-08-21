#ifndef SINGLEPP_TRAIN_INTEGRATED_HPP
#define SINGLEPP_TRAIN_INTEGRATED_HPP

#include "macros.hpp"

#include "scaled_ranks.hpp"
#include "train_single.hpp"
#include "Intersection.hpp"

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <cstdint>

/**
 * @file IntegratedBuilder.hpp
 *
 * @brief Prepare for integrated classification across references.
 */

namespace singlepp {

template<typename Value_, typename Index_, typename Label_>
struct TrainIntegratedInput {
    /**
     * @cond
     */
    const tatami::Matrix<Value_, Index_>* ref;

    const Label_* labels;

    std::vector<std::vector<Index_> > markers;

    bool check_availability = false;

    std::unordered_map<Index_, Index_> gene_mapping;
    /**
     * @endcond
     */
};

/**
 * Add a reference dataset to this object for later use in `finish()`.
 * This overload assumes that the reference and test datasets have the same features,
 * and that the reference dataset has already been processed through `BasicBuilder::run()`.
 * `ref` and `labels` are expected to remain valid until `finish()` is called.
 *
 * @param ref Matrix containing the reference expression values.
 * Rows are features and columns are reference samples.
 * The number and identity of features should be identical to the test dataset to be classified in `IntegratedScorer`.
 * @param[in] labels Pointer to an array of label assignments.
 * The smallest label should be 0 and the largest label should be equal to the total number of unique labels minus 1.
 * @param built The built reference created by running `BasicBuilder::run()` on `ref` and `labels`.
 */
template<typename Value_, typename Index_, typename Label_, typename Float_>
TrainIntegratedInput<Value_, Index_, Label_> prepare_integrated_input(
    const tatami::Matrix<Value_, Index_>& ref,
    const Label_* labels, 
    const TrainedSingle<Index_, Float_>& trained)
{
    TrainIntegratedInput<Value_, Index_, Label_> output;
    output.ref = &ref;
    output.labels = labels;

    const auto& subset = trained.get_subset();
    const auto& old_markers = trained.get_markers();
    size_t nlabels = old_markers.size();

    // Adding the markers for each label, indexed according to their
    // position in the test matrix. This assumes that 'mat_subset' is
    // appropriately specified to contain the test's row indices. 
    auto& new_markers = output.markers;
    new_markers.reserve(nlabels);
    std::unordered_set<Index_> unified;

    for (size_t i = 0; i < nlabels; ++i) {
        unified.clear();
        for (const auto& x : old_markers[i]) {
            unified.insert(x.begin(), x.end());
        }
        new_markers.emplace_back(unified.begin(), unified.end());
        auto& cur_new_markers = new_markers.back();
        for (auto& y : cur_new_markers) {
            y = subset[y];
        }
    }

    return output;
}

/**
 * Add a reference dataset to this object for later use in `finish()`.
 * This overload requires an existing intersection between the test and reference datasets,
 * and assumes that the reference dataset has already been processed through `BasicBuilder::run()`.
 * `ref` and `labels` are expected to remain valid until `finish()` is called.
 *
 * @param intersection Vector defining the intersection of features between the test and reference datasets.
 * Each entry is a pair where the first element is the row index in the test matrix,
 * and the second element is the row index for the corresponding feature in the reference matrix.
 * Each row index for either matrix should occur no more than once in `intersection`.
 * @param ref An expression matrix for the reference expression profiles, where rows are genes and columns are cells.
 * This should have non-zero columns.
 * @param[in] labels An array of length equal to the number of columns of `ref`, containing the label for each sample.
 * The smallest label should be 0 and the largest label should be equal to the total number of unique labels minus 1.
 * @param built The built reference created by running `BasicBuilder::run()` on all preceding arguments.
 */
template<typename Index_, typename Value_, typename Label_, typename Float_>
TrainIntegratedInput<Value_, Index_, Label_> prepare_integrated_input_intersect(
    const Intersection<Index_>& intersection,
    const tatami::Matrix<Value_, Index_>* ref, 
    const Label_* labels, 
    const BasicBuilder::PrebuiltIntersection<Index_, Float_>& trained) 
{
    TrainIntegratedInput<Value_, Index_, Label_> output;
    output.ref = &ref;
    output.labels = labels;

    // Manually constructing the markers. This involves (i) pruning out the
    // markers that aren't present in the intersection, and (ii) updating 
    // their indices so that they point to rows of 'mat', not 'ref'.
    auto& mapping = output.gene_mapping;
    for (const auto& i : intersection) {
        reverse_map[i.second] = i.first;
    }

    auto subindex = [&](int i) -> int {
        if constexpr(!std::is_same<Subset, bool>::value) {
            return ref_subset[i];
        } else {
            return i;
        }
    };

    auto& new_markers = output.markers;
    size_t nlabels = old_markers.size();
    new_markers.resize(nlabels);

    for (size_t i = 0; i < nlabels; ++i) {
        const auto& cur_old_markers = old_markers[i];

        std::unordered_set<Index_> unified;
        for (const auto& x : cur_old_markers) {
            unified.insert(x.begin(), x.end());
        }
        auto& cur_new_markers = new_markers[i];
        cur_new_markers.reserve(unified.size());

        for (auto y : unified) {
            auto it = reverse_map.find(subindex(y));
            if (it != reverse_map.end()) {
                cur_new_markers.push_back(it->second);
            }
        }
    }

    // Constructing the mapping of mat's rows to the reference rows.
    output.check_availability = true;
    mapping.clear();
    for (const auto& i : intersection) {
        mapping[i.first] = i.second;
    }

    return output;
}

/**
 * Add a reference dataset to this object for later use in `finish()`.
 * This overload automatically identifies the intersection of features between the test and reference datasets,
 * and assumes that the reference dataset has already been processed through `BasicBuilder::run()`.
 * `ref` and `labels` are expected to remain valid until `finish()` is called.
 * `mat_id` and `mat_nrow` should also be constant for all invocations to `add()`.
 *
 * @tparam Id Type of the gene identifier for each row.
 *
 * @param mat_nrow Number of rows (genes) in the test dataset.
 * @param[in] mat_id Pointer to an array of identifiers of length equal to `mat_nrow`.
 * This should contain a unique identifier for each row of `mat` (typically a gene name or index).
 * If any duplicate IDs are present, only the first occurrence is used.
 * @param ref An expression matrix for the reference expression profiles, where rows are genes and columns are cells.
 * This should have non-zero columns.
 * @param[in] ref_id Pointer to an array of identifiers of length equal to the number of rows of any `ref`.
 * This should contain a unique identifier for each row in `ref`, and should be comparable to `mat_id`.
 * If any duplicate IDs are present, only the first occurrence is used.
 * @param[in] labels An array of length equal to the number of columns of `ref`, containing the label for each sample.
 * The smallest label should be 0 and the largest label should be equal to the total number of unique labels minus 1.
 * @param built The built reference created by running `BasicBuilder::run()` on all preceding arguments.
 */
template<typename Index_, typename Id_, typename Value_, typename Label_, typename Float_>
TrainIntegratedInput<Value_, Index_, Label_> prepare_integrated_input_intersect(
    Index_ test_nrow,
    const Id_* test_id, 
    const tatami::Matrix<Value_, Index_>& ref, 
    const Id_* ref_id, 
    const Label_* labels,
    const BasicBuilder::PrebuiltIntersection<Index_, Float_>& trained) 
{
    auto intersection = intersect_genes(mat_nrow, mat_id, ref->nrow(), ref_id);
    return prepare_integrated_input_intersect(intersection, ref, labels, built);
}

/**
 * @brief Classifier that integrates multiple reference datasets.
 */
template<typename Index_>
class TrainedIntegrated {
public:
    /**
     * @return Number of reference datasets.
     * Each object corresponds to the reference used in an `IntegratedBuilder::add()` call, in the same order.
     */
    size_t num_references() const {
        return markers.size();
    }

    /**
     * @param r Reference dataset of interest.
     * @return Number of labels in this reference.
     */
    size_t num_labels(size_t r) const {
        return markers[r].size();
    }

    /**
     * @param r Reference dataset of interest.
     * @return Number of profiles in this reference.
     */
    size_t num_profiles(size_t r) const {
        size_t n = 0;
        for (const auto& ref : ranked[r]) {
            n += ref.size();
        }
        return n;
    }

public:
    /**
     * @cond
     */
    // Technically this should be private, but it's a pain to add
    // templated friend functions, so I can't be bothered.
    std::vector<Index_> universe; // To be used by IntegratedScorer for indexed extraction.

    std::vector<uint8_t> check_availability;
    std::vector<std::unordered_set<Index_> > available; // indices to 'universe'
    std::vector<std::vector<std::vector<Index_> > > markers; // indices to 'universe'
    std::vector<std::vector<std::vector<RankedVector<Index_, Index_> > > > ranked; // .second contains indices to 'universe'
    /**
     * @endcond
     */
};

/**
 * @brief Options for `train_integrated()` and friends.
 */
struct TrainIntegratedOptions {
    /**
     * Number of threads to use.
     */
    int num_threads = 1;
};


/**
 * @cond
 */
namespace internal {

template<typename Value_, typename Index_, typename Label_>
void fill_ranks(
    const tatami::Matrix<Value_, Index_>& ref,
    const Label_* curlab, 
    const std::vector<Index_>& subset, 
    const std::vector<Index_>& positions, 
    std::vector<std::vector<RankedVector<Index_, Index_> > >& cur_ranked,
    int num_threads) 
{
    tatami::parallelize([&](size_t, Index_ start, Index_ len) -> void {
        RankedVector<Value_, Index_> tmp_ranked;
        tmp_ranked.reserve(subset.size());

        // 'subset' is guaranteed to be sorted and unique, see its derivation in finish().
        // This means we can directly use it for indexed extraction.
        tatami::VectorPtr<Index_> subset_ptr(tatami::Vector<Index_>{}, &subset);
        auto wrk = tatami::consecutive_extractor<false>(&ref, false, start, len, std::move(subset_ptr)); 
        std::vector<Value_> buffer(subset.size());

        for (Index_ c = start, end = start + len; c < end; ++c) {
            auto ptr = wrk->fetch(buffer.data());

            tmp_ranked.clear();
            for (int i = 0, end = subset.size(); i < end; ++i, ++ptr) {
                tmp_ranked.emplace_back(*ptr, i);
            }
            std::sort(tmp_ranked.begin(), tmp_ranked.end());

            auto& final_ranked = cur_ranked[curlab[c]][positions[c]];
            simplify_ranks(tmp_ranked, final_ranked);
        }
    }, ref->ncol(), num_threads);
}

template<typename Value_, typename Index_, typename Label_>
void fill_ranks_intersect(
    const tatami::Matrix<Value_, Index_>& ref, 
    const Label_* curlab, 
    const std::vector<Index_>& subset, 
    const std::vector<Index_>& positions,
    const std::unordered_map<Index_, Index_>& cur_mapping,
    std::unordered_set<Index_>& cur_available,
    std::vector<std::vector<RankedVector<Index_, Index_> > >& cur_ranked,
    int num_threads)
{
    // If we need to check availability, then we need to check
    // the mapping of test genes to row indices of the reference.
    std::vector<std::pair<Index_, Index_> > remapping; 
    remapping.reserve(subset.size());

    for (Index_ i = 0, end = subset.size(); i < end; ++i) {
        auto it = cur_mapping.find(subset[i]);
        if (it != cur_mapping.end()) {
            remapping.emplace_back(it->second, i); // using 'i' instead of 'subset[i]', as we want to work with indices into 'subset', not the values of 'subset' themselves.
            cur_available.insert(i);
        }
    }

    std::sort(remapping.begin(), remapping.end());

    // This section is just to enable indexed extraction by tatami.
    // There's no need to consider duplicates among the
    // 'remapping[i].first', 'cur_mapping->second' is guaranteed to be
    // unique as a consequence of how intersect_genes() works.
    std::vector<Index_> remapped_subset; 
    remapped_subset.reserve(remapping.size());
    for (const auto& p : remapping) {
        remapped_subset.push_back(p.first);
    }

    tatami::parallelize([&](size_t, Index_ start, Index_ len) -> void {
        RankedVector<Value_, Index_> tmp_ranked;
        tmp_ranked.reserve(remapped_subset.size());

        std::vector<Value_> buffer(remapped_subset.size());
        tatami::VectorPtr<Index_> remapped_subset_ptr(tatami::VectorPtr<Index_>{}, &remapped_subset);
        auto wrk = tatami::consecutive_extractor<false>(&ref, false, start, len, std::move(remapped_subset_ptr));

        for (size_t c = start, end = start + len; c < end; ++c) {
            auto ptr = wrk->fetch(c, buffer.data());

            tmp_ranked.clear();
            for (const auto& p : remapping) {
                tmp_ranked.emplace_back(*ptr, p.second); // remember, 'p.second' corresponds to indices Index_o 'subset'.
                ++ptr;
            }
            std::sort(tmp_ranked.begin(), tmp_ranked.end());

            auto& final_ranked = cur_ranked[curlab[c]][positions[c]];
            simplify_ranks(tmp_ranked, final_ranked);
        }
    }, ref.ncol(), num_threads);
}

template<typename Index_, typename Inputs_>
TrainedIntegrated<Index_> train_integrated(Inputs_& input, const TrainIntegratedOptions& options) {
    TrainedIntegrated<Index_> output;
    size_t nrefs = input.size();
    output.check_availability.resize(nrefs);
    output.available.resize(nrefs);
    output.markers.resize(nrefs);
    output.ranked.resize(nrefs);

    // Identify the union of all marker genes.
    auto& subset = output.universe;
    std::unordered_map<Index_, Index_> remap_to_universe;
    {
        std::unordered_set<Index_> subset_tmp;
        for (const auto& in : inputs) {
            for (const auto& mrk : in.markers) {
                subset_tmp.insert(mrk.begin(), mrk.end());
            }
        }

        subset.insert(subset.end(), subset_tmp.begin(), subset_tmp.end());
        std::sort(subset.begin(), subset.end());
        remap_to_universe.resize(subset.size());
        for (Index_ i = 0, end = subset.size(); i < end; ++i) {
            remap_to_universe[subset[i]] = i;
        }
    }

    for (size_t r = 0; r < nrefs; ++r) {
        const auto& curinput = inputs[r];
        auto curlab = curinput.labels;
        const auto& ref = *(curinput.ref);

        // Reindexing the markers to point to the universe.
        auto& curmarkers = references.markers[r];
        if constexpr(std::is_const<Input_>::value) {
            curmarkers.swap(curinput.markers);
        } else {
            curmarkers = curinput.markers;
        }
        for (auto& outer : curmarkers) {
            for (auto& x : outer) {
                x = remap_to_universe.find(x)->second;
            }
        }

        // Preparing the vectors of pre-ranked expression..
        auto& cur_ranked = references.ranked[r];
        std::vector<Index_> positions;
        {
            size_t nlabels = curmarkers.size();
            size_t NC = ref->ncol();
            positions.reserve(NC);                

            std::vector<Index_> samples_per_label(nlabels);
            for (size_t c = 0; c < NC; ++c) {
                auto& pos = samples_per_label[curlab[c]];
                positions.push_back(pos);
                ++pos;
            }

            cur_ranked.resize(nlabels);
            for (size_t l = 0; l < nlabels; ++l) {
                cur_ranked[l].resize(samples_per_label[l]);
            }
        }

        // Finally filling the rankings.
        if (!curinput.check_availability) {
            fill_ranks(ref, curlab, subset, positions, cur_ranked, options.num_threads);
        } else {
            references.check_availability[r] = 1;
            fill_ranks_intersect(ref, curlab, subset, positions, curinput.gene_mapping, references.available[r], cur_ranked, options.num_threads);
        }
    }

    return references;
}

}
/**
 * @endcond
 */

template<typename Value_, typename Index_, typename Label_>
TrainedIntegrated<Index_> train_integrated(const std::vector<TrainIntegratedInput<Value_, Index_, Label_>& inputs, const TrainIntegratedOptions& options) {
    return train_integrated(inputs, options);
}

template<typename Value_, typename Index_, typename Label_>
TrainedIntegrated<Index_> train_integrated(std::vector<TrainIntegratedInput<Value_, Index_, Label_> inputs, const TrainIntegratedOptions& options) {
    return train_integrated(inputs, options);
}

}

#endif
