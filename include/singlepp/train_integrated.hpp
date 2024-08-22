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
#include <memory>

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

    bool with_intersection = false;

    const Intersection<Index_>* user_intersection = NULL;

    Intersection<Index_> auto_intersection;
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
    const tatami::Matrix<Value_, Index_>& ref, 
    const Label_* labels, 
    const TrainedSingleIntersect<Index_, Float_>& trained) 
{
    TrainIntegratedInput<Value_, Index_, Label_> output;
    output.ref = &ref;
    output.labels = labels;

    // Updating the markers so that they point to rows of the test matrix.
    const auto& old_markers = trained.get_markers();
    size_t nlabels = old_markers.size();
    auto& new_markers = output.markers;
    new_markers.resize(nlabels);

    const auto& test_subset = trained.get_test_subset();
    std::unordered_set<Index_> unified;

    for (size_t i = 0; i < nlabels; ++i) {
        const auto& cur_old_markers = old_markers[i];

        unified.clear();
        for (const auto& x : cur_old_markers) {
            unified.insert(x.begin(), x.end());
        }

        auto& cur_new_markers = new_markers[i];
        cur_new_markers.reserve(unified.size());
        for (auto y : unified) {
            cur_new_markers.push_back(test_subset[y]);
        }
    }

    output.with_intersection = true;
    output.user_intersection = &intersection;
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
    const TrainedSingleIntersect<Index_, Float_>& trained) 
{
    auto intersection = intersect_genes(test_nrow, test_id, ref.nrow(), ref_id);
    auto output = prepare_integrated_input_intersect(intersection, ref, labels, trained);
    output.user_intersection = NULL;
    output.auto_intersection.swap(intersection);
    return output;
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
    std::vector<std::vector<std::vector<internal::RankedVector<Index_, Index_> > > > ranked; // .second contains indices to 'universe'
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

template<typename Value_, typename Index_, typename Input_>
void train_integrated(
    size_t ref_i,
    Input_& curinput,
    TrainedIntegrated<Index_>& output,
    const std::unordered_map<Index_, Index_> remap_to_universe,
    const TrainIntegratedOptions& options)
{
    auto curlab = curinput.labels;
    const auto& ref = *(curinput.ref);

    // Reindexing the markers so that they contain indices into to the universe.
    auto& curmarkers = output.markers[ref_i];
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

    // Pre-allocating the vectors of pre-ranked expression.
    auto& cur_ranked = output.ranked[ref_i];
    std::vector<Index_> positions;
    {
        size_t nlabels = curmarkers.size();
        Index_ NC = ref.ncol();
        positions.reserve(NC);                

        std::vector<Index_> samples_per_label(nlabels);
        for (Index_ c = 0; c < NC; ++c) {
            auto& pos = samples_per_label[curlab[c]];
            positions.push_back(pos);
            ++pos;
        }

        cur_ranked.resize(nlabels);
        for (size_t l = 0; l < nlabels; ++l) {
            cur_ranked[l].resize(samples_per_label[l]);
        }
    }

    if (!curinput.with_intersection) {
        tatami::parallelize([&](size_t, Index_ start, Index_ len) -> void {
            internal::RankedVector<Value_, Index_> tmp_ranked;
            tmp_ranked.reserve(output.universe.size());

            // The universe is guaranteed to be sorted and unique, see its derivation above.
            // This means we can directly use it for indexed extraction.
            tatami::VectorPtr<Index_> universe_ptr(tatami::VectorPtr<Index_>{}, &(output.universe));
            auto wrk = tatami::consecutive_extractor<false>(&ref, false, start, len, std::move(universe_ptr)); 
            std::vector<Value_> buffer(output.universe.size());

            for (Index_ c = start, end = start + len; c < end; ++c) {
                auto ptr = wrk->fetch(buffer.data());

                tmp_ranked.clear();
                for (int i = 0, end = output.universe.size(); i < end; ++i, ++ptr) {
                    tmp_ranked.emplace_back(*ptr, i);
                }
                std::sort(tmp_ranked.begin(), tmp_ranked.end());

                auto& final_ranked = cur_ranked[curlab[c]][positions[c]];
                simplify_ranks(tmp_ranked, final_ranked);
            }
        }, ref.ncol(), options.num_threads);

    } else {
        output.check_availability[ref_i] = 1;

        // Need to remap from indices on the test matrix to those in the current reference matrix
        // so that we can form an appropriate vector for indexed tatami extraction. 
        const auto& intersection = (curinput.user_intersection == NULL ? curinput.auto_intersection : *(curinput.user_intersection));
        std::unordered_map<Index_, Index_> intersection_map;
        intersection_map.reserve(intersection.size());
        for (const auto& in : intersection) {
            intersection_map[in.first] = in.second;
        }

        std::vector<std::pair<Index_, Index_> > intersection_in_universe;
        intersection_in_universe.reserve(output.universe.size());
        auto& cur_available = output.available[ref_i];
        cur_available.reserve(output.universe.size());

        for (Index_ i = 0, end = output.universe.size(); i < end; ++i) {
            auto it = intersection_map.find(output.universe[i]);
            if (it != intersection_map.end()) {
                intersection_in_universe.emplace_back(it->second, i); // using 'i' as we want to work with indices into 'universe', not the indices of the universe itself.
                cur_available.insert(i);
            }
        }
        std::sort(intersection_in_universe.begin(), intersection_in_universe.end());

        std::vector<Index_> to_extract; 
        to_extract.reserve(intersection_in_universe.size());
        for (const auto& p : intersection_in_universe) {
            to_extract.push_back(p.first);
        }

        tatami::parallelize([&](size_t, Index_ start, Index_ len) -> void {
            internal::RankedVector<Value_, Index_> tmp_ranked;
            tmp_ranked.reserve(to_extract.size());

            std::vector<Value_> buffer(to_extract.size());
            tatami::VectorPtr<Index_> to_extract_ptr(tatami::VectorPtr<Index_>{}, &to_extract);
            auto wrk = tatami::consecutive_extractor<false>(&ref, false, start, len, std::move(to_extract_ptr));

            for (size_t c = start, end = start + len; c < end; ++c) {
                auto ptr = wrk->fetch(buffer.data());

                tmp_ranked.clear();
                for (const auto& p : intersection_in_universe) {
                    tmp_ranked.emplace_back(*ptr, p.second); // remember, 'p.second' corresponds to indices into the universe.
                    ++ptr;
                }
                std::sort(tmp_ranked.begin(), tmp_ranked.end());

                auto& final_ranked = cur_ranked[curlab[c]][positions[c]];
                simplify_ranks(tmp_ranked, final_ranked);
            }
        }, ref.ncol(), options.num_threads);
    }
}

template<typename Value_, typename Index_, typename Inputs_>
TrainedIntegrated<Index_> train_integrated(Inputs_& inputs, const TrainIntegratedOptions& options) {
    TrainedIntegrated<Index_> output;
    size_t nrefs = inputs.size();
    output.check_availability.resize(nrefs);
    output.available.resize(nrefs);
    output.markers.resize(nrefs);
    output.ranked.resize(nrefs);

    // Identify the union of all marker genes.
    std::unordered_map<Index_, Index_> remap_to_universe;
    std::unordered_set<Index_> subset_tmp;
    for (const auto& in : inputs) {
        for (const auto& mrk : in.markers) {
            subset_tmp.insert(mrk.begin(), mrk.end());
        }
    }

    output.universe.insert(output.universe.end(), subset_tmp.begin(), subset_tmp.end());
    std::sort(output.universe.begin(), output.universe.end());
    remap_to_universe.reserve(output.universe.size());
    for (Index_ i = 0, end = output.universe.size(); i < end; ++i) {
        remap_to_universe[output.universe[i]] = i;
    }

    for (size_t r = 0; r < nrefs; ++r) {
        train_integrated<Value_>(r, inputs[r], output, remap_to_universe, options);
    }

    return output;
}

}
/**
 * @endcond
 */

template<typename Value_, typename Index_, typename Label_>
TrainedIntegrated<Index_> train_integrated(const std::vector<TrainIntegratedInput<Value_, Index_, Label_> >& inputs, const TrainIntegratedOptions& options) {
    return internal::train_integrated<Value_, Index_>(inputs, options);
}

template<typename Value_, typename Index_, typename Label_>
TrainedIntegrated<Index_> train_integrated(std::vector<TrainIntegratedInput<Value_, Index_, Label_> >&& inputs, const TrainIntegratedOptions& options) {
    return internal::train_integrated<Value_, Index_>(inputs, options);
}

}

#endif
