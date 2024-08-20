#ifndef SINGLEPP_FINE_TUNE_HPP
#define SINGLEPP_FINE_TUNE_HPP

#include "macros.hpp"

#include "knncolle/knncolle.hpp"

#include "scaled_ranks.hpp"
#include "compute_scores.hpp"
#include "build_indices.hpp"
#include "Markers.hpp"

#include <vector>
#include <algorithm>
#include <type_traits>

namespace singlepp {

namespace internal {

template<typename Stat_, typename Label_>
std::pair<Label_, Stat_> fill_labels_in_use(const std::vector<Stat_>& scores, Stat_ threshold, std::vector<Label_>& in_use) {
    static_assert(std::is_floating_point<Stat_>::value);
    static_assert(std::is_integral<Label_>::value);

    auto it = std::max_element(scores.begin(), scores.end());
    Label_ best_label = it - scores.begin();
    Stat_ max_score = *it;

    in_use.clear();
    constexpr Stat_ DUMMY = -1000;
    Stat_ next_score = DUMMY;
    const Stat_ bound = max_score - threshold;

    size_t nscores = scores.size();
    for (size_t i = 0; i < nscores; ++i) {
        auto val = scores[i];
        if (val >= bound) {
            in_use.push_back(i);
        }
        if (i != best_label && next_score < val) {
            next_score = val;
        }
    }

    return std::make_pair(best_label, max_score - next_score); 
}

template<typename Stat_, typename Label_>
std::pair<Label_, Stat_> replace_labels_in_use(const std::vector<Stat_>& scores, Stat_ threshold, std::vector<Label_>& in_use) {
    static_assert(std::is_floating_point<Stat_>::value);
    static_assert(std::is_integral<Label_>::value);

    auto it = std::max_element(scores.begin(), scores.end());
    size_t best_index = it - scores.begin();
    Stat_ max_score = *it;

    Label_ best_label = in_use[best_index];
    size_t counter = 0;

    constexpr Stat_ DUMMY = -1000;
    Stat_ next_score = DUMMY;
    const Stat_ bound = max_score - threshold;

    size_t nscores = scores.size();
    for (size_t i = 0; i < nscores; ++i) {
        const auto& val = scores[i];
        if (val >= bound) {
            in_use[counter] = in_use[i];
            ++counter;
        }
        if (i != best_index && next_score < val) {
            next_score = val;
        }
    }

    in_use.resize(counter);
    return std::make_pair(best_label, max_score - next_score); 
}

template<typename Label_, typename Index_, typename Float_, typename Value_>
class FineTuner {
private:
    std::vector<Label_> my_labels_in_use;

    RankRemapper<Index_> my_gene_subset;

    std::vector<Float_> my_scaled_left, my_scaled_right;

    std::vector<Float_> my_all_correlations;

    RankedVector<Value_, Index_> my_input_sub;

    RankedVector<Index_, Index_> my_ref_sub;

public:
    template<bool test_ = false>
    std::pair<Label_, Float_> run(
        const RankedVector<Value_, Index_>& input, 
        const std::vector<PerLabelReference<Index_, Float_> >& ref,
        const Markers<Index_>& markers,
        std::vector<Float_>& scores,
        Float_ quantile,
        Float_ threshold)
    {
        if (scores.size() <= 1) {
            return std::pair<Label_, Float_>(0, std::numeric_limits<Float_>::quiet_NaN());
        } 

        auto candidate = fill_labels_in_use(scores, threshold, my_labels_in_use);

        // If there's only one top label, we don't need to do anything else.
        if (my_labels_in_use.size() == 1) {
            return candidate;
        } 

        // We also give up if every label is in range, because any subsequent
        // calculations would use all markers and just give the same result.
        // The 'test' parameter allows us to skip this bypass for testing.
        if constexpr(!test_) {
            if (my_labels_in_use.size() == ref.size()) {
                return candidate;
            }
        }

        // Use the input_size as a hint for the number of addressable genes.
        // This should be exact if subset_to_markers() was used on the input,
        // but the rest of the code is safe even if the hint isn't perfect.
        my_gene_subset.reserve(input.size());

        while (my_labels_in_use.size() > 1) {
            my_gene_subset.clear();
            for (auto l : my_labels_in_use) {
                for (auto l2 : my_labels_in_use){ 
                    for (auto c : markers[l][l2]) {
                        my_gene_subset.add(c);
                    }
                }
            }

            my_gene_subset.remap(input, my_input_sub);
            my_scaled_left.resize(my_input_sub.size());
            my_scaled_right.resize(my_input_sub.size());
            scaled_ranks(my_input_sub, my_scaled_left.data());
            scores.clear();

            size_t nlabels_used = my_labels_in_use.size();
            for (size_t i = 0; i < nlabels_used; ++i) {
                auto curlab = my_labels_in_use[i];

                my_all_correlations.clear();
                const auto& curref = ref[curlab];
                size_t NR = curref.index->num_dimensions();
                size_t NC = curref.index->num_observations();

                for (size_t c = 0; c < NC; ++c) {
                    // Technically we could be faster if we remembered the
                    // subset from the previous fine-tuning iteration, but this
                    // requires us to (possibly) make a copy of the entire
                    // reference set; we can't afford to do this in each thread.
                    my_gene_subset.remap(curref.ranked[c], my_ref_sub);
                    scaled_ranks(my_ref_sub, my_scaled_right.data());

                    Float_ cor = distance_to_correlation<Float_>(my_scaled_left, my_scaled_right);
                    my_all_correlations.push_back(cor);
                }

                Float_ score = correlations_to_scores(my_all_correlations, quantile);
                scores.push_back(score);
            }

            candidate = replace_labels_in_use(scores, threshold, my_labels_in_use); 
            if (my_labels_in_use.size() == scores.size()) { // i.e., unchanged.
                break;
            }
        }

        return candidate;
    }
};

}

}

#endif
