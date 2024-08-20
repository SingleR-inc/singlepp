#ifndef SINGLEPP_ANNOTATE_CELLS_HPP
#define SINGLEPP_ANNOTATE_CELLS_HPP

#include "macros.hpp"

#include "tatami/tatami.hpp"

#include "scaled_ranks.hpp"
#include "SubsetSanitizer.hpp"
#include "build_indices.hpp"
#include "fine_tune.hpp"

#include <vector>
#include <algorithm>
#include <cmath>

namespace singlepp {

namespace internal {

template<typename Value_, typename Index_, typename Float_, typename Label_>
void annotate_cells_simple(
    const tatami::Matrix<Value_, Index_>& mat,
    size_t num_subset,
    const Index_* subset,
    const std::vector<PerLabelReference<Index_, Float_> >& ref,
    const Markers<Index_>& markers,
    Float_ quantile,
    bool fine_tune,
    Float_ threshold,
    Label_* best, 
    std::vector<Float_*>& scores,
    Float_* delta,
    int num_threads)
{
    // Figuring out how many neighbors to keep and how to compute the quantiles.
    const size_t num_labels = ref.size();
    std::vector<Index_> search_k(num_labels);
    std::vector<std::pair<Float_, Float_> > coeffs(num_labels);

    for (size_t r = 0; r < num_labels; ++r) {
        Float_ denom = static_cast<Float_>(ref[r].index->num_observations()) - 1;
        Float_  prod = denom * (1 - quantile);
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

    std::vector<Index_> subcopy(subset, subset + num_subset);
    SubsetSanitizer<Index_> subsorted(subcopy);

    tatami::parallelize([&](size_t, Index_ start, Index_ length) -> void {
        std::vector<Value_> buffer(num_subset);
        tatami::VectorPtr<Index_> mock_ptr(tatami::VectorPtr<Index_>{}, &(subsorted.extraction_subset()));
        auto wrk = tatami::consecutive_extractor<false>(mat, false, start, length, std::move(mock_ptr));

        std::vector<std::unique_ptr<knncolle::Searcher<Index_, Float_> > > searchers(num_labels);
        for (size_t r = 0; r < num_labels; ++r) {
            searchers[r] = ref[r].index->initialize();
        }
        std::vector<Float_> distances;

        RankedVector<Value_, Index_> vec;
        vec.reserve(num_subset);
        FineTuner<Label_, Index_, Float_, Value_> ft;
        std::vector<Float_> curscores(num_labels);

        for (Index_ c = start, end = start + length; c < end; ++c) {
            auto ptr = wrk->fetch(buffer.data());
            subsorted.fill_ranks(ptr, vec);
            scaled_ranks(vec, buffer.data()); // 'buffer' can be re-used for output here, as all data is already extracted to 'vec'.

            curscores.resize(num_labels);
            for (size_t r = 0; r < num_labels; ++r) {
                size_t k = search_k[r];
                auto current = searchers[r]->search(buffer.data(), k, NULL, &distances);

                Float_ last = distances[k - 1];
                last = 1 - 2 * last * last;
                if (k == 1) {
                    curscores[r] = last;
                } else {
                    Float_ next = distances[k - 2];
                    next = 1 - 2 * next * next;
                    curscores[r] = coeffs[r].first * next + coeffs[r].second * last;
                }

                if (scores[r]) {
                    scores[r][c] = curscores[r];
                }
            }

            if (!fine_tune) {
                auto top = std::max_element(curscores.begin(), curscores.end());
                best[c] = top - curscores.begin();
                if (delta) {
                    if (curscores.size() > 1) {
                        Float_ topscore = *top;
                        *top = -100; // replace max value with a negative value to find the second-max value easily.
                        delta[c] = topscore - *std::max_element(curscores.begin(), curscores.end());
                    } else {
                        delta[c] = std::numeric_limits<Float_>::quiet_NaN();
                    }
                }
            } else {
                auto tuned = ft.run(num_subset, vec, ref, markers, curscores, quantile, threshold);
                best[c] = tuned.first;
                if (delta) {
                    delta[c] = tuned.second;
                }
            }
        }

    }, mat->ncol(), num_threads);

    return;
}

}

}

#endif
