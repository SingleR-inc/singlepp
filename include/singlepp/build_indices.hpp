#ifndef SINGLEPP_BUILD_INDICES_HPP
#define SINGLEPP_BUILD_INDICES_HPP

#include <vector>
#include <memory>
#include "knncolle/knncolle.hpp"

#include "process_features.hpp"
#include "scaled_ranks.hpp"

namespace singlepp {

struct Reference {
    std::vector<double> data;
    std::shared_ptr<knncolle::Base<int, double> > index;
};

template<class Mat, class Builder>
std::vector<Reference> build_indices(const std::vector<int>& subset, const std::vector<Mat*>& ref, const Builder& build) {
    std::vector<Reference> nnrefs(ref.size());

    size_t NR = subset.size();
    size_t first = 0, last = 0;
    if (NR) {
        first = *std::min_element(subset.begin(), subset.end());
        last = *std::max_element(subset.begin(), subset.end());
    }

    #pragma omp parallel for
    for (size_t g = 0; g < ref.size(); ++g) {
        auto curref = ref[g];
        auto& nref = nnrefs[g];
        auto& indexable = nref.data;

        RankedVector ranked(NR);
        size_t NC = curref->ncol();
        indexable.resize(NR * NC);
        std::vector<double> buffer(curref->nrow());
        auto wrk = curref->new_workspace(false);

        for (size_t c = 0; c < NC; ++c) {
            auto ptr = curref->column(c, buffer.data(), first, last, wrk.get());
            for (size_t r = 0; r < NR; ++r) {
                ranked[r].first = ptr[subset[r] - first];
                ranked[r].second = r;
            }
            scaled_ranks(NR, ranked, indexable.data() + c * NR);
        }

        nref.index = build(NR, NC, indexable.data());
    }

    return nnrefs;
}

}

#endif
