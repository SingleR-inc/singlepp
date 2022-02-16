#ifndef SINGLEPP_BUILD_INDICES_HPP
#define SINGLEPP_BUILD_INDICES_HPP

#include <vector>
#include <memory>
#include "knncolle/knncolle.hpp"
#include "tatami/tatami.hpp"

#include "process_features.hpp"
#include "scaled_ranks.hpp"

namespace singlepp {

inline size_t get_nlabels(size_t n, const int* labels) { 
    if (n == 0) {
        throw std::runtime_error("reference dataset must have at least one column");
    }
    return *std::max_element(labels, labels + n) + 1;
}

struct Reference {
    std::vector<double> data;
    std::shared_ptr<knncolle::Base<int, double> > index;
};

template<class Builder>
std::vector<Reference> build_indices(const tatami::Matrix<double, int>* ref, const int* labels, const std::vector<int>& subset, const Builder& build) {
    size_t NC = ref->ncol();
    size_t nlabels = get_nlabels(NC, labels);
    std::vector<int> label_count(nlabels);
    for (size_t i = 0; i < NC; ++i) {
        ++label_count[labels[i]];
    }

    size_t NR = subset.size();
    size_t first = 0, last = 0;
    if (NR) {
        first = *std::min_element(subset.begin(), subset.end());
        last = *std::max_element(subset.begin(), subset.end()) + 1;
    }

    std::vector<Reference> nnrefs(nlabels);
    std::vector<double*> starts(nlabels);
    for (size_t l = 0; l < nlabels; ++l) {
        if (label_count[l] == 0) {
            throw std::runtime_error(std::string("no entries for label ") + std::to_string(l));
        }
        nnrefs[l].data.resize(label_count[l] * NR);
        starts[l] = nnrefs[l].data.data();
    }

    std::vector<double*> scaled(NC);
    for (size_t c = 0; c < NC; ++c) {
        auto& ptr = starts[labels[c]];
        scaled[c] = ptr;
        ptr += NR;
    }

    #pragma omp parallel
    {
        RankedVector ranked(NR);
        std::vector<double> buffer(ref->nrow());
        auto wrk = ref->new_workspace(false);

        #pragma omp for
        for (size_t c = 0; c < NC; ++c) {
            auto ptr = ref->column(c, buffer.data(), first, last, wrk.get());
            for (size_t r = 0; r < NR; ++r) {
                ranked[r].first = ptr[subset[r] - first];
                ranked[r].second = r;
            }
            scaled_ranks(NR, ranked, scaled[c]);
        }
    }

    #pragma omp parallel for
    for (size_t l = 0; l < nlabels; ++l) {
        nnrefs[l].index = build(NR, label_count[l], nnrefs[l].data.data());
    }

    return nnrefs;
}

}

#endif