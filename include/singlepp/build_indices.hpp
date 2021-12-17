#ifndef SINGLEPP_MATRICES_TO_INDICES_HPP
#define SINGLEPP_MATRICES_TO_INDICES_HPP

#include <vector>
#include <memory>
#include "knncolle/knncolle.hpp"

#include "Markers.hpp"

namespace singlepp {

template<class Mat, class Builder>
std::vector<std::shared_ptr<knncolle::Base<int, double> > > matrices_to_indices(const std::vector<Mat*>& ref, const Builder& build) {
    std::vector<std::shared_ptr<knncolle::Base<int, double> > > nnrefs(ref.size());
    std::vector<double> indexable; 

    for (size_t g = 0; g < ref.size(); ++g) {
        auto curref = ref[g];
        size_t NR = curref->nrow(); // should be the same.
        size_t NC = curref->ncol();
        indexable.resize(NR * NC);

        for (size_t c = 0; c < NC; ++c) {
            curref->column_copy(c, indexable.data() + c * NR);
        }
        nnrefs[g] = build(NR, NC, indexable.data());
    }

    return nnrefs;
}

template<class Index>
std::vector<const Index*> retrieve_index_pointers(const std::vector<std::shared_ptr<Index> >& nnrefs) {
    std::vector<const Index*> nnref_ptrs(nnrefs.size());
    for (size_t g = 0; g < nnrefs.size(); ++g) {
        nnref_ptrs[g] = nnrefs[g].get();
    }
    return nnref_ptrs;
}


}

#endif
