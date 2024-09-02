#ifndef SINGLEPP_BUILD_INDICES_HPP
#define SINGLEPP_BUILD_INDICES_HPP

#include "defs.hpp"

#include "knncolle/knncolle.hpp"
#include "tatami/tatami.hpp"

#include "scaled_ranks.hpp"
#include "SubsetSanitizer.hpp"

#include <vector>
#include <memory>
#include <algorithm>

namespace singlepp {

namespace internal {

template<typename Label_>
size_t get_nlabels(size_t n, const Label_* labels) { 
    if (n == 0) {
        throw std::runtime_error("reference dataset must have at least one column");
    }
    return static_cast<size_t>(*std::max_element(labels, labels + n)) + 1;
}

template<typename Index_, typename Float_>
struct PerLabelReference {
    std::vector<RankedVector<Index_, Index_> > ranked;
    std::shared_ptr<knncolle::Prebuilt<Index_, Index_, Float_> > index;
};

template<typename Value_, typename Index_, typename Label_, typename Float_>
std::vector<PerLabelReference<Index_, Float_> > build_indices(
    const tatami::Matrix<Value_, Index_>& ref,
    const Label_* labels,
    const std::vector<Index_>& subset,
    const knncolle::Builder<knncolle::SimpleMatrix<Index_, Index_, Float_>, Float_>& builder,
    int num_threads)
{
    size_t NR = subset.size();
    size_t NC = ref.ncol();
    size_t nlabels = get_nlabels(NC, labels);

    std::vector<size_t> label_count(nlabels);
    std::vector<size_t> label_offsets(NC);
    for (size_t i = 0; i < NC; ++i) {
        auto& lcount = label_count[labels[i]];
        label_offsets[i] = lcount;
        ++lcount;
    }

    std::vector<PerLabelReference<Index_, Float_> > nnrefs(nlabels);
    std::vector<std::vector<Float_> > nndata(nlabels);
    for (size_t l = 0; l < nlabels; ++l) {
        if (label_count[l] == 0) {
            throw std::runtime_error(std::string("no entries for label ") + std::to_string(l));
        }
        nnrefs[l].ranked.resize(label_count[l]);
        nndata[l].resize(label_count[l] * NR); // already size_t, no need to cast to avoid overflow.
    }

    SubsetSanitizer<Index_> subsorter(subset);
    tatami::VectorPtr<Index_> subset_ptr(tatami::VectorPtr<Index_>{}, &(subsorter.extraction_subset()));

    tatami::parallelize([&](int, Index_ start, Index_ len) {
        auto ext = tatami::consecutive_extractor<false>(&ref, false, start, len, subset_ptr);
        std::vector<Value_> buffer(NR);
        RankedVector<Value_, Index_> ranked;
        ranked.reserve(NR);

        for (Index_ c = start, end = start + len; c < end; ++c) {
            auto ptr = ext->fetch(buffer.data());
            subsorter.fill_ranks(ptr, ranked); 

            auto curlab = labels[c];
            auto curoff = label_offsets[c];
            auto scaled = nndata[curlab].data() + curoff * NR; // these are already size_t's, so no need to cast.
            scaled_ranks(ranked, scaled); 

            // Storing as a pair of ints to save space; as long
            // as we respect ties, everything should be fine.
            auto& stored_ranks = nnrefs[curlab].ranked[curoff];
            stored_ranks.reserve(ranked.size());
            simplify_ranks(ranked, stored_ranks);
        }
    }, ref.ncol(), num_threads);

    tatami::parallelize([&](int, size_t start, size_t len) {
        for (size_t l = start, end = start + len; l < end; ++l) {
            nnrefs[l].index = builder.build_shared(knncolle::SimpleMatrix<Index_, Index_, Float_>(NR, label_count[l], nndata[l].data()));

            // Trying to free the memory as we go along, to offset the copying
            // of nndata into the memory store owned by the knncolle index.
            nndata[l].clear();
            nndata[l].shrink_to_fit();
        }
    }, nlabels, num_threads);

    return nnrefs;
}

}

}

#endif
