#ifndef SINGLEPP_L2_HPP
#define SINGLEPP_L2_HPP

#include <cmath>
#include <cassert>
#include <limits>
#include <vector>
#include <algorithm>

#include "utils.hpp"
#include "scaled_ranks.hpp"

namespace singlepp {

template<typename Float_>
constexpr Float_ define_sentinel() {
    return std::numeric_limits<Float_>::infinity();
}

template<typename Index_, typename Float_>
Float_ dense_l2(const Index_ num_markers, const Float_* vec1, const Float_* vec2) {
    Float_ l2 = 0;
    for (Index_ d = 0; d < num_markers; ++d) {
        const Float_ delta = vec1[d] - vec2[d]; 
        l2 += delta * delta;
    }
    return l2;
}

template<typename Index_, typename Float_>
Index_ get_sparse_num(const SparseScaled<Index_, Float_>& x) { return x.nonzero.size(); }

template<typename Index_, typename Float_>
Index_ get_sparse_index(const SparseScaled<Index_, Float_>& x, const Index_ i) { return x.nonzero[i].first; }

template<typename Index_, typename Float_>
Float_ get_sparse_value(const SparseScaled<Index_, Float_>& x, const Index_ i) { return x.nonzero[i].second; }

template<typename Index_, typename Float_>
Float_ get_sparse_zero(const SparseScaled<Index_, Float_>& x) { return x.zero; }

template<typename Index_, typename Float_>
void check_sparse_index_sorted_and_unique(const SparseScaled<Index_, Float_>& x) { assert(is_sorted_unique(x.nonzero.size(), x.nonzero.data())); }

template<typename Index_, typename Float_, class SparseInput1_, typename SparseInput2_>
Float_ sparse_l2(const Index_ num_markers, const SparseInput1_& query_vec, std::vector<Float_>& query_remapping, const SparseInput2_& ref_vec) {
    const auto num_query = get_sparse_num(query_vec);
    const auto zero_query = get_sparse_zero(query_vec);
    const auto num_ref = get_sparse_num(ref_vec);
    const auto zero_ref = get_sparse_zero(ref_vec);
    assert(sanisizer::is_greater_than_or_equal(num_markers, num_query));
    assert(sanisizer::is_greater_than_or_equal(num_markers, num_ref));
    constexpr auto sentinel = define_sentinel<Float_>();

    // Sortedness is not strictly necessary for correctness, but we want it for more efficient memory access when restoring 'query_remapping'.
    // Sortdness for 'query_vec' is cheap as we do it once per query and this can be re-used across many reference profiles; 
    // by comparison, sorting 'ref_vec' would be too expensive as it would need to be done for each reference profile.
    check_sparse_index_sorted_and_unique(query_vec);

    Float_ l2 = 0;
    Index_ both = 0;

    for (Index_ ir = 0; ir < num_ref; ++ir) {
        const auto val_ref = get_sparse_value(ref_vec, ir);
        const auto idx_ref = get_sparse_index(ref_vec, ir);
        auto& entry = query_remapping[idx_ref];
        if (entry == sentinel) {
            const Float_ delta = val_ref - zero_query;
            l2 += delta * delta;
        } else {
            const Float_ delta = val_ref - entry;
            l2 += delta * delta;
            entry = sentinel;
            ++both;
        }
    }

    for (Index_ iq = 0; iq < num_query; ++iq) {
        const auto val_query = get_sparse_value(query_vec, iq);
        const auto idx_query = get_sparse_index(query_vec, iq);
        auto& entry = query_remapping[idx_query];
        if (entry != sentinel) {
            const Float_ delta = val_query - zero_ref;
            l2 += delta * delta;
        } else {
            entry = val_query;
        }
    } 

    const Float_ delta = zero_query - zero_ref;
    l2 += (num_markers - num_query - (num_ref - both)) * (delta * delta);
    return l2;
}

template<typename Index_, class SparseInput_, typename Float_>
void setup_sparse_l2_remapping(const Index_ num_markers, const SparseInput_& vec, std::vector<Float_>& remap_buffer) {
    assert(sanisizer::is_greater_than_or_equal(remap_buffer.size(), num_markers));
    std::fill_n(remap_buffer.data(), num_markers, define_sentinel<Float_>());
    const auto num = get_sparse_num(vec);
    for (Index_ i = 0; i < num; ++i) {
        remap_buffer[get_sparse_index(vec, i)] = get_sparse_value(vec, i);
    }
}

template<typename Index_, class SparseInput_, typename Float_>
void densify_sparse_vector(const Index_ num_markers, const SparseInput_& vec, std::vector<Float_>& buffer) {
    assert(sanisizer::is_greater_than_or_equal(buffer.size(), num_markers));
    std::fill_n(buffer.data(), num_markers, get_sparse_zero(vec));
    const auto num = get_sparse_num(vec);
    for (Index_ i = 0; i < num; ++i) {
        buffer[get_sparse_index(vec, i)] = get_sparse_value(vec, i);
    }
}

}

#endif
