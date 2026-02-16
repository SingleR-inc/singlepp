#ifndef SINGLEPP_L2_HPP
#define SINGLEPP_L2_HPP

#include <cmath>
#include <cassert>

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
Float_ get_sparse_remapping(const SparseScaled<Index_, Float_>& x) { return x.remapping; }

template<typename Index_, typename Float_, class SparseInput1_, typename SparseInput2_>
Float_ sparse_l2(const Index_ num_markers, const SparseInput1_& vec1, std::vector<Float_>& remapping1, const SparseInput2_& vec2) {
    const auto num1 = get_sparse_num(vec1);
    const auto zero1 = get_sparse_zero(vec1);
    const auto num2 = get_sparse_num(vec2);
    const auto zero2 = get_sparse_zero(vec2);
    assert(sanisizer::is_greater_than_or_equal(num_markers, num1));
    assert(sanisizer::is_greater_than_or_equal(num_markers, num2));
    constexpr auto sentinel = define_sentinel<Float_>();

    Float_ l2 = 0;
    Index_ both = 0;

    for (Index_ i2 = 0; i2 < num2; ++i2) {
        const auto val2 = get_sparse_value(vec2, i2);
        const auto idx2 = get_sparse_index(vec2, i2);
        auto& entry = remapping1[idx2];
        if (entry == sentinel) {
            const Float_ delta = val2 - zero1;
            l2 += delta * delta;
        } else {
            const Float_ delta = val2 - entry;
            l2 += delta * delta;
            entry = sentinel;
            ++both;
        }
    }

    for (Index_ i1 = 0; i1 < num1; ++i1) {
        const auto val1 = get_sparse_value(vec1, i1);
        const auto idx1 = get_sparse_index(vec1, i1);
        auto& entry = remapping1[idx1];
        if (entry != sentinel) {
            const Float_ delta = val1 - zero2;
            l2 += delta * delta;
        } else {
            entry = val1;
        }
    } 

    const Float_ delta = zero1 - zero2;
    l2 += (num_markers - num1 - (num2 - both)) * (delta * delta);
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
