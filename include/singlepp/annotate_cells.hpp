#ifndef SINGLEPP_ANNOTATE_CELLS_HPP
#define SINGLEPP_ANNOTATE_CELLS_HPP

#include "tatami/tatami.h"
#include "knncolle/knncolle.hpp"
#include "scaled_ranks.hpp"

namespace singlepp {

inline void annotate_cells_simple(
    const tatami::Matrix<double, int>* mat, 
    const knncolle::Base<int, double>* ref,
    const int* labels,
    const Markers& markers,
    bool fine_tune,
    int* best, 
    std::vector<double*>& scores,
    double* delta) 
{
    const size_t NR = mat->nrow();
    const size_t NC = mat->ncol();

    #pragma omp parallel
    {
        std::vector<double> buffer(NR);
        ranked_vector vec;
        vec.reserve(NR);
        auto wrk = mat->new_workspace(false);

        #pragma omp for
        for (size_t c = 0; c < NC; ++c) {
            auto ptr = mat->column(c, buffer.data(), wrk.get());
            scaled_ranks(ptr, genes, vec, output);
        }
    }



}

}

#endif
