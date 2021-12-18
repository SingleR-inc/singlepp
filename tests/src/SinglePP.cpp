#include <gtest/gtest.h>

#include "singlepp/annotate_cells.hpp"
#include "tatami/tatami.hpp"

TEST(AnnotateCellsTest, Stub) {
    std::vector<double> values(20);
    std::iota(values.begin(), values.end(), 0);
    auto mat = tatami::DenseColumnMatrix<double, int>(5, 4, values);
    
    auto ref = tatami::DenseColumnMatrix<double, int>(5, 4, values);
    std::vector<const tatami::Matrix<double, int>*> refs(2, &ref);

    singlepp::Markers markers(2);
    for (size_t i = 0; i < 2; ++i) {
        markers[i].resize(2);
    }
    markers[0][1].push_back(1);
    markers[0][1].push_back(3);
    markers[1][0].push_back(1);
    markers[1][0].push_back(2);

    std::vector<int> best(mat.ncol());
    std::vector<std::vector<double> > scores(refs.size(), std::vector<double>(mat.ncol()));
    std::vector<double*> score_ptrs(scores.size());
    for (size_t s = 0; s < scores.size(); ++s) {
        score_ptrs[s] = scores[s].data();
    }
    std::vector<double> delta(mat.ncol());

    singlepp::annotate_cells_simple(
        &mat, 
        refs, 
        [](size_t nr, size_t nc, const double* ptr) { return std::shared_ptr<knncolle::Base<int, double> >(new knncolle::VpTreeEuclidean<int, double>(nr, nc, ptr)); },
        markers,
        2,
        0.2,
        true,
        0.05,
        best.data(),
        score_ptrs,
        delta.data()
    );
}
