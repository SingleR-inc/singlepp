#include <gtest/gtest.h>

#include "singlepp/SinglePP.hpp"
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

    singlepp::SinglePP runner;
    auto output = runner.run(&mat, refs, markers);
}
