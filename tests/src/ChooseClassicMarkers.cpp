#include <gtest/gtest.h>
#include "custom_parallel.h"

#include "singlepp/ChooseClassicMarkers.hpp"
#include "spawn_matrix.h"

class ChooseClassicMarkersTest : public ::testing::TestWithParam<int> {};

TEST_P(ChooseClassicMarkersTest, Simple) { 
    size_t ngenes = 100;
    size_t nlabels = 10;
    int requested = GetParam();

    auto mat = spawn_matrix(ngenes, nlabels, 1234 * requested);
    std::vector<int> groupings(nlabels);
    std::iota(groupings.begin(), groupings.end(), 0);

    singlepp::ChooseClassicMarkers mrk;
    mrk.set_number(requested);
    auto output = mrk.run(mat.get(), groupings.data());

    // Comparing against a naive implementation.
    EXPECT_EQ(output.size(), nlabels);
    for (size_t l = 0; l < nlabels; ++l) {
        const auto& current = output[l];
        EXPECT_EQ(current.size(), nlabels);
        auto left = mat->column(l);

        for (size_t l2 = 0; l2 < nlabels; ++l2) {
            const auto& markers = current[l2];
            int nmarkers = markers.size();

            if (l == l2) {
                EXPECT_TRUE(markers.empty());
                continue;
            } else if (requested >= 0) {
                EXPECT_TRUE(nmarkers <= requested);
            }

            auto right = mat->column(l2);
            std::vector<std::pair<double, int> > sorted;
            for (size_t i = 0; i < right.size(); ++i) {
                sorted.emplace_back(right[i] - left[i], i);
            }

            std::sort(sorted.begin(), sorted.end());
            for (int g = 0; g < nmarkers; ++g) {
                EXPECT_TRUE(sorted[g].first < 0);
                EXPECT_EQ(sorted[g].second, markers[g]);
            }

            if (nmarkers < requested && nmarkers + 1 < static_cast<int>(ngenes)) {
                EXPECT_TRUE(sorted[nmarkers].first >= 0); // early rejection.
            }
        }
    }
}

INSTANTIATE_TEST_CASE_P(
    ChooseClassicMarkers,
    ChooseClassicMarkersTest,
    ::testing::Values(-1, 5, 10, 20, 100, 1000) // number of top genes.
);
