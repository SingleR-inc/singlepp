#include <gtest/gtest.h>
#include "singlepp/annotate_cells_integrated.hpp"
#include "singlepp/train_integrated.hpp"
#include "mock_markers.h"
#include "spawn_matrix.h"
#include "fill_ranks.h"
#include "naive_method.h"

class FineTuneIntegratedTest : public ::testing::Test {
protected: 
    inline static std::vector<std::shared_ptr<tatami::Matrix<double, int> > > references;
    inline static std::vector<std::vector<int> > labels;
    inline static singlepp::TrainedIntegrated<int> trained;

    inline static size_t ngenes = 2000;
    inline static size_t nprofiles = 50;
    inline static size_t nrefs = 3;

    static void SetUpTestSuite() {
        if (references.size()) { 
            return;
        }

        std::vector<singlepp::TrainIntegratedInput<double, int, int> > inputs;
        references.reserve(nrefs);
        labels.reserve(nrefs);

        for (size_t r = 0; r < nrefs; ++r) {
            size_t seed = r * 1000;
            size_t nlabels = 3 + r;
            references.push_back(spawn_matrix(ngenes, nprofiles, seed));
            labels.push_back(spawn_labels(nprofiles, nlabels, seed * 2));
            auto current = mock_markers<int>(nlabels, 10, ngenes, seed * 3);

            singlepp::TrainSingleOptions topt;
            auto single_built = singlepp::train_single(*(references.back()), labels.back().data(), std::move(current), topt);
            inputs.push_back(singlepp::prepare_integrated_input(*(references.back()), labels.back().data(), single_built));
        }

        trained = singlepp::train_integrated(std::move(inputs), singlepp::TrainIntegratedOptions());
    }

protected:
    std::vector<int> reflabels_in_use;
    std::unordered_set<int> miniverse_tmp;
    std::vector<int> miniverse;
    std::vector<const int*> assigned;

protected:
    void SetUp() {
        assigned.resize(nrefs);
        for (size_t r = 0; r < nrefs; ++r) {
            assigned[r] = labels[r].data();
        }
    }
};

TEST_F(FineTuneIntegratedTest, EdgeCases) {
    singlepp::RankedVector<double, int> placeholder; 

    // Check early exit conditions, when there is one clear winner or all of
    // the references are equal (i.e., no contraction of the feature space).
    {
        std::vector<double> scores { 0.2, 0.5, 0.1 };
        ASSERT_EQ(scores.size(), nrefs);
        auto output = singlepp::fine_tune_integrated(0, placeholder, scores, trained, assigned, reflabels_in_use, miniverse_tmp, miniverse, workspace, 0.8, 0.05);
        EXPECT_EQ(output.first, 1);
        EXPECT_EQ(output.second, 0.3);

        std::fill(scores.begin(), scores.end(), 0.5);
        scores[0] = 0.51;
        output = singlepp::fine_tune_integrated(0, placeholder, scores, trained, assigned, reflabels_in_use, miniverse_tmp, miniverse, workspace, 0.8, 0.05);
        EXPECT_EQ(output.first, 0); // first entry of scores is maxed.
        EXPECT_FLOAT_EQ(output.second, 0.01);
    }

    // Check edge case when there is only a single reference, based on the length of 'scores'.
    {
        std::vector<double> scores { 0.5 };
        auto output = singlepp::fine_tune_integrated(1, placeholder, scores, trained, assigned, reflabels_in_use, miniverse_tmp, miniverse, workspace, 0.8, 0.05);
        EXPECT_EQ(output.first, 0);
        EXPECT_TRUE(std::isnan(output.second));
    }
}

TEST_F(FineTuneIntegratedTest, ExactRecovery) {
    // Checking that we eventually pick up the correct reference, if the input
    // profile is identical to one of the profiles in one of the references. We
    // set the quantile to 1 to guarantee a score of 1 from a correlation of 1.
    std::vector<double> buffer(ngenes);
    for (size_t r = 0; r < nrefs; ++r) {
        auto wrk = references[r]->dense_column(trained.universe);
        for (size_t c = 0; c < nprofiles; ++c) {
            auto vec = wrk->fetch(c, buffer.data());
            auto ranked = fill_ranks<int>(trained.universe.size(), vec);

            std::vector<double> scores(nrefs, 0.5);
            scores[(r + 1) % nrefs] = 0; // forcing another reference to be zero so that it actually does the fine-tuning.
            auto output = singlepp::internal::fine_tune_integrated<int>(c, ranked, scores, trained, assigned, reflabels_in_use, miniverse_tmp, miniverse, workspace, 1.0, 0.05);
            EXPECT_EQ(output.first, r);

            std::fill(scores.begin(), scores.end(), 0.5);
            scores[r] = 0; // forcing it to match to some other reference. 
            auto output2 = singlepp::internal::fine_tune_integrated<int>(c, ranked, scores, trained, assigned, reflabels_in_use, miniverse_tmp, miniverse, workspace, 1.0, 0.05);
            EXPECT_NE(output2.first, r);
        }
    }
}

TEST_F(FineTuneIntegratedTest, ExactRecoveryIntersected) {
    // Same as above, but each reference is now limited to its own marker genes.
    // This checks that the fine-tuning works correctly with intersections.
    auto tcopy = trained;
    for (size_t r = 0; r < nrefs; ++r) {
        tcopy.check_availability[r] = true;
        for (const auto& x : tcopy.markers[r]) {
            tcopy.available[r].insert(x.begin(), x.end());
        }
    }

    std::vector<double> buffer(ngenes);
    for (size_t r = 0; r < nrefs; ++r) {
        auto wrk = references[r]->dense_column(trained.universe);
        for (size_t c = 0; c < nprofiles; ++c) {
            auto vec = wrk->fetch(c, buffer.data());
            auto ranked = fill_ranks<int>(trained.universe.size(), vec);

            std::vector<double> scores(nrefs, 0.5);
            scores[(r + 1) % nrefs] = 0; // forcing another reference to be zero so that it actually does the fine-tuning.
            auto output = singlepp::internal::fine_tune_integrated<int>(c, ranked, scores, trained, assigned, reflabels_in_use, miniverse_tmp, miniverse, workspace, 1.0, 0.05);
            EXPECT_EQ(output.first, r);

            std::fill(scores.begin(), scores.end(), 0.5);
            scores[r] = 0; // forcing it to match to some other reference. 
            auto output2 = singlepp::internal::fine_tune_integrated<int>(c, ranked, scores, trained, assigned, reflabels_in_use, miniverse_tmp, miniverse, workspace, 1.0, 0.05);
            EXPECT_NE(output2.first, r);
        }
    }
}
