#ifndef SINGLEPP_SINGLEPP_HPP
#define SINGLEPP_SINGLEPP_HPP

#include "knncolle/knncolle.hpp"
#include "build_indices.hpp"
#include "annotate_cells.hpp"
#include "process_features.hpp"

#include <vector> 
#include <stdexcept>

/**
 * @file SinglePP.hpp
 *
 * @brief Defines the `SinglePP` class.
 */

namespace singlepp {

/**
 * @brief Automatically assign cell type labels based on an expression matrix.
 *
 * This implements the [**SingleR**](https://bioconductor.org/packages/SingleR) algorithm for automated annotation of single-cell RNA-seq data.
 * For each cell, we compute the Spearman rank correlation between that cell and the reference expression profiles.
 * This is done using only the subset of genes that are label-specific markers,
 * most typically the top genes from pairwise comparisons between each label's expression profiles.
 * For each label, we take the correlations involving that label's reference profiles and convert it into a score.
 * The label with the highest score is used as an initial label for that cell.
 *
 * For each cell, we apply fine-tuning iterations to improve the label accuracy by refining the feature space.
 * At each iteration, we find the subset of labels with scores that are close to the maximum score according to some threshold.
 * We recompute the scores based on the markers for this label subset, and we repeat the process until only one label is left in the subset or the subset is unchanged.
 * At the end of the iterations, the label with the highest score (or the only label, if just one is left) is used as the label for the cell.
 * This process aims to remove noise by eliminating irrelevant genes when attempting to distinguish closely related labels.
 * 
 * Each label's score is defined as a user-specified quantile of the distribution of correlations across all reference profiles assigned to that label.
 * (We typically consider a large quantile, e.g., the 80% percentile of the correlations.)
 * The use of a quantile avoids problems with differences in the number of reference profiles per label;
 * in contrast, just using the "top X correlations" would implicitly favor labels with more reference profiles.
 *
 * The choice of Spearman's correlation provides some robustness against batch effects when comparing reference and test datasets.
 * Only the relative expression _within_ each cell needs to be comparable, not their relative expression across cells.
 * As a result, it does not matter whether raw counts are supplied or log-transformed expression values, as the latter is a monotonic transformation of the latter (within each cell).
 * The algorithm is also robust to differences in technologies between reference and test profiles, though it is preferable to have like-for-like comparisons. 
 *
 * @see
 * Aran D et al. (2019). 
 * Reference-based analysis of lung single-cell sequencing reveals a transitional profibrotic macrophage.
 * _Nat. Immunol._ 20, 163-172
 */
class SinglePP {
public:
    /**
     * @brief Default parameters for annotation.
     */
    struct Defaults {
        /**
         * See `set_quantile()` for details.
         */
        static constexpr double quantile = 0.2;

        /**
         * See `set_fine_tune_threshold()` for details.
         */
        static constexpr double fine_tune_threshold = 0.05;

        /**
         * See `set_fine_tune()` for details.
         */
        static constexpr bool fine_tune = true;

        /**
         * See `set_top()` for details.
         */
        static constexpr int top = 20;

        /**
         * See `set_approximate()` for details.
         */
        static constexpr bool approximate = false;
    };

private:
    double quantile;
    double fine_tune_threshold;
    bool fine_tune;
    int top;
    bool approximate;

public:
    /**
     * @param q Quantile to use to compute a per-label score from the correlations.
     *
     * @return A reference to this `SinglePP` object.
     */
    SinglePP& set_quantile(double q = Defaults::quantile) {
        quantile = q;
        return *this;
    }

    /**
     * @param t Threshold to use to select the top-scoring subset of labels during fine-tuning.
     *
     * @return A reference to this `SinglePP` object.
     */
    SinglePP& set_fine_tune_threshold(double t = Defaults::fine_tune_threshold) {
        fine_tune_threshold = t;
        return *this;
    }

    /**
     * @param f Whether to perform fine-tuning.
     * This can be disabled for speed at the cost of accuracy.
     *
     * @return A reference to this `SinglePP` object.
     */
    SinglePP& set_fine_tune(bool f = Defaults::fine_tune) {
        fine_tune = f;
        return *this;
    }

    /**
     * @param t Number of top markers to use from each pairwise comparison between labels.
     *
     * @return A reference to this `SinglePP` object.
     */
    SinglePP& set_top(int t = Defaults::top) {
        top = t;
        return *this;
    }

    /**
     * @param a Whether to use an approximate method to quickly find the quantile.
     *
     * @return A reference to this `SinglePP` object.
     */
    SinglePP& set_approximate(bool a = Defaults::approximate) {
        approximate = a;
        return *this;
    }

private:
    template<class Mat>
    std::vector<Reference> build_internal(const std::vector<int>& subset, const std::vector<Mat*>& ref) { 
        std::vector<Reference> subref;
        if (approximate) {
            subref = build_indices(subset, ref, 
                [](size_t nr, size_t nc, const double* ptr) { 
                    return std::shared_ptr<knncolle::Base<int, double> >(new knncolle::AnnoyEuclidean<int, double>(nr, nc, ptr)); 
                }
            );
        } else {
            subref = build_indices(subset, ref, 
                [](size_t nr, size_t nc, const double* ptr) { 
                    return std::shared_ptr<knncolle::Base<int, double> >(new knncolle::VpTreeEuclidean<int, double>(nr, nc, ptr)); 
                }
            );
        }
        return subref;
    }

    template<class Mat>
    void check_references(const std::vector<Mat*>& ref) {
        if (ref.size()==0) {
            throw std::runtime_error("reference must contain at least one label");
        }
        size_t nr = ref[0]->nrow();
        for (auto r : ref) {
            if (r->nrow() != nr) {
                throw std::runtime_error("reference matrices must have the same number of rows");
            }
        }
    }

public:
    struct Prebuilt {
        Prebuilt(Markers m, std::vector<int> s, std::vector<Reference> r) : 
            markers(std::move(m)), subset(std::move(s)), references(std::move(r)) {}

        Markers markers;
        std::vector<int> subset;
        std::vector<Reference> references;
    };

    template<class Mat>
    Prebuilt build(const std::vector<Mat*>& ref, Markers markers) {
        check_references(ref);
        auto subset = subset_markers(markers, top);
        auto subref = build_internal(subset, ref);
        return Prebuilt(std::move(markers), std::move(subset), std::move(subref));
    }

public:
    void run(const tatami::Matrix<double, int>* mat, const Prebuilt& refs, int* best, std::vector<double*>& scores, double* delta) {
        annotate_cells_simple(mat, refs.subset, refs.references, refs.markers, quantile, fine_tune, fine_tune_threshold, best, scores, delta);
        return;
    }

    template<class Mat>
    void run(const tatami::Matrix<double, int>* mat, const std::vector<Mat*>& ref, Markers markers, int* best, std::vector<double*>& scores, double* delta) {
        auto prebuilt = build(ref, std::move(markers));
        run(mat, prebuilt, best, scores, delta);
        return;
    }

public:
    struct Results {
        Results(size_t ncells, size_t nlabels) : best(ncells), scores(nlabels, std::vector<double>(ncells)), delta(ncells) {}
        std::vector<int> best;
        std::vector<std::vector<double> > scores;
        std::vector<double> delta;

        std::vector<double*> scores_to_pointers() {
            std::vector<double*> output(scores.size());
            for (size_t s = 0; s < scores.size(); ++s) {
                output[s] = scores[s].data();
            }
            return output;
        };
    };

    Results run(const tatami::Matrix<double, int>* mat, const Prebuilt& refs) {
        size_t nlabels = refs.references.size();
        Results output(mat->ncol(), nlabels);
        auto scores = output.scores_to_pointers();
        run(mat, refs, output.best.data(), scores, output.delta.data());
        return output;
    }

    template<class Mat>
    Results run(const tatami::Matrix<double, int>* mat, const std::vector<Mat*>& ref, Markers markers) {
        auto prebuilt = build(ref, std::move(markers));
        return run(mat, prebuilt);
    }

public:
    template<class Id, class Mat>
    void run(const tatami::Matrix<double, int>* mat, const Id* mat_id, const std::vector<Mat*>& ref, const Id* ref_id, Markers markers, int* best, std::vector<double*>& scores, double* delta) {
        check_references(ref);
        auto intersection = intersect_features(mat->nrow(), mat_id, ref[0]->nrow(), ref_id);
        subset_markers(intersection, markers, top);
        auto pairs = unzip(intersection);
        auto subref = build_internal(pairs.second, ref);
        annotate_cells_simple(mat, pairs.first, subref, markers, quantile, fine_tune, fine_tune_threshold, best, scores, delta);
        return;
    }

    template<class Id, class Mat>
    Results run(const tatami::Matrix<double, int>* mat, const Id* mat_id, const std::vector<Mat*>& ref, const Id* ref_id, Markers markers) {
        Results output(mat->ncol(), ref.size());
        auto scores = output.scores_to_pointers();
        run(mat, mat_id, ref, ref_id, std::move(markers), output.best.data(), scores, output.delta.data());
        return output;
    }
};

}

#endif
