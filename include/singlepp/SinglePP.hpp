#ifndef SINGLEPP_SINGLEPP_HPP
#define SINGLEPP_SINGLEPP_HPP

#include "knncolle/knncolle.hpp"
#include "build_indices.hpp"
#include "annotate_cells.hpp"
#include "process_features.hpp"

#include <vector> 
#include <stdexcept>

namespace singlepp {

class SinglePP {
public:
    struct Defaults {
        static constexpr double quantile = 0.2;

        static constexpr double fine_tune_threshold =0.05;

        static constexpr bool fine_tune = true;

        static constexpr int top = 20;

        static constexpr bool approximate = false;
    };

private:
    double quantile;
    double fine_tune_threshold;
    bool fine_tune;
    int top;
    bool approximate;

public:
    SinglePP& set_quantile(double q = Defaults::quantile) {
        quantile = q;
        return *this;
    }

    SinglePP& set_fine_tune_threshold(double t = Defaults::fine_tune_threshold) {
        fine_tune_threshold = t;
        return *this;
    }

    SinglePP& set_fine_tune(bool f = Defaults::fine_tune) {
        fine_tune = f;
        return *this;
    }

    SinglePP& set_top(int t = Defaults::top) {
        top = t;
        return *this;
    }

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
        std::cout << "Subsetted: " << subset.size() << "\t" << subset.front() << " -> " << subset.back() << std::endl;
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

        std::cout << "Simple references:" << std::endl;
        for (size_t s = 0; s < prebuilt.references.size(); ++s) {
            const auto& current = prebuilt.references[s];
            std::cout << s << "\t" << current.data.size() << "\t";
            for (int i = 0; i < 10; ++i) {
                std::cout << current.data[i] << ", ";
            }
            std::cout << std::endl;
        }

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

        std::cout << "Subsetted: " << pairs.second.size() << "\t";
        for (int i = 0; i < 10; ++i) {
            std::cout << pairs.second[i] << ", ";
        }
        std::cout << std::endl;

        std::cout << "Intersect references:" << std::endl;
        for (size_t s = 0; s < subref.size(); ++s) {
            const auto& current = subref[s];
            std::cout << s << "\t" << current.data.size() << "\t";
            for (int i = 0; i < 10; ++i) {
                std::cout << current.data[i] << ", ";
            }
            std::cout << std::endl;
        }

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
