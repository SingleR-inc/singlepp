#ifndef SINGLEPP_INTEGRATOR_HPP
#define SINGLEPP_INTEGRATOR_HPP

#include "SinglePP.hpp"
#include "scaled_ranks.hpp"

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>

namespace singlepp {

struct IntegratedReference {
    bool check_availability = false;
    std::unordered_set<int> available;
    std::vector<std::vector<int> > markers;
    std::vector<std::vector<RankedVector<int, int> > > ranked;
};

class Integrator {
    std::vector<const tatami::Matrix<double, int>*> stored_matrices;
    std::vector<const int*> stored_labels;
    std::vector<IntegratedReference> references;
    std::vector<std::unordered_map<int, int> > gene_mapping;

private:
    void add(const tatami::Matrix<double, int>* ref, const int* labels, const Markers& old_markers, const std::vector<int>& subset) {
        stored_matrices.push_back(ref);
        stored_labels.push_back(labels);
        references.resize(references.size() + 1);

        // Adding the markers for each label.
        auto& new_markers = references.back().markers;
        for (size_t i = 0; i < old_markers.size(); ++i) {
            const auto& cur_old_markers = old_markers[i];

            std::unordered_set<int> unified;
            for (const auto& x : cur_old_markers) {
                unified.insert(x.begin(), x.end());                
            }

            new_markers.emplace_back(unified.begin(), unified.end());
            auto& cur_new_markers = new_markers.back();
            for (auto& y : cur_new_markers) {
                y = subset[y];
            }
        }

        gene_mapping.resize(gene_mapping.size() + 1); 
        return;
    }

public:
    void add(const tatami::Matrix<double, int>* ref, const int* labels, const SinglePP::Prebuilt& built) {
        add(ref, labels, built.markers, built.subset);
        return;
    }

public:
    template<typename Id>
    void add(size_t mat_nrow,
        const Id* mat_id,
        const tatami::Matrix<double, int>* ref, 
        const Id* ref_id,
        const int* labels, 
        const SinglePP::PrebuiltIntersection& built) 
    {
        add(ref, labels, built.markers, built.mat_subset);
        references.back().check_availability = true;

        auto intersection = intersect_features(mat_nrow, mat_id, ref->nrow(), ref_id);
        auto& mapping = gene_mapping.back();
        for (const auto& i : intersection) {
            mapping[i.first] = i.second;
        }
        return;
    }

public:
    std::vector<IntegratedReference> finish() {
        // Identify the global set of all genes that will be in use here.
        std::unordered_set<int> in_use_tmp;
        for (const auto& ref : references) {
            for (const auto& mrk : ref.markers) {
                in_use_tmp.insert(mrk.begin(), mrk.end());
            }
        }

        std::vector<int> in_use(in_use_tmp.begin(), in_use_tmp.end());
        std::sort(in_use.begin(), in_use.end());

        for (size_t r = 0; r < references.size(); ++r) {
            auto& curref = references[r];
            auto curlab = stored_labels[r];
            auto curmat = stored_matrices[r];

            size_t NR = curmat->ncol();
            size_t NC = curmat->ncol();
            size_t nlabels = curref.markers.size();

            std::vector<int> positions(NC);
            std::vector<int> samples_per_label(nlabels);
            for (size_t c = 0; c < NC; ++c) {
                auto& pos = samples_per_label[curlab[c]];
                positions[c] = pos;
                ++pos;
            }

            curref.ranked.resize(nlabels);
            for (size_t l = 0; l < nlabels; ++l) {
                curref.ranked[l].resize(samples_per_label[l]);
            }

            if (!curref.check_availability) {
                // If we don't need to check availability, this implies that 
                // the reference has 1:1 feature mapping to the test dataset.
                // In that case, we can proceed quite simply.
                size_t first = 0, last = 0;
                if (in_use.size()) {
                    first = in_use.front();
                    last = in_use.back() + 1;
                }

                #pragma omp parallel
                {
                    RankedVector<double, int> tmp_ranked;
                    tmp_ranked.reserve(in_use.size());
                    std::vector<double> buffer(NR);
                    auto wrk = curmat->new_workspace(false);

                    #pragma omp for
                    for (size_t c = 0; c < NC; ++c) {
                        auto ptr = curmat->column(c, buffer.data(), first, last, wrk.get());

                        tmp_ranked.clear();
                        for (auto u : in_use) {
                            tmp_ranked.emplace_back(ptr[u - first], u);
                        }
                        std::sort(tmp_ranked.begin(), tmp_ranked.end());

                        auto& final_ranked = curref.ranked[curlab[c]][positions[c]];
                        simplify_ranks(tmp_ranked, final_ranked);
                    }
                }

            } else {
                // If we do need to check availability, then we need to check
                // the mapping of test genes to their reference row indices.
                const auto& cur_mapping = gene_mapping[r];
                auto& cur_available = curref.available;
                std::unordered_map<int, int> remapping;
                remapping.reserve(in_use.size());
                size_t first = NR, last = 0;

                for (auto u : in_use) {
                    auto it = cur_mapping.find(u);
                    if (it == cur_mapping.end()) {
                        continue;
                    }

                    remapping[u] = it->second;
                    cur_available.insert(u);
                    if (it->second < first) {
                        first = it->second;
                    }
                    if (it->second > last) {
                        last = it->second;
                    }
                }
                last = std::max(last + 1, first);

                #pragma omp parallel
                {
                    RankedVector<double, int> tmp_ranked;
                    tmp_ranked.reserve(in_use.size());
                    std::vector<double> buffer(NR);
                    auto wrk = curmat->new_workspace(false);

                    #pragma omp for
                    for (size_t c = 0; c < NC; ++c) {
                        auto ptr = curmat->column(c, buffer.data(), first, last, wrk.get());

                        tmp_ranked.clear();
                        for (auto p : remapping) {
                            tmp_ranked.emplace_back(ptr[p.second - first], p.first);
                        }
                        std::sort(tmp_ranked.begin(), tmp_ranked.end());

                        auto& final_ranked = curref.ranked[curlab[c]][positions[c]];
                        simplify_ranks(tmp_ranked, final_ranked);
                    }
                }
            }
        }

        return std::move(references);
    }
};

}

#endif
