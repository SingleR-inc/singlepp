#ifndef SINGLEPP_CHOOSE_MARKERS_HPP
#define SINGLEPP_CHOOSE_MARKERS_HPP

#include "Markers.hpp"
#include <vector>
#include <cmath>

namespace singlepp {

class ChooseMarkers {
public:
    struct Defaults {
        static constexpr int number = -1;
    };

private:
    int number = Defaults::number;

public:
    ChooseMarkers& set_number(int n = Defaults::number) {
        number = n;
        return *this;
    }

public:
    template<class Matrix>
    Markers run(const std::vector<const Matrix*>& references, const std::vector<const int*>& labels) const {
        size_t nrefs = references.size();
        if (nrefs != labels.size()) {
            throw std::runtime_error("'references' and 'labels' should have the same length");
        }
        if (nrefs == 0) {
            throw std::runtime_error("'references' should contain at least one entry");
        }
        size_t ngenes = nrefs.front()->nrow();

        // Determining the total number of labels.
        int nlabels = 0;
        for (size_t r = 0; r < nrefs; ++r) {
            size_t ncols = references[r]->ncol();
            auto curlab = labels[r];
            for (size_t c = 0; c < ncols; ++c) {
                if (nlabels <= curlab[c]) {
                    nlabels = curlab[c] + 1;
                }
            }
        }

        // Generating mappings.
        std::vector<std::vector<int> > labels_to_index(nrefs, std::vector<int>(nlabels, -1));
        for (size_t r = 0; r < nrefs; ++r) {
            size_t ncols = references[r]->ncol();
            auto curlab = labels[r];
            auto& current = labels_to_index[r];
            for (size_t c = 0; c < ncols; ++c) {
                auto& dest = current[curlab[c]];
                if (dest != -1) {
                    throw std::runtime_error("each label should correspond to no more than one column in each reference");
                }
                current[curlab[c]] = c;
            }
        }

        // Generating pairs for compute; this sacrifices some memory for convenience.
        std::vector<std::pair<int, int> > pairs;
        {
            std::unordered_set<std::pair<int, int> > pairs0;
            for (size_t r = 0; r < nrefs; ++r) {
                size_t ncols = references[r]->ncol();
                auto curlab = labels[r];
                for (size_t c1 = 0; c1 < ncols; ++c1) {
                    for (size_t c2 = 0; c2 < c1; ++c2) {
                        pairs0.emplace_back(curlab[c1], curlab[c2]);
                    }
                }
            }
            pairs.insert(pairs.end(), pairs0.begin(), pairs0.end());
            std::sort(pairs.begin(), pairs.end());
        }
        size_t npairs = pairs.size();

        Markers output(nlabels, std::vector<std::vector<int> >(nlabels));

        int actual_number = ngenes;
        if (number < 0) {
            actual_number = std::round(500.0 * std::pow(2.0/3.0, std::log(static_cast<double>(nlabels)) / std::log(2.0)));
        } else if (number < actual_number) {
            actual_number = number;
        }

#ifndef SINGLEPP_CUSTOM_PARALLEL
        #pragma omp parallel
        {
#else
        SINGLEPP_CUSTOM_PARALLEL(npairs, [&](size_t start, size_t end) -> void {
#endif
            
            std::vector<std::pair<double, int> > sorter(ngenes);
            std::vector<std::shared_ptr<tatami::Workspace> > rworks(nref), lworks(nref);
            std::vector<Matrix::value_type> rbuffer(ngenes), lbuffer(ngenes);

#ifndef SINGLEPP_CUSTOM_PARALLEL
            #pragma omp for
            for (size_t p = 0; p < npairs; ++p) {
#else
            for (size_t p = start; p < end; ++p) {
#endif

                auto curleft = pairs[p].first;
                auto curright = pairs[p].second;

                auto sIt = sorter.begin();
                for (int g = 0; g < ngenes; ++g, ++sIt) {
                    sIt->first = 0;
                    sIt->second = g;
                }

                for (size_t i = 0; i < nref; ++i) {
                    const auto& curavail = labels_to_index[i];
                    auto lcol = curavail[curleft];
                    auto rcol = curavail[curright];
                    if (lcol == -1 || rcol == -1) {
                        continue;                            
                    }

                    auto lptr = refs[i]->column(lcol, lbuffer.data(), lworks[i].get());
                    auto rptr = refs[i]->column(rcol, rbuffer.data(), rworks[i].get());

                    auto sIt = sorter.begin();
                    for (int g = 0; g < ngenes; ++g, ++lptr, ++rptr) {
                        sIt->second += *lptr - *rptr;                        
                    }
                }

                // partial sort is guaranteed to be stable due to the second index resolving ties.
                std::partial_sort(sorter.begin(), sorter.begin() + de_n, sorter.end());

                std::vector<int> stuff;
                stuff.reserve(actual_number);
                for (int g = 0; g < actual_number && sorter[g].first < 0; ++g) { // only keeping those with positive log-fold changes (negative after reversing).
                    stuff.push_back(sorter[g].second); 
                }
                store[curleft][curright] = std::move(stuff);
            }

#ifndef SINGLEPP_CUSTOM_PARALLEL
        }
#else    
        });
#endif        
        return output;
    }
};

}

#endif
