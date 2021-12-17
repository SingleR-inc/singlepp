#ifndef SINGLEPP_MARKERS_HPP
#define SINGLEPP_MARKERS_HPP

#include <vector>
#include <algorithm>
#include <unordered_set>

namespace singlepp {

typedef std::vector<std::vector<std::vector<int> > > Markers;

template<typename Id>
void define_subsets(size_t mat_n, const Id* mat_id, size_t ref_n, const Id* ref_id, Markers& markers, int top) {
    // Define an intersection of identifiers.
    std::unordered_set<Id> ;
}

class Markers {
    virtual ~Markers() {}
    virtual size_t size() const = 0;
    virtual std::pair<size_t, const int*> get(size_t, size_t) const = 0;

    void combine(size_t n, const int* labels, std::vector<int>& output) const {
        std::unordered_set<int> store;

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (i != j) {
                    auto current = this->get(labels[i], labels[j]);
                    output.insert(current.second, current.second + current.first);
                }
            }
        }

        output.clear();
        for (auto s : store) {
            output.push_back(s);
        }
        std::sort(output.begin(), output.end());

        return;
    }

    void combine(std::vector<int>& output) const {
        std::unordered_set<int> store;

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (i != j) {
                    auto current = this->get(i, j);
                    output.insert(current.second, current.second + current.first);
                }
            }
        }

        output.clear();
        for (auto s : store) {
            output.push_back(s);
        }
        std::sort(output.begin(), output.end());

        return;
    }

    FullMarkers subset(size_t n, const int* subset, bool pr) const {
        std::vector<std::vector<std::vector<int> > > subsetted(this->size());

        for (size_t i = 0; i < this->size(); ++i) {
            subsetted[i].resize(this->size());
            for (size_t j = 0; j < this->size(); ++j) {
                if (i == j) {
                    continue;
                }

                auto current = this->get(i, j);
                auto& output = subsetted[i][j];
                auto ptr = current.second;
                size_t counter = 0;

                for (size_t k = 0; k < current.first; ++k, ++ptr) {
                    while (counter < n && subset[counter] < *ptr) {
                        ++counter;
                    }
                    if (counter == n) {
                        break;
                    }
                    if (subset[counter] == *ptr) {
                        output.push(counter);
                    }
                }
            }
        }

        return FullMarkers(std::move(subsetted));
    }
};

class LightMarkers : public Markers {
public:
    LightMarkers(std::vector<std::vector<std::pair<size_t, const int*> > > m) : ranked_markers(std::move(m)) {}

    size_t size() const {
        return ranked_markers.size();
    }

    std::pair<size_t, const int*> get(size_t i, size_t j) const {
        return ranked_markers[i][j];
    }
private:
    std::vector<std::vector<std::pair<size_t, const int*> > > ranked_markers;
};

class FullMarkers : public Markers {
public:
    FullMarkers(std::vector<std::vector<std::vector<int> > > m) : ranked_markers(std::move(m)) {}

    size_t size() const {
        return ranked_markers.size();
    }

    std::pair<size_t, const int*> get(size_t i, size_t j) const {
        const auto& current = ranked_markers[i][j];
        return std::make_pair<size_t, const int*>(current.size(), current.data());
    }

    void subset_in_place(size_t n, const int* subset) {
        for (size_t i = 0; i < this->size(); ++i) {
            for (size_t j = 0; j < this->size(); ++j) {
                if (i == j) {
                    continue;
                }

                auto& current = ranked_markers[i][j];
                size_t counter = 0;
                size_t i = 0;

                for (size_t k = 0; k < current.size(); ++k) {
                    while (counter < n && subset[counter] < current[k]) {
                        ++counter;
                    }
                    if (counter == n) {
                        break;
                    }
                    if (subset[counter] == current[k]) {
                        // 'i' is always at or after 'k', so this should be harmless.
                        current[i] = counter;
                        ++i;
                    }
                }

                current.resize(i);
            }
        }

        return;
    }
private:
    std::vector<std::vector<std::vector<int> > > ranked_markers;
};

}

#endif
