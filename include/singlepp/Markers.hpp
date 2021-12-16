#ifndef SINGLEPP_MARKERS_HPP
#define SINGLEPP_MARKERS_HPP

#include <vector>
#include <algorithm>

namespace singlepp {

class Markers {
public:
    Markers(std::vector<std::pair<size_t, int*> > m) : ranked_markers(std::move(m)) {}

    size_t size() const {
        return ranked_markers.size();
    }

    const auto& operator[](size_t l) const {
        return ranked_markers[l];        
    }

    void combine(size_t n, const int* chosen, std::vector<int>& output) {
        output.clear();
        for (size_t i = 0; i < n; ++i) {
            const auto& current = ranked_markers[i];
            output.insert(output.end(), current.second, current.second + current.first);
        }

        if (output.size()) {
            std::sort(output.begin(), output.end());
            size_t counter = 0;
            int last = output[0];
            for (size_t i = 1; i < output.size(); ++i) {
                if (output[i] != last) {
                    output[counter] = output[i];
                    ++counter;
                    last = output[i];
                }
            }
            output.resize(counter);
        }

        return;
    }
private:
    std::vector<std::pair<size_t, int*> > ranked_markers;
};

}

#endif
