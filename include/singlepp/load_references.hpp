#ifndef SINGLEPP_LOADERS_HPP
#define SINGLEPP_LOADERS_HPP

#include <string>
#include <vector>
#include <cctype>

#include "Markers.hpp"
#include "buffin/parse_text_file.hpp"

#ifdef SINGLEPP_USE_ZLIB
#include "buffin/parse_gzip_file.hpp"
#include "buffin/parse_zlib_buffer.hpp"
#endif

/**
 * @file load_references.hpp
 *
 * @brief Load reference datasets from a few expected formats.
 */

namespace singlepp {

/** 
 * @cond
 */
struct LabelLoader {
    template<typename B>
    void add (const B* buffer, size_t n) {
        size_t i = 0;
        while (i < n) {
            if (buffer[i] == '\n') {
                if (!non_empty) {
                    throw std::runtime_error("label index must be an integer");
                }
                labels.push_back(current);
                current = 0;
                non_empty = false;
            } else if (std::isdigit(buffer[i])) {
                non_empty = true;
                current *= 10;
                current += (buffer[i] - '0');
            } else {
                throw std::runtime_error("label index must be an integer");
            }
            ++i;
        }
    }

    void finish() {
        if (non_empty) {
            labels.push_back(current);
        }
    }

    bool non_empty = false;
    int current = 0;
    std::vector<int> labels;
};
/** 
 * @endcond
 */

/**
 * @param path Path to a text file containing the labels.
 * @param buffer_size Size of the buffer to use when reading the file.
 *
 * @return Vector containing the label index for each reference profile.
 *
 * The file should contain one line per profile, containing an integer label index for that profile.
 * Label indices refer to another array containing the actual names of the labels.
 * The total number of lines should be equal to the number of profiles in the dataset.
 * The file should not contain any header.
 */
inline std::vector<int> load_labels_from_text_file(const char* path, size_t buffer_size = 65536) {
    LabelLoader loader;
    buffin::parse_text_file(path, loader, buffer_size);
    loader.finish();
    return loader.labels;
}

#ifdef SINGLEPP_USE_ZLIB

/**
 * @param path Path to a Gzip-compressed file containing the labels.
 * @param buffer_size Size of the buffer to use when reading the file.
 *
 * @return Vector containing the label index for each reference profile.
 *
 * See `load_labels_from_text_file()` for details about the format.
 */
inline std::vector<int> load_labels_from_gzip_file(const char* path, size_t buffer_size = 65536) {
    LabelLoader loader;
    buffin::parse_gzip_file(path, loader, buffer_size);
    loader.finish();
    return loader.labels;
}

/**
 * @param[in] buffer Pointer to an array containing a Zlib/Gzip-compressed string of labels.
 * @param len Length of the array for `buffer`.
 * @param buffer_size Size of the buffer to use when decompressing the buffer.
 *
 * @return Vector containing the label index for each reference profile.
 *
 * See `load_labels_from_text_file()` for details about the format.
 */
inline std::vector<int> load_labels_from_zlib_buffer(const unsigned char* buffer, size_t len, size_t buffer_size = 65536) {
    LabelLoader loader;
    buffin::parse_zlib_buffer(const_cast<unsigned char*>(buffer), len, loader, 3, buffer_size);
    loader.finish();
    return loader.labels;
}

#endif

/** 
 * @cond
 */
struct FeatureLoader {
    template<typename B>
    void add (const B* buffer, size_t n) {
        size_t last = 0;
        size_t i = 0;
        while (i < n) {
            if (buffer[i] == '\n') {
                if (field != 1) {
                    throw std::runtime_error("two fields (Ensembl ID and symbol) expected on each line");
                }

                if (continuing) {
                    symbols.back() += std::string(buffer + last, buffer + i);
                    continuing = false;
                } else {
                    symbols.emplace_back(buffer + last, buffer + i);
                }
                field = 0;
                last = i + 1;

            } else if (buffer[i] == ',') {
                if (continuing) {
                    ensembl.back() += std::string(buffer + last, buffer + i);
                    continuing = false;
                } else {
                    ensembl.emplace_back(buffer + last, buffer + i);
                }
                ++field;
                last = i + 1;
            }

            ++i;
        }

        if (last != n) {
            auto& target = (field == 0 ? ensembl : symbols);
            if (continuing) {
                target.back() += std::string(buffer + last, buffer + n);
            } else {
                continuing = true;
                target.emplace_back(buffer + last, buffer + n);
            }
        }
    }

    void finish() {
        if (ensembl.size() != symbols.size()) {
            throw std::runtime_error("two fields (Ensembl ID and symbol) expected on the last line");
        }
    }

    int field = 0;
    bool continuing = false;
    std::vector<std::string> ensembl, symbols;
};
/** 
 * @endcond
 */

/**
 * @param path Path to a text file containing the feature annotation.
 * @param buffer_size Size of the buffer to use when reading the file.
 *
 * @return Pair of vectors, each of length equal to the number of features.
 * The first contains Ensembl IDs while the second contains gene symbols.
 *
 * The file should contain one line per feature, with total number of lines equal to the number of features in the dataset.
 * Each line should contain two strings separated by a comma.
 * The first string should be the Ensembl ID while the second string should be the gene symbol; either string may be empty.
 * The file should not contain any header.
 */
inline std::pair<std::vector<std::string>, std::vector<std::string> > load_features_from_text_file(const char* path, size_t buffer_size = 65536) {
    FeatureLoader loader;
    buffin::parse_text_file(path, loader, buffer_size);
    loader.finish();
    return std::make_pair(std::move(loader.ensembl), std::move(loader.symbols));
}

#ifdef SINGLEPP_USE_ZLIB

/**
 * @param path Path to a Gzip-compressed file containing the feature annotation.
 * @param buffer_size Size of the buffer to use when reading the file.
 *
 * @return Pair of vectors, each of length equal to the number of features.
 * The first contains Ensembl IDs while the second contains gene symbols.
 *
 * See `load_features_from_text_file()` for details about the format.
 */
inline std::pair<std::vector<std::string>, std::vector<std::string> > load_features_from_gzip_file(const char* path, size_t buffer_size = 65536) {
    FeatureLoader loader;
    buffin::parse_gzip_file(path, loader, buffer_size);
    loader.finish();
    return std::make_pair(std::move(loader.ensembl), std::move(loader.symbols));
}

/**
 * @param[in] buffer Pointer to an array containing a Zlib/Gzip-compressed string containing the feature annotation.
 * @param len Length of the array for `buffer`.
 * @param buffer_size Size of the buffer to use when decompressing the buffer.
 *
 * @return Pair of vectors, each of length equal to the number of features.
 * The first contains Ensembl IDs while the second contains gene symbols.
 *
 * See `load_features_from_text_file()` for details about the format.
 */
inline std::pair<std::vector<std::string>, std::vector<std::string> > load_features_from_zlib_buffer(const unsigned char* buffer, size_t len, size_t buffer_size = 65536) {
    FeatureLoader loader;
    buffin::parse_zlib_buffer(const_cast<unsigned char*>(buffer), len, loader, 3, buffer_size);
    loader.finish();
    return std::make_pair(std::move(loader.ensembl), std::move(loader.symbols));
}
#endif

/** 
 * @cond
 */
struct RankingLoader {
    RankingLoader(size_t nf, size_t np) : nfeatures(nf), nprofiles(np) {}

    template<typename B>
    void add (const B* buffer, size_t n) {
        size_t i = 0;
        while (i < n) {
            if (buffer[i] == '\n') {
                if (field + 1 != nfeatures) {
                    throw std::runtime_error("number of fields on each line should be equal to the number of features");
                }
                if (!non_empty) {
                    throw std::runtime_error("fields should not be empty");
                }
                values.push_back(current);
                current = 0;
                field = 0;
                non_empty = false;
                ++line;

            } else if (buffer[i] == ',') {
                if (!non_empty) {
                    throw std::runtime_error("fields should not be empty");
                }
                values.push_back(current);
                current = 0;
                ++field;
                non_empty = false;

            } else if (std::isdigit(buffer[i])) {
                non_empty = true;
                current *= 10;
                current += (buffer[i] - '0');

            } else {
                throw std::runtime_error("fields should only contain integer ranks");
            }

            ++i;
        }
    }

    void finish() {
        if (field || non_empty) { // aka no terminating newline.
            if (field + 1 != nfeatures) {
                throw std::runtime_error("number of fields on each line should be equal to the number of features");
            }
            if (!non_empty) {
                throw std::runtime_error("fields should not be empty");
            }
            values.push_back(current);
            ++line;
        }
        if (line != nprofiles) {
            throw std::runtime_error("number of lines is not consistent with the expected number of samples");
        }
    }

    const size_t nfeatures, nprofiles;
    size_t line = 0;

    int field = 0;
    bool continuing = false;
    bool non_empty = false;
    int current = 0;
    std::vector<int> values;
};
/** 
 * @endcond
 */

/**
 * @param path Path to a text file containing the ranking matrix.
 * @param nfeatures Number of features in the ranking matrix.
 * @param nprofiles Number of profiles in the ranking matrix. 
 * @param buffer_size Size of the buffer to use when reading the file.
 *
 * @return Vector corresponding to a column-major matrix of rankings.
 * Each column corresponds to a reference profile while each row corresponds to a feature.
 *
 * The file should contain one line per reference profile, with the total number of lines equal to the number of profiles in the dataset.
 * Each line should contain the rank of each feature's expression within that profile, separated by commas.
 * The number of comma-separated fields on each line should be equal to the number of features.
 * Ranks should be strictly integer - tied ranks should default to the minimum rank among the index set of ties.
 */
inline std::vector<int> load_rankings_from_text_file(const char* path, size_t nfeatures, size_t nprofiles, size_t buffer_size = 65536) {
    RankingLoader loader(nfeatures, nprofiles);
    buffin::parse_text_file(path, loader, buffer_size);
    loader.finish();
    return loader.values;
}

#ifdef SINGLEPP_USE_ZLIB

/**
 * @param path Path to a Gzip-compressed file containing the ranking matrix.
 * @param nfeatures Number of features in the ranking matrix.
 * @param nprofiles Number of profiles in the ranking matrix. 
 * @param buffer_size Size of the buffer to use when reading the file.
 *
 * @return Vector corresponding to a column-major matrix of rankings.
 * Each column corresponds to a reference profile while each row corresponds to a feature.
 *
 * See `load_rankings_from_text_file()` for details about the format.
 */
inline std::vector<int> load_rankings_from_gzip_file(const char* path, size_t nfeatures, size_t nprofiles, size_t buffer_size = 65536) {
    RankingLoader loader(nfeatures, nprofiles);
    buffin::parse_gzip_file(path, loader, buffer_size);
    loader.finish();
    return loader.values;
}

/**
 * @param[in] buffer Pointer to an array containing a Zlib/Gzip-compressed string containing the ranking matrix.
 * @param len Length of the array for `buffer`.
 * @param nfeatures Number of features in the ranking matrix.
 * @param nprofiles Number of profiles in the ranking matrix. 
 * @param buffer_size Size of the buffer to use when reading the file.
 *
 * @return Vector corresponding to a column-major matrix of rankings.
 * Each column corresponds to a reference profile while each row corresponds to a feature.
 *
 * See `load_rankings_from_text_file()` for details about the format.
 */
inline std::vector<int> load_rankings_from_zlib_buffer(const unsigned char* buffer, size_t len, size_t nfeatures, size_t nprofiles, size_t buffer_size = 65536) {
    RankingLoader loader(nfeatures, nprofiles);
    buffin::parse_zlib_buffer(const_cast<unsigned char*>(buffer), len, loader, 3, buffer_size);
    loader.finish();
    return loader.values;
}
#endif

/** 
 * @cond
 */
struct MarkerLoader {
    MarkerLoader(size_t nf, size_t nl) : nfeatures(nf), markers(nl) {
        for (auto& m : markers) {
            m.resize(nl);
        }
    }

    void newline() {
        if (field < 2) {
            throw std::runtime_error("each line should contain at least three fields");
        }
        if (!non_empty) {
            throw std::runtime_error("gene index fields should not be empty");
        }
        values.push_back(current);

        for (auto v : values) {
            if (static_cast<size_t>(v) >= nfeatures) {
                throw std::runtime_error("gene index out of range");
            }
        }

        if (first >= markers.size()) {
            throw std::runtime_error("first label index out of range");
        }
        if (second >= markers.size()) {
            throw std::runtime_error("second label index out of range");
        }

        auto& store = markers[first][second];
        if (!store.empty()) {
            throw std::runtime_error("multiple marker sets listed for a single pairwise comparison");
        }
        store.swap(values);
        values.clear();
        return;
    }

    template<typename B>
    void add (const B* buffer, size_t n) {
        size_t i = 0;
        while (i < n) {
            if (buffer[i] == '\n') {
                newline();
                current = 0;
                field = 0;
                non_empty = false;

            } else if (buffer[i] == '\t') {
                if (!non_empty) {
                    throw std::runtime_error("fields should not be empty");
                }

                if (field == 0) {
                    first = current;
                } else if (field == 1) {
                    second = current;
                } else {
                    values.push_back(current);
                }

                current = 0;
                ++field;
                non_empty = false;

            } else if (std::isdigit(buffer[i])) {
                non_empty = true;
                current *= 10;
                current += (buffer[i] - '0');

            } else {
                throw std::runtime_error("fields should only contain integer ranks");
            }

            ++i;
        }
    }

    void finish() {
        if (field || non_empty) { // aka no terminating newline.
            newline();
        }
    }

    int field = 0;
    bool continuing = false;
    bool non_empty = false;

    int current = 0;
    std::vector<int> values;
    size_t first, second;

    size_t nfeatures;
    singlepp::Markers markers;
};
/** 
 * @endcond
 */

/**
 * @param path Path to a text file containing the marker lists.
 * @param nfeatures Total number of features in the dataset.
 * @param nlabels Number of labels in the dataset.
 * @param buffer_size Size of the buffer to use when reading the file.
 *
 * @return A `Markers` object containing the markers from each pairwise comparison between labels.
 *
 * The file should contain one line per pairwise comparison between labels.
 * Each line should at least 3 tab-delimited fields - the index of the first label, the index of the second label, 
 * and then the indices of the features selected as marker genes for the first label relative to the second.
 * Any (non-zero) number of marker indices may be reported provided they are ordered by marker strength.
 * The total number of lines in this file should be equal to the total number of pairwise comparisons between different labels, including permutations.
 */
inline Markers load_markers_from_text_file(const char* path, size_t nfeatures, size_t nlabels, size_t buffer_size = 65536) {
    MarkerLoader loader(nfeatures, nlabels);
    buffin::parse_text_file(path, loader, buffer_size);
    loader.finish();
    return loader.markers;
}

#ifdef SINGLEPP_USE_ZLIB

/**
 * @param path Path to a Gzip-compressed file containing the marker lists.
 * @param nfeatures Total number of features in the dataset.
 * @param nlabels Number of labels in the dataset.
 * @param buffer_size Size of the buffer to use when reading the file.
 *
 * @return A `Markers` object containing the markers from each pairwise comparison between labels.
 *
 * See `load_markers_from_text_file()` for details about the format.
 */
inline Markers load_markers_from_gzip_file(const char* path, size_t nfeatures, size_t nlabels, size_t buffer_size = 65536) {
    MarkerLoader loader(nfeatures, nlabels);
    buffin::parse_gzip_file(path, loader, buffer_size);
    loader.finish();
    return loader.markers;
}

/**
 * @param[in] buffer Pointer to an array containing a Zlib/Gzip-compressed string containing the marker lists.
 * @param len Length of the array for `buffer`.
 * @param nfeatures Total number of features in the dataset.
 * @param nlabels Number of labels in the dataset.
 * @param buffer_size Size of the buffer to use when reading the file.
 *
 * @return A `Markers` object containing the markers from each pairwise comparison between labels.
 *
 * See `load_markers_from_text_file()` for details about the format.
 */
inline Markers load_markers_from_zlib_buffer(const unsigned char* buffer, size_t len, size_t nfeatures, size_t nlabels, size_t buffer_size = 65536) {
    MarkerLoader loader(nfeatures, nlabels);
    buffin::parse_zlib_buffer(const_cast<unsigned char*>(buffer), len, loader, 3, buffer_size);
    loader.finish();
    return loader.markers;
}
#endif

}

#endif
