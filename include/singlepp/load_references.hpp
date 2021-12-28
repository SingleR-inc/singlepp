#ifndef SINGLEPP_LOADERS_HPP
#define SINGLEPP_LOADERS_HPP

#include <string>
#include <vector>
#include <cctype>

#include "buffin/parse_text_file.hpp"

#ifdef SINGLEPP_USE_ZLIB
#include "buffin/parse_gzip_file.hpp"
#include "buffin/parse_zlib_buffer.hpp"
#endif

namespace singlepp {

/** 
 * @cond
 */
struct LabelLoader {
    template<typename B>
    void add (const B* buffer, size_t n) {
        size_t last = 0;
        size_t i = 0;
        while (i < n) {
            if (buffer[i] != '\n') {
                if (continuing) {
                    labels.back() += std::string(buffer + last, buffer + i);
                    continuing = false;
                } else {
                    labels.emplace_back(buffer + last, buffer + i);
                }
                last = i + 1;
            }
            ++i;
        }

        if (last != n) {
            if (continuing) {
                labels.back() += std::string(buffer + last, buffer + n);
            } else {
                continuing = true;
                labels.emplace_back(buffer + last, buffer + n);
            }
        }
    }

    bool continuing = false;
    std::vector<std::string> labels;
};
/** 
 * @endcond
 */

/** 
 * @cond
 */
struct FeatureLoader {
    template<typename B>
    void add (const B* buffer, size_t n) {
        size_t last = 0;
        size_t i = 0;
        while (i < n) {
            if (buffer[i] != '\n') {
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
        if (field != 1) {
            throw std::runtime_error("two fields (Ensembl ID and symbol) expected on each line");
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
 * @cond
 */
struct RankingLoader {
    RankingLoader(size_t nf, size_t ns) : nfeatures(nf), nsamples(ns) {}

    template<typename B>
    void add (const B* buffer, size_t n) {
        size_t i = 0;
        while (i < n) {
            if (buffer[i] != '\n') {
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
        if (line != nsamples) {
            throw std::runtime_error("number of lines is not consistent with the expected number of samples");
        }
    }

    const size_t nfeatures, nsamples;
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

inline std::vector<std::string> load_labels_from_text_file(const char* path, size_t buffer_size = 65536) {
    LabelLoader loader;
    buffin::parse_text_file(path, loader, buffer_size);
    return loader.labels;
}

#ifdef SINGLEPP_USE_ZLIB
inline std::vector<std::string> load_labels_from_gzip_file(const char* path, size_t buffer_size = 65536) {
    LabelLoader loader;
    buffin::parse_gzip_file(path, loader, buffer_size);
    return loader.labels;
}

inline std::vector<std::string> load_labels_from_zlib_buffer(const unsigned char* buffer, size_t len, size_t buffer_size = 65536) {
    LabelLoader loader;
    buffin::parse_zlib_buffer(buffer, len, loader, buffer_size);
    return loader.labels;
}
#endif

inline std::pair<std::vector<std::string>, std::vector<std::string> > load_features_from_text_file(const char* path, size_t buffer_size = 65536) {
    FeatureLoader loader;
    buffin::parse_text_file(path, loader, buffer_size);
    loader.finish();
    return std::make_pair(std::move(loader.ensembl), std::move(loader.symbols));
}

#ifdef SINGLEPP_USE_ZLIB
inline std::pair<std::vector<std::string>, std::vector<std::string> > load_features_from_gzip_file(const char* path, size_t buffer_size = 65536) {
    FeatureLoader loader;
    buffin::parse_gzip_file(path, loader, buffer_size);
    loader.finish();
    return std::make_pair(std::move(loader.ensembl), std::move(loader.symbols));
}

inline std::pair<std::vector<std::string>, std::vector<std::string> > load_features_from_zlib_buffer(const unsigned char* buffer, size_t len, size_t buffer_size = 65536) {
    FeatureLoader loader;
    buffin::parse_text_file(buffer, len, loader, buffer_size);
    loader.finish();
    return std::make_pair(std::move(loader.ensembl), std::move(loader.symbols));
}
#endif

inline std::vector<int> load_rankings_from_text_file(const char* path, size_t nfeatures, size_t nsamples, size_t buffer_size = 65536) {
    RankingLoader loader(nfeatures, nsamples);
    buffin::parse_text_file(path, loader, buffer_size);
    loader.finish();
    return loader.values;
}

#ifdef SINGLEPP_USE_ZLIB
inline std::vector<int> load_rankings_from_gzip_file(const char* path, size_t nfeatures, size_t nsamples, size_t buffer_size = 65536) {
    RankingLoader loader(nfeatures, nsamples);
    buffin::parse_gzip_file(path, loader, buffer_size);
    loader.finish();
    return loader.values;
}

inline std::vector<int> load_rankings_from_zlib_buffer(const unsigned char* buffer, size_t len, size_t nfeatures, size_t nsamples, size_t buffer_size = 65536) {
    RankingLoader loader(nfeatures, nsamples);
    buffin::parse_zlib_buffer(buffer, len, loader, buffer_size);
    loader.finish();
    return loader.values;
}
#endif

}

#endif
