#include <gtest/gtest.h>

#define SINGLEPP_USE_ZLIB
#include "singlepp/load_references.hpp"
#include "byteme/temp_file_path.hpp"
#include "zlib.h"

#include <fstream>
#include <string>
#include <vector>

/*************************************************/

class LoadFeaturesTest : public ::testing::TestWithParam<int> {};

TEST_P(LoadFeaturesTest, TextFile) {
    auto path = byteme::temp_file_path("feat_text");
    std::vector<std::string> ensembl, symbols;
    {
        std::ofstream out(path, std::ofstream::out);
        for (size_t i = 0; i < 1000; ++i) {
            auto id = std::to_string(i);
            auto sym = "GENE_" + id;
            ensembl.push_back(id);
            symbols.push_back(sym);
            out << id << "," << sym << "\n";
        }
    }

    auto reloaded = singlepp::load_features_from_text_file(path.c_str(), GetParam());
    EXPECT_EQ(reloaded.first, ensembl);
    EXPECT_EQ(reloaded.second, symbols);
}

TEST_P(LoadFeaturesTest, GzipFile) {
    auto path = byteme::temp_file_path("feat_gzip");
    std::vector<std::string> ensembl, symbols;
    {
        std::string output;
        for (size_t i = 0; i < 1000; ++i) {
            auto id = std::to_string(i);
            auto sym = "GENE_" + id;
            ensembl.push_back(id);
            symbols.push_back(sym);
            output += id + "," + sym + "\n";
        }

        gzFile ohandle = gzopen(path.c_str(), "w");
        gzwrite(ohandle, output.c_str(), output.size());
        gzclose(ohandle);
    }

    auto reloaded = singlepp::load_features_from_gzip_file(path.c_str(), GetParam());
    EXPECT_EQ(reloaded.first, ensembl);
    EXPECT_EQ(reloaded.second, symbols);

    std::ifstream in(path, std::ios::binary);
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(in), {});
    auto reloaded2 = singlepp::load_features_from_zlib_buffer(buffer.data(), buffer.size(), GetParam());
    EXPECT_EQ(reloaded2.first, ensembl);
    EXPECT_EQ(reloaded2.second, symbols);
}

void quick_dump(std::string path, std::string contents) {
    std::ofstream out (path, std::ofstream::out);
    out << contents;
}

void quick_feature_err(std::string path, std::string msg) {
    EXPECT_ANY_THROW({
        try {
            singlepp::load_features_from_text_file(path.c_str());
        } catch (std::exception& e) {
            EXPECT_TRUE(std::string(e.what()).find(msg) != std::string::npos);
            throw;
        }
    });
    return;
}

TEST(LoadFeatures, EdgeCases) {
    {
        auto path = byteme::temp_file_path("feat_err");
        quick_dump(path, "asd\nasdasd,asdasd");
        quick_feature_err(path, "two comma-separated fields");
    }

    {
        auto path = byteme::temp_file_path("feat_err");
        quick_dump(path, "asdasdasd,ad\nasdasd");
        quick_feature_err(path, "two comma-separated fields");
    }

    // Empty fields are ok, though, along with non-newline termination.
    {
        auto path = byteme::temp_file_path("feat_ok");
        quick_dump(path, "asdasdasd,\n,asdasd");
        auto output = singlepp::load_features_from_text_file(path.c_str());
        EXPECT_EQ(output.first[0], "asdasdasd");
        EXPECT_EQ(output.first[1], "");
        EXPECT_EQ(output.second[0], "");
        EXPECT_EQ(output.second[1], "asdasd");
    }
}

INSTANTIATE_TEST_SUITE_P(
    LoadFeatures,
    LoadFeaturesTest,
    ::testing::Values(10, 25, 100, 1000)
);

/*************************************************/

class LoadLabelsTest : public ::testing::TestWithParam<int> {};

TEST_P(LoadLabelsTest, TextFile) {
    auto path = byteme::temp_file_path("lab_text");
    std::vector<int> labels;
    {
        std::ofstream out(path, std::ofstream::out);
        for (size_t i = 0; i < 1000; ++i) {
            labels.push_back(i);
            out << i << "\n";
        }
    }

    auto reloaded = singlepp::load_labels_from_text_file(path.c_str(), GetParam());
    EXPECT_EQ(reloaded, labels);
}

TEST_P(LoadLabelsTest, GzipFile) {
    auto path = byteme::temp_file_path("lab_gzip");
    std::vector<int> labels;
    {
        std::string output;
        for (size_t i = 0; i < 1000; ++i) {
            labels.push_back(i);
            output += std::to_string(i) + "\n";
        }

        gzFile ohandle = gzopen(path.c_str(), "w");
        gzwrite(ohandle, output.c_str(), output.size());
        gzclose(ohandle);
    }

    auto reloaded = singlepp::load_labels_from_gzip_file(path.c_str(), GetParam());
    EXPECT_EQ(reloaded, labels);

    std::ifstream in(path, std::ios::binary);
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(in), {});
    auto reloaded2 = singlepp::load_labels_from_zlib_buffer(buffer.data(), buffer.size(), GetParam());
    EXPECT_EQ(reloaded2, labels);
}

void quick_label_err(std::string path, std::string msg) {
    EXPECT_ANY_THROW({
        try {
            singlepp::load_labels_from_text_file(path.c_str());
        } catch (std::exception& e) {
            EXPECT_TRUE(std::string(e.what()).find(msg) != std::string::npos);
            throw;
        }
    });
    return;
}

TEST(LoadLabels, EdgeCases) {
    {
        auto path = byteme::temp_file_path("label_err");
        quick_dump(path, "1\n2\n3a\n4\n");
        quick_label_err(path, "must be an integer");
    }

    {
        auto path = byteme::temp_file_path("label_err");
        quick_dump(path, "1\n2\n\n4\n");
        quick_label_err(path, "must be an integer");
    }

    // Non-newline termination is ok, as are empty fields.
    {
        auto path = byteme::temp_file_path("feat_ok");
        quick_dump(path, "1\n2");
        auto output = singlepp::load_labels_from_text_file(path.c_str());
        EXPECT_EQ(output.size(), 2);
        EXPECT_EQ(output[0], 1);
        EXPECT_EQ(output[1], 2);
    }
}

INSTANTIATE_TEST_SUITE_P(
    LoadLabels,
    LoadLabelsTest,
    ::testing::Values(10, 25, 100, 1000)
);

/*************************************************/

class LoadRankingsTest : public ::testing::TestWithParam<int> {};

std::vector<int> extract_ranks(const singlepp::RankMatrix<int, int>& mat) {
    std::vector<int> copy(mat.nrow() * mat.ncol());
    auto wrk = mat.dense_column();
    for (size_t i = 0; i < mat.ncol(); ++i) {
        wrk->fetch_copy(i, copy.data() + i * mat.nrow());
    }
    return copy;
}

TEST_P(LoadRankingsTest, TextFile) {
    auto path = byteme::temp_file_path("rank_text");
    size_t nfeat = 49, nprof = 13;
    std::vector<int> ranks;
    {
        std::ofstream out(path, std::ofstream::out);
        size_t counter = 23;

        for (size_t p = 0; p < nprof; ++p) {
            for (size_t f = 0; f < nfeat; ++f) {
                ranks.push_back(counter);
                if (f != 0) {
                    out << ",";
                }
                out << counter;

                // Imperfect wrap so that each line is different from the last.
                ++counter;
                if (counter == 100) {
                    counter = 0;
                }
            }
            out << "\n";
        }
    }

    auto reloaded = singlepp::load_rankings_from_text_file<int, int>(path.c_str(), GetParam());
    EXPECT_EQ(reloaded.nrow(), nfeat);
    EXPECT_EQ(reloaded.ncol(), nprof);

    auto copy = extract_ranks(reloaded);
    EXPECT_EQ(copy, ranks);
}

TEST_P(LoadRankingsTest, GzipFile) {
    auto path = byteme::temp_file_path("rank_gzip");
    size_t nfeat = 51, nprof = 17;
    std::vector<int> ranks;
    {
        std::string output;
        size_t counter = 31;

        for (size_t p = 0; p < nprof; ++p) {
            for (size_t f = 0; f < nfeat; ++f) {
                ranks.push_back(counter);
                if (f != 0) {
                    output += ",";
                }
                output += std::to_string(counter);

                // Imperfect wrap so that each line is different from the last.
                ++counter;
                if (counter == 100) {
                    counter = 0;
                }
            }
            output += "\n";
        }

        gzFile ohandle = gzopen(path.c_str(), "w");
        gzwrite(ohandle, output.c_str(), output.size());
        gzclose(ohandle);
    }

    auto reloaded = singlepp::load_rankings_from_gzip_file<int, int>(path.c_str(), GetParam());
    auto copy = extract_ranks(reloaded);
    EXPECT_EQ(copy, ranks);

    std::ifstream in(path, std::ios::binary);
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(in), {});
    auto reloaded2 = singlepp::load_rankings_from_zlib_buffer<int, int>(buffer.data(), buffer.size(), GetParam());
    auto copy2 = extract_ranks(reloaded2);
    EXPECT_EQ(copy2, ranks);
}

void quick_ranking_err(std::string path, std::string msg) {
    EXPECT_ANY_THROW({
        try {
            singlepp::load_rankings_from_text_file(path.c_str());
        } catch (std::exception& e) {
            EXPECT_TRUE(std::string(e.what()).find(msg) != std::string::npos);
            throw;
        }
    });
    return;
}

TEST(LoadRankings, EdgeCases) {
    {
        auto path = byteme::temp_file_path("rank_err");
        quick_dump(path, "a,v,b,d\n");
        quick_ranking_err(path, "integer ranks");
    }

    {
        auto path = byteme::temp_file_path("rank_err");
        quick_dump(path, "1,2,3,4\n1,2,3\n1,2,3,4\n");
        quick_ranking_err(path, "number of fields");
    }

    {
        auto path = byteme::temp_file_path("rank_err");
        quick_dump(path, "1,2,3,4\n1,2,3,\n1,2,3,4\n");
        quick_ranking_err(path, "not be empty");
    }

    {
        auto path = byteme::temp_file_path("rank_err");
        quick_dump(path, "1,2,3,4\n1,2,,4\n1,2,3,4\n");
        quick_ranking_err(path, "not be empty");
    }

    {
        auto path = byteme::temp_file_path("rank_err");
        quick_dump(path, "1,2,3,4\n1,2,3,4\n1,2,3\n");
        quick_ranking_err(path, "number of fields");
    }

    {
        auto path = byteme::temp_file_path("rank_err");
        quick_dump(path, "1,2,3,4\n1,2,3,4\n1,2,3,\n");
        quick_ranking_err(path, "not be empty");
    }

    // Non-newline termination is ok.
    {
        auto path = byteme::temp_file_path("feat_ok");
        quick_dump(path, "1,2,3,4\n5,6,7,8");
        auto output = singlepp::load_rankings_from_text_file<int, int>(path.c_str());
        std::vector<int> expected { 1,2,3,4,5,6,7,8 };
        EXPECT_EQ(extract_ranks(output), expected);
    }
}

INSTANTIATE_TEST_SUITE_P(
    LoadRankings,
    LoadRankingsTest,
    ::testing::Values(10, 25, 100, 1000)
);

/*************************************************/

class LoadMarkersTest : public ::testing::TestWithParam<int> {
protected:
    void compare_markers(const singlepp::Markers& ref, const singlepp::Markers& obs) { 
        ASSERT_EQ(ref.size(), obs.size());

        for (size_t m = 0; m < obs.size(); ++m) {
            const auto& observed = obs[m];
            const auto& expected = ref[m];

            ASSERT_EQ(observed.size(), expected.size());
            for (size_t m2 = 0; m2 < observed.size(); ++m2) {
                EXPECT_EQ(observed[m2], expected[m2]);
            }
        }
    }        
};

TEST_P(LoadMarkersTest, TextFile) {
    auto path = byteme::temp_file_path("mark_text");

    std::mt19937_64 rng(GetParam());
    size_t nfeatures = 1000, nlabels = 3;
    singlepp::Markers markers(nlabels);
    {
        std::ofstream out(path, std::ofstream::out);

        for (size_t l = 0; l < nlabels; ++l) {
            markers[l].resize(nlabels);
            for (size_t l2 = 0; l2 < nlabels; ++l2) {
                if (l == l2) {
                    continue;
                }
                out << l << "\t" << l2;

                size_t ngenes = rng() % 20 + 1;
                for (size_t i = 0; i < ngenes; ++i) {
                    auto current = rng() % nfeatures;
                    markers[l][l2].push_back(current);
                    out << "\t" << current;
                }

                out << "\n";
            }
        }
    }

    auto reloaded = singlepp::load_markers_from_text_file(path.c_str(), nfeatures, nlabels, GetParam());
    compare_markers(markers, reloaded);
}

TEST_P(LoadMarkersTest, GzipFile) {
    auto path = byteme::temp_file_path("mark_text");

    std::mt19937_64 rng(GetParam());
    size_t nfeatures = 1000, nlabels = 3;
    singlepp::Markers markers(nlabels);
    {
        std::ofstream out(path, std::ofstream::out);
        std::string output;

        for (size_t l = 0; l < nlabels; ++l) {
            markers[l].resize(nlabels);
            for (size_t l2 = 0; l2 < nlabels; ++l2) {
                if (l == l2) {
                    continue;
                }
                output += std::to_string(l) + "\t" + std::to_string(l2);

                size_t ngenes = rng() % 20 + 1;
                for (size_t i = 0; i < ngenes; ++i) {
                    auto current = rng() % nfeatures;
                    markers[l][l2].push_back(current);
                    output += "\t" + std::to_string(current); 
                }
                output += "\n";
            }
        }

        gzFile ohandle = gzopen(path.c_str(), "w");
        gzwrite(ohandle, output.c_str(), output.size());
        gzclose(ohandle);
    }

    auto reloaded = singlepp::load_markers_from_gzip_file(path.c_str(), nfeatures, nlabels, GetParam());
    compare_markers(markers, reloaded);

    std::ifstream in(path, std::ios::binary);
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(in), {});
    auto reloaded2 = singlepp::load_markers_from_zlib_buffer(buffer.data(), buffer.size(), nfeatures, nlabels, GetParam());
    compare_markers(markers, reloaded2);
}

void quick_marker_err(std::string path, size_t nf, size_t nl, std::string msg) {
    EXPECT_ANY_THROW({
        try {
            singlepp::load_markers_from_text_file(path.c_str(), nf, nl);
        } catch (std::exception& e) {
            EXPECT_TRUE(std::string(e.what()).find(msg) != std::string::npos);
            throw;
        }
    });
    return;
}

TEST(LoadMarkers, EdgeCases) {
    {
        auto path = byteme::temp_file_path("mark_err");
        quick_dump(path, "1\t2\t1000\n");
        quick_marker_err(path, 1, 3, "gene index out of range");
    }

    {
        auto path = byteme::temp_file_path("mark_err");
        quick_dump(path, "5\t1\t2\n");
        quick_marker_err(path, 5, 3, "label index out of range");
    }

    {
        auto path = byteme::temp_file_path("mark_err");
        quick_dump(path, "1\t5\t2\n");
        quick_marker_err(path, 5, 3, "label index out of range");
    }

    {
        auto path = byteme::temp_file_path("mark_err");
        quick_dump(path, "1\t1\t\n");
        quick_marker_err(path, 5, 3, "not be empty");
    }

    {
        auto path = byteme::temp_file_path("mark_err");
        quick_dump(path, "1\t1\t\t1\n");
        quick_marker_err(path, 5, 3, "not be empty");
    }

    {
        auto path = byteme::temp_file_path("mark_err");
        quick_dump(path, "1\t1\n");
        quick_marker_err(path, 5, 3, "at least three tab-separated fields");
    }

    {
        auto path = byteme::temp_file_path("mark_err");
        quick_dump(path, "1\t1\t1\n1\t1\t1\n");
        quick_marker_err(path, 5, 3, "multiple marker");
    }

    {
        auto path = byteme::temp_file_path("mark_err");
        quick_dump(path, "2\t1\t1\n1\t2\t1a\n");
        quick_marker_err(path, 5, 3, "integer");
    }

    {
        auto path = byteme::temp_file_path("mark_ok");
        quick_dump(path, "2\t1\t1\n1\t2\t0");
        auto output = singlepp::load_markers_from_text_file(path.c_str(), 5, 3);
        EXPECT_EQ(output.size(), 3);
        EXPECT_EQ(output[1][2].size(), 1);
        EXPECT_EQ(output[1][2].front(), 0);
    }
}

INSTANTIATE_TEST_SUITE_P(
    LoadMarkers,
    LoadMarkersTest,
    ::testing::Values(10, 25, 100, 1000)
);

/*************************************************/

class LoadLabelNamesTest : public ::testing::TestWithParam<int> {};

TEST_P(LoadLabelNamesTest, TextFile) {
    auto path = byteme::temp_file_path("lab_text");
    std::vector<std::string> labels;
    {
        std::ofstream out(path, std::ofstream::out);
        for (size_t i = 0; i < 150; ++i) {
            auto lab = "LABEL_" + std::to_string(i);
            labels.push_back(lab);
            out << lab << "\n";
        }
    }

    auto reloaded = singlepp::load_label_names_from_text_file(path.c_str(), GetParam());
    EXPECT_EQ(reloaded, labels);
}

TEST_P(LoadLabelNamesTest, GzipFile) {
    auto path = byteme::temp_file_path("lab_gzip");
    std::vector<std::string> labels;
    {
        std::string output;
        for (size_t i = 0; i < 150; ++i) {
            auto lab = "LABEL_" + std::to_string(i);
            labels.push_back(lab);
            output += lab + "\n";
        }

        gzFile ohandle = gzopen(path.c_str(), "w");
        gzwrite(ohandle, output.c_str(), output.size());
        gzclose(ohandle);
    }

    auto reloaded = singlepp::load_label_names_from_gzip_file(path.c_str(), GetParam());
    EXPECT_EQ(reloaded, labels);

    std::ifstream in(path, std::ios::binary);
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(in), {});
    auto reloaded2 = singlepp::load_label_names_from_zlib_buffer(buffer.data(), buffer.size(), GetParam());
    EXPECT_EQ(reloaded2, labels);
}

TEST(LoadLabelNames, EdgeCases) {
    // Non-newline termination is ok, as are empty fields.
    {
        auto path = byteme::temp_file_path("feat_ok");
        quick_dump(path, "asdasdasd\n\nasdasd");
        auto output = singlepp::load_label_names_from_text_file(path.c_str());
        EXPECT_EQ(output[0], "asdasdasd");
        EXPECT_EQ(output[1], "");
        EXPECT_EQ(output[2], "asdasd");
    }
}

INSTANTIATE_TEST_SUITE_P(
    LoadLabelNames,
    LoadLabelNamesTest,
    ::testing::Values(10, 25, 100, 1000)
);
