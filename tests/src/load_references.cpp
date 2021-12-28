#include <gtest/gtest.h>

#define SINGLEPP_USE_ZLIB
#include "singlepp/load_references.hpp"
#include "buffin/temp_file_path.hpp"
#include "zlib.h"

#include <fstream>
#include <string>
#include <vector>

/*************************************************/

class LoadFeaturesTest : public ::testing::TestWithParam<int> {};

TEST_P(LoadFeaturesTest, TextFile) {
    auto path = buffin::temp_file_path("feat_text");
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
    auto path = buffin::temp_file_path("feat_gzip");
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
        auto path = buffin::temp_file_path("feat_err");
        quick_dump(path, "asd\nasdasd,asdasd");
        quick_feature_err(path, "two fields");
    }

    {
        auto path = buffin::temp_file_path("feat_err");
        quick_dump(path, "asdasdasd,ad\nasdasd");
        quick_feature_err(path, "last line");
    }

    // Empty fields are ok, though, along with non-newline termination.
    {
        auto path = buffin::temp_file_path("feat_ok");
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
    auto path = buffin::temp_file_path("lab_text");
    std::vector<std::string> labels;
    {
        std::ofstream out(path, std::ofstream::out);
        for (size_t i = 0; i < 1000; ++i) {
            auto lab = "LABEL_" + std::to_string(i);
            labels.push_back(lab);
            out << lab << "\n";
        }
    }

    auto reloaded = singlepp::load_labels_from_text_file(path.c_str(), GetParam());
    EXPECT_EQ(reloaded, labels);
}

TEST_P(LoadLabelsTest, GzipFile) {
    auto path = buffin::temp_file_path("lab_gzip");
    std::vector<std::string> labels;
    {
        std::string output;
        for (size_t i = 0; i < 1000; ++i) {
            auto lab = "LABEL_" + std::to_string(i);
            labels.push_back(lab);
            output += lab + "\n";
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
    // Non-newline termination is ok, as are empty fields.
    {
        auto path = buffin::temp_file_path("feat_ok");
        quick_dump(path, "asdasdasd\n\nasdasd");
        auto output = singlepp::load_labels_from_text_file(path.c_str());
        EXPECT_EQ(output[0], "asdasdasd");
       EXPECT_EQ(output[1], "");
        EXPECT_EQ(output[2], "asdasd");
    }
}

INSTANTIATE_TEST_SUITE_P(
    LoadLabels,
    LoadLabelsTest,
    ::testing::Values(10, 25, 100, 1000)
);

/*************************************************/

class LoadRankingsTest : public ::testing::TestWithParam<int> {};

TEST_P(LoadRankingsTest, TextFile) {
    auto path = buffin::temp_file_path("rank_text");
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

    auto reloaded = singlepp::load_rankings_from_text_file(path.c_str(), nfeat, nprof, GetParam());
    EXPECT_EQ(reloaded, ranks);
}

TEST_P(LoadRankingsTest, GzipFile) {
    auto path = buffin::temp_file_path("rank_gzip");
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

    auto reloaded = singlepp::load_rankings_from_gzip_file(path.c_str(), nfeat, nprof, GetParam());
    EXPECT_EQ(reloaded, ranks);

    std::ifstream in(path, std::ios::binary);
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(in), {});
    auto reloaded2 = singlepp::load_rankings_from_zlib_buffer(buffer.data(), buffer.size(), nfeat, nprof, GetParam());
    EXPECT_EQ(reloaded2, ranks);
}

void quick_ranking_err(std::string path, size_t nf, size_t np, std::string msg) {
    EXPECT_ANY_THROW({
        try {
            singlepp::load_rankings_from_text_file(path.c_str(), nf, np);
        } catch (std::exception& e) {
            EXPECT_TRUE(std::string(e.what()).find(msg) != std::string::npos);
            throw;
        }
    });
    return;
}

TEST(LoadRankings, EdgeCases) {
    {
        auto path = buffin::temp_file_path("rank_err");
        quick_dump(path, "a,v,b,d\n");
        quick_ranking_err(path, 4, 1, "integer ranks");
    }

    {
        auto path = buffin::temp_file_path("rank_err");
        quick_dump(path, "1,2,3,4\n1,2,3\n1,2,3,4\n");
        quick_ranking_err(path, 4, 3, "number of fields");
    }

    {
        auto path = buffin::temp_file_path("rank_err");
        quick_dump(path, "1,2,3,4\n1,2,3,\n1,2,3,4\n");
        quick_ranking_err(path, 4, 3, "not be empty");
    }

    {
        auto path = buffin::temp_file_path("rank_err");
        quick_dump(path, "1,2,3,4\n1,2,,4\n1,2,3,4\n");
        quick_ranking_err(path, 4, 3, "not be empty");
    }

    {
        auto path = buffin::temp_file_path("rank_err");
        quick_dump(path, "1,2,3,4\n1,2,3,4\n1,2,3\n");
        quick_ranking_err(path, 4, 3, "number of fields");
    }

    {
        auto path = buffin::temp_file_path("rank_err");
        quick_dump(path, "1,2,3,4\n1,2,3,4\n1,2,3,\n");
        quick_ranking_err(path, 4, 3, "not be empty");
    }

    {
        auto path = buffin::temp_file_path("rank_err");
        quick_dump(path, "1,2,3,4\n1,2,3,4\n1,2,3,4\n");
        quick_ranking_err(path, 4, 4, "not consistent");
    }

    // Non-newline termination is ok.
    {
        auto path = buffin::temp_file_path("feat_ok");
        quick_dump(path, "1,2,3,4\n5,6,7,8");
        auto output = singlepp::load_rankings_from_text_file(path.c_str(), 4, 2);
        std::vector<int> expected { 1,2,3,4,5,6,7,8 };
        EXPECT_EQ(output, expected);
    }
}

INSTANTIATE_TEST_SUITE_P(
    LoadRankings,
    LoadRankingsTest,
    ::testing::Values(10, 25, 100, 1000)
);


