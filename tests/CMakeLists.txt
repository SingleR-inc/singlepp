include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/97a467571a0f615a4d96e79e4399c43221ca1232.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
    libtest 
    #src/Classifier.cpp
    src/scaled_ranks.cpp
    src/compute_scores.cpp
    src/process_features.cpp
    #src/fine_tune.cpp
    #src/IntegratedBuilder.cpp
    #src/IntegratedScorer.cpp
    #src/ChooseClassicMarkers.cpp
)

FetchContent_Declare(
  tatami_stats
  GIT_REPOSITORY https://github.com/tatami-inc/tatami_stats
  GIT_TAG master
)

FetchContent_MakeAvailable(tatami_stats)

target_link_libraries(
    libtest
    gtest_main
    singlepp
    tatami_stats
)

set(USE_OPENMP OFF CACHE BOOL "Compile with OpenMP support")
if (USE_OPENMP)
    find_package(OpenMP)
    target_link_libraries(libtest OpenMP::OpenMP_CXX)
endif()

# Adding the load tests.
add_executable(
    loadtest
    src/load_references.cpp
)

target_link_libraries(
    loadtest
    gtest_main
    singlepp
)

include(CheckIncludeFiles)
check_include_files(filesystem HAVE_CXX_FS)
if (NOT HAVE_CXX_FS)
    target_link_libraries(loadtest stdc++fs)
endif()

find_package(ZLIB)
target_link_libraries(loadtest ZLIB::ZLIB)

# Check that the custom parallelization schemes are properly set up.
add_executable(
    custom_parallel
    src/Classifier.cpp
    src/IntegratedBuilder.cpp
    src/IntegratedScorer.cpp
    src/ChooseClassicMarkers.cpp
)

set_target_properties(custom_parallel PROPERTIES COMPILE_DEFINITIONS "TEST_SINGLEPP_CUSTOM_PARALLEL")

target_link_libraries(
    custom_parallel
    gtest_main
    singlepp
)

set(CODE_COVERAGE "Enable coverage testing" OFF)
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(libtest PRIVATE -O0 -g --coverage)
    target_link_options(libtest PRIVATE --coverage)
    target_compile_options(loadtest PRIVATE -O0 -g --coverage)
    target_link_options(loadtest PRIVATE --coverage)
endif()


include(GoogleTest)
gtest_discover_tests(libtest)
gtest_discover_tests(loadtest)
gtest_discover_tests(custom_parallel)
