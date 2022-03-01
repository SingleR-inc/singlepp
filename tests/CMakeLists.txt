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
    src/SinglePP.cpp
    src/scaled_ranks.cpp
    src/compute_scores.cpp
    src/process_features.cpp
    src/fine_tune.cpp
    src/IntegratedBuilder.cpp
    src/IntegratedScorer.cpp
)

target_link_libraries(
    libtest
    gtest_main
    singlepp
)

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
