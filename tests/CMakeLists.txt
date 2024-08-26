include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/97a467571a0f615a4d96e79e4399c43221ca1232.zip
)

FetchContent_Declare(
  tatami_stats
  GIT_REPOSITORY https://github.com/tatami-inc/tatami_stats
  GIT_TAG master
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_MakeAvailable(tatami_stats)

add_executable(
    libtest 
    src/classify_single.cpp
    src/scaled_ranks.cpp
    src/correlations_to_score.cpp
    src/Intersection.cpp
    src/subset_to_markers.cpp
    src/SubsetSanitizer.cpp
    src/fill_labels_in_use.cpp
    src/annotate_cells_single.cpp
    src/annotate_cells_integrated.cpp
    src/classify_integrated.cpp
    src/choose_classic_markers.cpp
)

target_link_libraries(libtest gtest_main singlepp tatami_stats)
target_compile_options(libtest PRIVATE -Wall -Werror -Wpedantic -Wextra)

option(CODE_COVERAGE "Enable coverage testing" OFF)
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(libtest PRIVATE -O0 -g --coverage)
    target_link_options(libtest PRIVATE --coverage)
endif()

include(GoogleTest)
gtest_discover_tests(libtest)
