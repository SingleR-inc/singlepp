include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest
  GIT_TAG main
)

FetchContent_Declare(
  quickstats
  GIT_REPOSITORY https://github.com/LTLA/quickstats
  GIT_TAG master
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_MakeAvailable(quickstats)

add_executable(
    libtest 
    src/classify_single.cpp
    src/scaled_ranks.cpp
    src/l2.cpp
    src/SubsetRemapper.cpp
    src/correlations_to_score.cpp
    src/Intersection.cpp
    src/subset_to_markers.cpp
    src/SubsetSanitizer.cpp
    src/fill_labels_in_use.cpp
    src/classify_integrated.cpp
    src/utils.cpp
)

target_link_libraries(libtest gtest_main singlepp quickstats)
target_compile_options(libtest PRIVATE -Wall -Werror -Wpedantic -Wextra)

option(CODE_COVERAGE "Enable coverage testing" OFF)
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(libtest PRIVATE -O0 -g --coverage)
    target_link_options(libtest PRIVATE --coverage)
endif()

include(GoogleTest)
gtest_discover_tests(libtest)
