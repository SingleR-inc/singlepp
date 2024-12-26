cmake_minimum_required(VERSION 3.14)

project(singlepp
    VERSION 2.0.1
    DESCRIPTION "C++ port of the SingleR algorithm"
    LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Library
add_library(singlepp INTERFACE)
add_library(singler::singlepp ALIAS singlepp)

target_include_directories(singlepp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/singlepp>)
target_compile_features(singlepp INTERFACE cxx_std_17)

# Dependencies
option(SINGLEPP_FETCH_EXTERN "Automatically fetch singlepp's external dependencies." ON)
if(SINGLEPP_FETCH_EXTERN)
    add_subdirectory(extern)
else()
    find_package(knncolle_knncolle 2.0.0 CONFIG REQUIRED)
    find_package(tatami_tatami 3.0.0 CONFIG REQUIRED)
endif()

target_link_libraries(singlepp INTERFACE knncolle::knncolle tatami::tatami)

# Tests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(SINGLEPP_TESTS "Build singlepp's test suite." ON)
else()
    option(SINGLEPP_TESTS "Build singlepp's test suite." OFF)
endif()

if(SINGLEPP_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif() 
endif()

# Install
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/singlepp)

install(TARGETS singlepp
    EXPORT singleppTargets)

install(EXPORT singleppTargets
    FILE singler_singleppTargets.cmake
    NAMESPACE singler::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/singler_singlepp)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/singler_singleppConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/singler_singlepp)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/singler_singleppConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/singler_singleppConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/singler_singleppConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/singler_singlepp)
