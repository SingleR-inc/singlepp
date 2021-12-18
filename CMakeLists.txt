cmake_minimum_required(VERSION 3.14)

project(singlepp
    VERSION 1.0.0
    DESCRIPTION "C++ port of the SingleR algorithm"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(singlepp INTERFACE)

target_include_directories(singlepp INTERFACE include/)

add_subdirectory(extern)

target_link_libraries(singlepp INTERFACE tatami knncolle)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif()
endif()
