# C++ port of SingleR 

## Overview

This repository contains a C++ port of the [**SingleR**](https://bioconductor.org/packages/SingleR) R package for automated cell type annotation.
It primarily focuses on the prediction step given a set of references; the preparation of the references themselves is left to the user (see below).
The library contains methods for simple and multi-reference predictions, returning scores and labels for each cell in the test dataset.
Each cell is treated independently so the entire process is trivially parallelizable.

## Quick start

**singlepp** is a header-only library, so it can be easily used by just `#include`ing the relevant source files.
Assuming the reference matrix, labels and markers are available, we can easily run the classification:

```cpp
#include "singlepp/singlepp.hpp"

// Prepare the reference matrix as a tatami::NumericMatrix.
ref_mat;

// Prepare a vector of labels, one per column of ref_mat.
ref_labels;

// Prepare a vector of vectors of markers for pairwise comparisons between labels.
ref_markers;

// Building the classifier.
singlepp::TrainSingleOptions<int, double> train_opt;
auto trained = singlepp::train_single(ref_mat, ref_labels.data(), train_opt);

// Running the classification on the test matrix.
singlepp::ClassifySingleOptions<double> class_opt;
auto res = singlepp::classify_single<int>(test_mat, trained, class_opt);
```

See the [reference documentation](https://singler-inc.github.io/singlepp) for more details.

## Identifying markers

Given a reference dataset, **singlepp** implements a simple method of identifying marker genes between labels.
This is based on ranking the differences in median log-expression values between labels and is the "classic" method provided in the original **SingleR** package.

```cpp
singlepp::ChooseClassicMarkersOptions mrk;
auto classic_markers = singlepp::choose_classic_markers<int>(
    ref_mat.get(),
    ref_labels.data(),
    m_opt
);
```

The `classic_markers` can then be directly used in `train_single()`.
Of course, other marker detection schemes can be used, depending on the type of reference dataset.
For single-cell references, users may be interested in some of the differential analysis methods in the [**libscran**](https://github.com/libscran/scran_markers) library.

By default, it is expected that the `markers` supplied to `train_single()` has already been filtered to only the top markers for each pairwise comparison.
However, in some cases, it might be more convenient for `markers` to contain a ranking of all genes such that the desired subset of top markers can be chosen later.
This is achieved by setting `TrainSingleOptions::top` to the desired number of markers per comparison, e.g., for 20 markers:

```cpp
train_opt.top = 20;
auto trained20 = singlepp::train_single(
    ref_mat,
    ref_labels.data(),
    train_opt
);
```

Doing so is roughly equivalent to slicing each vector in `markers` to the top 20 entries before calling `train_single()`.
In fact, calling `set_top()` is the better approach when intersecting feature spaces - see below -
as the top set will not be contaminated by genes that are not present in the test dataset.

## Intersecting feature sets

Often the reference dataset will not have the same genes as the test dataset.
To handle this case, users should use `train_single_intersect()` with identifiers for the rows of the reference and test matrices.

```cpp
test_names; // vector of feature IDs for the test data
ref_names; // vector of feature IDs for the reference data

auto trained_intersect = singlepp::train_single_intersect(
    test_mat.nrow(),
    test_names.data(),
    ref_mat,
    ref_names.data(),
    ref_labels.data(), 
    ref_markers,
    train_opt
);
```

Then, `classify_single_intersect()` will perform classification using only the intersection of genes:

```cpp
auto res_intersect = singlepp::classify_single_intersect<int>(
    test_mat,
    trained_intersect,
    class_opt
);
```

The gene identifiers can be anything that can be hashed and compared.
These are most commonly `std::string`s but can also be integers (e.g., for Entrez IDs).

## Integrating results across references

To combine results from multiple references, we first need to perform classification within each reference. 
Let's say we have two references A and B:

```cpp
auto trainA = singlepp::train_single(refA_mat, refA_labels.data(), refA_markers, train_opt);
auto resA = singlepp::classify_single<int>(test_mat, trainA, class_opt);

auto trainB = singlepp::train_single(refB_mat, refB_labels.data(), refB_markers, train_opt);
auto resB = singlepp::classify_single<int>(test_mat, trainB, class_opt);
```

We build the integrated classifier:

```cpp
std::vector<singlepp::TrainIntegratedInput<double, int, int> > inputs;
inputs.push_back(singlepp::prepare_integrated_input(refA_mat, refA_labels.data(), preA));
inputs.push_back(singlepp::prepare_integrated_input(refB_mat, refB_labels.data(), preB));

singlepp::TrainIntegratedOptions ti_opt;
auto train_integrated = singlepp::train_integrated(inputs, ti_opt);
```

And then we can finally run the scoring.
For each cell in the test dataset, `classify_integrated()` picks the best label among the assignments from each individual reference.

```cpp
singlepp::ClassifyIntegratedOptions<double> ci_opt;
auto ires = single.run(test_mat, train_integrated, ci_opt);
ires.best; // index of the best reference.
```

## Building projects 

### CMake with `FetchContent`

If you're using CMake, you just need to add something like this to your `CMakeLists.txt`:

```cmake
include(FetchContent)

FetchContent_Declare(
  singlepp
  GIT_REPOSITORY https://github.com/singler-inc/singlepp
  GIT_TAG master # or any version of interest
)

FetchContent_MakeAvailable(singlepp)
```

Then you can link to **singlepp** to make the headers available during compilation:

```cmake
# For executables:
target_link_libraries(myexe singlepp)

# For libaries
target_link_libraries(mylib INTERFACE singlepp)
```

### CMake with `find_package()`

```cmake
find_package(singler_singlepp CONFIG REQUIRED)
target_link_libraries(mylib INTERFACE singler::singlepp)
```

To install the library, use:

```sh
mkdir build && cd build
cmake .. -DSINGLEPP_TESTS=OFF
cmake --build . --target install
```

By default, this will use `FetchContent` to fetch all external dependencies.
If you want to install them manually, use `-DSINGLEPP_FETCH_EXTERN=OFF`.
See the tags in [`extern/CMakeLists.txt`](extern/CMakeLists.txt) to find compatible versions of each dependency.

### Manual

If you're not using CMake, the simple approach is to just copy the files in `include/` - either directly or with Git submodules - and include their path during compilation with, e.g., GCC's `-I`.
This requires the external dependencies listed in [`extern/CMakeLists.txt`](extern/CMakeLists.txt), which also need to be made available during compilation.

## References

Aran D et al. (2019). 
Reference-based analysis of lung single-cell sequencing reveals a transitional profibrotic macrophage.
_Nat. Immunol._ 20, 163-172
