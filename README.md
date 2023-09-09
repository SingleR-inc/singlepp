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

// Running the classification on a test matrix.
singlepp::Classifier runner;
auto res = runner.run(test_mat.get(), ref_mat.get(), ref_labels.data(), ref_markers);
```

This returns an object containing the scores and predicted label for each cell in the test matrix.
Users can also supply their own arrays to be filled with the results:

```cpp
std::vector<int> assignments(ncells);
std::vector<std::vector<double> > scores(nlabels, std::vector<double>(ncells));
std::vector<const double*> score_ptrs(nlabels);
for (size_t l = 0; l < nlabels; ++l) {
    score_ptrs[l] = scores[l].data();
}
std::vector<double> delta(ncells);

runner.run(
    test_mat.get(),
    ref_mat.get(), 
    ref_labels.data(), 
    ref_markers,
    assignments.data(),
    score_ptrs,
    delta.data()
);
```

See the [reference documentation](https://ltla.github.io/singlepp) for more details.

## Preparing references

A reference dataset should have at least three components:

- The "expression" matrix, where rows are features and columns are reference profiles.
  Only the rank of the expression values are used by **singlepp**, so one could apply any transformation that preserves the ranks.
- A vector of length equal to the number of columns of the matrix, containing the label for each reference profile.
  These labels should be integers from `[0, N)` where `N` is the number of unique labels.
- A vector of vector of integer vectors, containing the chosen marker genes from pairwise comparisons between labels.
  Say that `y` is this object, then `y[i][j][k]` should contain the `k`-th best marker gene that is upregulated in label `i` compared to label `j`. 
  Marker genes should be reported as row indices of the expression matrix.

In practical usage, they will also contain:

- Feature names for each row of the expression matrix.
  This can be used by **singlepp** to match to the features of the test matrix, if the feature sets are not the same.
- Label names, to map the integer labels to something that is meaningful to the user.
  This is not used by **singlepp** itself, which only deals with the integers.

See [here](https://github.com/clusterfork/singlepp-references) for some references that have been formatted in this manner.

## Identifying markers

Given a reference dataset, **singlepp** implements a simple method of identifying marker genes between labels.
This is based on ranking the differences in median log-expression values between labels and is the "classic" method provided in the original **SingleR** package.

```cpp
singlepp::ChooseClassicMarkers mrk;
auto markers = mrk.run(ref_mat.get(), ref_labels.data());
```

`markers` can then be directly used in `Classifier::run()`.
Of course, other marker detection schemes can be used depending on the type of reference dataset;
for single-cell references, users may be interested in some of the differential analysis methods in the [**libscran**](https://github.com/LTLA/libscran) package.

By default, it is expected that the `markers` supplied to `Classifier::run()` has already been filtered to only the top markers for each pairwise comparison.
However, in some cases, it might be more convenient for `markers` to contain a ranking of all genes such that the desired subset of top markers can be chosen later.
This is achieved by calling `Classifier::set_top()` to the desired number of markers per comparison, e.g., for 20 markers:

```cpp
runner.set_top(20);
auto res20 = mrk.run(ref_mat.get(), ref_labels.data());
```

Doing so is roughly equivalent to slicing each vector in `markers` to the top 20 entries before calling `Classifier::run()`.
In fact, calling `set_top()` is the better approach when intersecting feature spaces - see below -
as the top set will not be contaminated by genes that are not present in the test dataset.

## Intersecting feature sets

Often the reference dataset will not have the same features as the test dataset.
To handle this case, users can provide identifiers for the rows of the reference and test matrices.
**singlepp** will then perform classification using the intersection of features.

```cpp
test_names; // vector of feature names of the test data
ref_names; // vector of feature names of the reference data

auto res = runner.run(
    test_mat.get(),
    test_names.data(),
    ref_mat.get(), 
    ref_names.data(),
    ref_labels.data(), 
    ref_markers
);
```

The identifiers can be anything that can be hashed and compared.
These are most commonly `std::string`s.

## Prebuilding the references

For repeated classification with the same reference, advanced users can call `build()` before `run()`.
This ensures that the reference set-up cost is only paid once.

```cpp
auto pre = runner.build(ref_mat.get(), ref_labels.data(), ref_markers);
auto res = runner.run(test_mat.get(), pre);
```

The same approach works when considering an intersection of features,
assuming that all test matrices have the same order of features as in the `test_names`.

```cpp
auto pre2 = runner.build(
    test_names.size(), 
    test_names.data(),
    ref_mat.get(), 
    ref_names.data(),
    ref_labels.data(), 
    ref_markers
);
auto res2 = runner.run(test_mat.get(), pre);
```

## Integrating results across references

To combine results from multiple references, we first need to perform classification within each reference using the `build()` + `run()` approach.
Let's say we have two references A and B:

```cpp
auto preA = runner.build(refA_mat.get(), refA_labels.data(), refA_markers);
auto resA = runner.run(test_mat.get(), preA);

auto preB = runner.build(refB_mat.get(), refB_labels.data(), refB_markers);
auto resB = runner.run(test_mat.get(), preB);
```

We build the integrated references:

```cpp
singlepp::IntegratedBuilder ibuilder;
ibuilder.add(refA_mat.get(), refA_labels.data(), preA);
ibuilder.add(refB_mat.get(), refB_labels.data(), preB);
auto irefs = ibuilder.finish();
```

And then we can finally run the scoring:

```cpp
singlepp::IntegratedScorer iscorer;
auto ires = iscorer.run(test_mat.get(), irefs);
```

## Building projects 

If you're using CMake, you just need to add something like this to your `CMakeLists.txt`:

```
include(FetchContent)

FetchContent_Declare(
  kmeans 
  GIT_REPOSITORY https://github.com/LTLA/singlepp
  GIT_TAG master # or any version of interest
)

FetchContent_MakeAvailable(singlepp)
```

Then you can link to **singlepp** to make the headers available during compilation:

```
# For executables:
target_link_libraries(myexe singlepp)

# For libaries
target_link_libraries(mylib INTERFACE singlepp)
```

## References

Aran D et al. (2019). 
Reference-based analysis of lung single-cell sequencing reveals a transitional profibrotic macrophage.
_Nat. Immunol._ 20, 163-172
