# C++ port of SingleR 

This repository contains a C++ port of the [**SingleR**](https://github.com/LTLA/SingleR) R package for automated cell type annotation.
It primarily focuses on the prediction step given a set of references; the preparation of the references themselves is left to the user (see below).
The library contains methods for simple and multi-reference predictions, returning a matrix of scores and labels for each cell in the test dataset.
Each cell is treated independently so the entire process is trivially parallelizable.
