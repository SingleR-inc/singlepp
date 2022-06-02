# This runs the overall tests against the reference R implementation.
# library(testthat); library(singlepp.tests); source("setup.R"); source("test-overall.R")

test_that("results matches up with the reference R implementation (no fine tuning)", {
    ngenes <- 5000
    nlabels <- 5

    # Setting up the data.
    set.seed(10000)
    mat <- matrix(rnorm(ngenes * 100), nrow=ngenes)
    ref <- matrix(rnorm(ngenes * 20), nrow=ngenes)
    labels <- mock.labels(ncol(ref), nlabels)
    markers <- mock.markers(ngenes, nlabels)

    res <- reference(mat, ref, labels, markers, fine.tune = FALSE)
    obs <- run_singlepp(mat, ref, labels, markers, fine_tune = FALSE)
    expect_equal(obs$scores, res$scores)
    expect_identical(obs$best, res$best)
    expect_equal(obs$delta, res$delta)
})

test_that("results matches up with the reference R implementation (standard fine tuning)", {
    ngenes <- 1000
    nlabels <- 4

    # Setting up the data.
    set.seed(20000)
    mat <- matrix(rnorm(ngenes * 100), nrow=ngenes)
    ref <- matrix(rnorm(ngenes * 20), nrow=ngenes)
    labels <- mock.labels(ncol(ref), nlabels)
    markers <- mock.markers(ngenes, nlabels)

    # Running the reference pipeline.
    res <- reference(mat, ref, labels, markers)
    obs <- run_singlepp(mat, ref, labels, markers)
    expect_equal(obs$scores, res$scores)
    expect_identical(obs$best, res$best)
    expect_equal(obs$delta, res$delta)
})

test_that("results matches up with the reference R implementation (tight fine tuning)", {
    ngenes <- 1000
    nlabels <- 6

    # Setting up the data.
    set.seed(30000)
    mat <- matrix(rnorm(ngenes * 100), nrow=ngenes)
    ref <- matrix(rnorm(ngenes * 20), nrow=ngenes)
    labels <- mock.labels(ncol(ref), nlabels)
    markers <- mock.markers(ngenes, nlabels)

    # Running the reference pipeline.
    res <- reference(mat, ref, labels, markers, tune.thresh=0.01)
    obs <- run_singlepp(mat, ref, labels, markers, tune_thresh=0.01)
    expect_equal(obs$scores, res$scores)
    expect_identical(obs$best, res$best)
    expect_equal(obs$delta, res$delta)
})

test_that("results matches up with the reference R implementation (different top)", {
    ngenes <- 1000
    nlabels <- 3

    # Setting up the data.
    set.seed(40000)
    mat <- matrix(rnorm(ngenes * 100), nrow=ngenes)
    ref <- matrix(rnorm(ngenes * 20), nrow=ngenes)
    labels <- mock.labels(ncol(ref), nlabels)
    markers <- mock.markers(ngenes, nlabels)

    # Running the reference pipeline.
    res <- reference(mat, ref, labels, markers, top = 10)
    obs <- run_singlepp(mat, ref, labels, markers, top = 10)
    expect_equal(obs$scores, res$scores)
    expect_identical(obs$best, res$best)
    expect_equal(obs$delta, res$delta)
})
