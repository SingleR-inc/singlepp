# This runs the overall tests against the naive_single R implementation.
# library(testthat); library(singlepp.tests); source("setup.R"); source("test-single.R")

test_that("single reference, no fine tuning", {
    ngenes <- 5000
    nlabels <- 5

    # Setting up the data.
    set.seed(10000)
    mat <- matrix(rnorm(ngenes * 100), nrow=ngenes)
    ref <- matrix(rnorm(ngenes * 20), nrow=ngenes)
    labels <- mock.labels(ncol(ref), nlabels)
    markers <- mock.markers(ngenes, nlabels)

    res <- naive_single(mat, ref, labels, markers, fine.tune = FALSE)
    obs <- classify_single(mat, ref, labels, markers, fine_tune = FALSE)
    expect_equal(obs$scores, res$scores)
    expect_identical(obs$best, res$best)
    expect_equal(obs$delta, res$delta)
})

test_that("single reference, standard fine tuning", {
    ngenes <- 1000
    nlabels <- 4

    # Setting up the data.
    set.seed(20000)
    mat <- matrix(rnorm(ngenes * 100), nrow=ngenes)
    ref <- matrix(rnorm(ngenes * 20), nrow=ngenes)
    labels <- mock.labels(ncol(ref), nlabels)
    markers <- mock.markers(ngenes, nlabels)

    # Running the naive_single pipeline.
    res <- naive_single(mat, ref, labels, markers)
    obs <- classify_single(mat, ref, labels, markers)
    expect_equal(obs$scores, res$scores)
    expect_identical(obs$best, res$best)
    expect_equal(obs$delta, res$delta)
})

test_that("single reference, tight fine tuning", {
    ngenes <- 1000
    nlabels <- 6

    # Setting up the data.
    set.seed(30000)
    mat <- matrix(rnorm(ngenes * 100), nrow=ngenes)
    ref <- matrix(rnorm(ngenes * 20), nrow=ngenes)
    labels <- mock.labels(ncol(ref), nlabels)
    markers <- mock.markers(ngenes, nlabels)

    # Running the naive_single pipeline.
    res <- naive_single(mat, ref, labels, markers, tune.thresh=0.01)
    obs <- classify_single(mat, ref, labels, markers, tune_thresh=0.01)
    expect_equal(obs$scores, res$scores)
    expect_identical(obs$best, res$best)
    expect_equal(obs$delta, res$delta)
})

test_that("single reference, different top", {
    ngenes <- 1000
    nlabels <- 3

    # Setting up the data.
    set.seed(40000)
    mat <- matrix(rnorm(ngenes * 100), nrow=ngenes)
    ref <- matrix(rnorm(ngenes * 20), nrow=ngenes)
    labels <- mock.labels(ncol(ref), nlabels)
    markers <- mock.markers(ngenes, nlabels)

    # Running the naive_single pipeline.
    res <- naive_single(mat, ref, labels, markers, top = 10)
    obs <- classify_single(mat, ref, labels, markers, top = 10)
    expect_equal(obs$scores, res$scores)
    expect_identical(obs$best, res$best)
    expect_equal(obs$delta, res$delta)
})

test_that("single reference, intersection", {
    nlabels <- 3
    common <- sprintf("GENE_%s", 1:1500)

    # Setting up the data with non-shared genes.
    set.seed(50000)
    ngenes_mat <- 1000
    mat <- matrix(rnorm(ngenes_mat * 100), nrow=ngenes_mat)
    rownames(mat) <- sample(common, ngenes_mat)

    ngenes_ref <- 1200 
    ref <- matrix(rnorm(ngenes_ref * 20), nrow=ngenes_ref)
    rownames(ref) <- sample(common, ngenes_ref)

    labels <- mock.labels(ncol(ref), nlabels)
    markers <- mock.markers(ngenes_ref, nlabels)
    named.markers <- relist(rownames(ref)[unlist(markers)], markers)

    # Running the naive_single pipeline.
    res <- naive_single(mat, ref, labels, named.markers, top = 20)
    obs <- intersect_single(mat, rownames(mat), ref, rownames(ref), labels, markers, top = 20)
    expect_equal(obs$scores, res$scores)
    expect_identical(obs$best, res$best)
    expect_equal(obs$delta, res$delta)
})
