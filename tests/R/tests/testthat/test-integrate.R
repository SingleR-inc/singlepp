# This runs the integration tests against the reference R implementation.
# library(testthat); library(singlepp.tests); source("setup.R"); source("test-integrate.R")

test_that("integration results match up with the R implementation", {
    ngenes <- 5000

    # Setting up the data.
    set.seed(10000)
    mat <- matrix(rnorm(ngenes * 100), nrow=ngenes)

    refs <- labels <- markers <- results <- vector("list", 3)
    for (i in seq_along(refs)) {
        nlabels <- 5 + i
        nprofiles <- 10 * i
        refs[[i]] <- matrix(rnorm(ngenes * nprofiles), nrow=ngenes)
        labels[[i]] <- mock.labels(nprofiles, nlabels)
        markers[[i]] <- mock.markers(ngenes, nlabels, ntop = 20)
        results[[i]] <- sample(nlabels, ncol(mat), replace=TRUE)
    }

    ref <- integrate(mat, results, refs, labels, markers)
    obs <- integrate_singlepp(mat, results, refs, labels, markers)

    expect_identical(ref$best, obs$best)
    expect_equal(ref$scores, obs$scores)
    expect_equal(ref$delta, obs$delta)
})

test_that("integration results match up with the R implementation (different quantile)", {
    ngenes <- 5000

    # Setting up the data.
    set.seed(10000)
    mat <- matrix(rnorm(ngenes * 100), nrow=ngenes)

    refs <- labels <- markers <- results <- vector("list", 3)
    for (i in seq_along(refs)) {
        nlabels <- 5 + i
        nprofiles <- 10 * i
        refs[[i]] <- matrix(rnorm(ngenes * nprofiles), nrow=ngenes)
        labels[[i]] <- mock.labels(nprofiles, nlabels)
        markers[[i]] <- mock.markers(ngenes, nlabels, ntop = 20)
        results[[i]] <- sample(nlabels, ncol(mat), replace=TRUE)
    }

    ref <- integrate(mat, results, refs, labels, markers, quantile = 0.7)
    obs <- integrate_singlepp(mat, results, refs, labels, markers, quantile = 0.7)

    expect_identical(ref$best, obs$best)
    expect_equal(ref$scores, obs$scores)
    expect_equal(ref$delta, obs$delta)
})
