# This runs the integration tests against the reference R implementation.
# library(testthat); library(singlepp.tests); source("setup.R"); source("test-integrate.R")

test_that("integrated references, basic", {
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

    ref <- naive_integrate(mat, results, refs, labels, markers)
    obs <- classify_integrate(mat, results, refs, labels, markers)

    expect_identical(ref$best, obs$best)
    expect_equal(ref$scores, obs$scores)
    expect_equal(ref$delta, obs$delta)
})

test_that("integrated references, different quantile", {
    ngenes <- 5000

    # Setting up the data.
    set.seed(20000)
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

    ref <- naive_integrate(mat, results, refs, labels, markers, quantile = 0.7)
    obs <- classify_integrate(mat, results, refs, labels, markers, quantile = 0.7)

    expect_identical(ref$best, obs$best)
    expect_equal(ref$scores, obs$scores)
    expect_equal(ref$delta, obs$delta)
})

test_that("integrated references, intersection", {
    all.ngenes <- 5000
    all.genes <- sprintf("GENE_%i", seq_len(all.ngenes))

    # Setting up the data.
    set.seed(30000)
    ngenes_mat <- 4500
    mat <- matrix(rnorm(ngenes_mat * 100), nrow=ngenes_mat)
    rownames(mat) <- sample(all.genes, ngenes_mat)

    refs <- labels <- markers <- results <- named.markers <- vector("list", 3)
    for (i in seq_along(refs)) {
        nlabels <- 5 + i
        nprofiles <- 10 * i
        ngenes_ref <- 4000 + i * 100

        refs[[i]] <- matrix(rnorm(ngenes_ref * nprofiles), nrow=ngenes_ref)
        rownames(refs[[i]]) <- sample(all.genes, ngenes_ref)

        markers[[i]] <- mock.markers(ngenes_ref, nlabels, ntop = 20)
        named.markers[[i]] <- relist(rownames(refs[[i]])[unlist(markers[[i]])], markers[[i]])

        labels[[i]] <- mock.labels(nprofiles, nlabels)
        results[[i]] <- sample(nlabels, ncol(mat), replace=TRUE)
    }

    ref <- naive_integrate(mat, results, refs, labels, named.markers, quantile = 0.7)
    obs <- intersect_integrate(mat, rownames(mat), results, refs, lapply(refs, rownames), labels, markers, quantile = 0.7)

    expect_identical(ref$best, obs$best)
    expect_equal(ref$scores, obs$scores)
    expect_equal(ref$delta, obs$delta)
})
