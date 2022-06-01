# This runs the overall tests against the reference R implementation.
# library(testthat); library(singlepp.tests); source("test-overall.R")

library(SingleR)
library(testthat)

run_singlepp <- singlepp.tests:::run_singlepp

test_that("results match up with SingleR (basic)", {
    # Setting up the data.
    set.seed(10000)
    mat <- matrix(rnorm(500000), nrow=5000)
    ref <- matrix(rnorm(100000), nrow=5000)
    labels <- sample(5, ncol(ref), replace=TRUE)
    rownames(mat) <- rownames(ref) <- paste0("GENE_", seq_len(nrow(mat)))

    # Running the reference SingleR pipeline.
    res <- SingleR(mat, ref, labels)

    # Reindexing the markers.
    markers <- metadata(res)$de.genes
    markers <- relist(match(unlist(markers, use.names=FALSE), rownames(mat)), markers)

    obs <- run_singlepp(mat, ref, labels, markers)
    expect_equal(obs$scores, unname(res$scores))
    expect_identical(obs$best, as.integer(res$labels))
})

test_that("results match up with SingleR (fine-tuning)", {
    # Setting up the data.
    set.seed(10000)
    mat <- matrix(rnorm(500000), nrow=5000)
    ref <- matrix(rnorm(100000), nrow=5000)
    labels <- sample(5, ncol(ref), replace=TRUE)
    rownames(mat) <- rownames(ref) <- paste0("GENE_", seq_len(nrow(mat)))

    # Setting up random markers. This encourages some activity during
    # fine-tuning, otherwise properly detected markers are too good to warrant
    # fine-tuning.
    markers <- vector("list", 5)
    names(markers) <- seq_along(markers)
    for (i in seq_along(markers)) {
        curmarkers <- vector("list", 5)
        names(curmarkers) <- seq_along(curmarkers)
        for (j in seq_along(curmarkers)) {
            if (i!=j) {
                curmarkers[[j]] <- sample(rownames(mat), 20)
            } else {
                curmarkers[[j]] <- character(0)
            }
        }
        markers[[i]] <- curmarkers
    }

    res <- SingleR(mat, ref, labels, genes=markers)
    res2 <- SingleR(mat, ref, labels, genes=markers, fine.tune=FALSE)
    expect_false(identical(res$labels, res2$labels)) # i.e., fine-tuning has an effect

    # Reindexing the markers.
    markers2 <- relist(match(unlist(markers, use.names=FALSE), rownames(mat)), markers)

    obs <- run_singlepp(mat, ref, labels, markers2)
    expect_equal(obs$scores, unname(res$scores))
    expect_identical(obs$best, as.integer(res$labels))
})
