mock.pairwise.markers <- function(ngenes, nlabels, ntop) {
    markers <- vector("list", nlabels)
    for (m in seq_len(nlabels)) {
        current <- vector("list", nlabels)
        for (n in seq_len(nlabels)) {
            if (m == n) {
                current[[n]] <- integer(0)
            } else {
                current[[n]] <- sample(ngenes, ntop)
            }
        }
        markers[[m]] <- current
    }
    markers
}

mock.per.label.markers <- function(ngenes, nlabels, ntop) {
    markers <- vector("list", nlabels)
    for (m in seq_len(nlabels)) {
        markers[[m]] <- sample(ngenes, ntop)
    }
    markers
}

mock.labels <- function(ncells, nlabels) {
    labels <- sample(nlabels, ncells, replace=TRUE)
    labels[1:nlabels] <- 1:nlabels # guarantee at least one of each.
    sample(labels)
}
