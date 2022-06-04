mock.markers <- function(ngenes, nlabels, ntop=Inf) {
    markers <- vector("list", nlabels)
    for (m in seq_len(nlabels)) {
        current <- vector("list", nlabels)
        for (n in seq_len(nlabels)) {
            if (m == n) {
                current[[n]] <- integer(0)
            } else {
                current[[n]] <- sample(seq_len(ngenes))
            }
            if (is.finite(ntop)) {
                current[[n]] <- head(current[[n]], ntop)
            }
        }
        markers[[m]] <- current
    }
    markers
}

mock.labels <- function(ncells, nlabels) {
    labels <- sample(nlabels, ncells, replace=TRUE)
    labels[1:nlabels] <- 1:nlabels # guarantee at least one of each.
    sample(labels)
}
