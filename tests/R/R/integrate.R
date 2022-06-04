#' @export
integrate <- function(test, results, refs, labels, markers, quantile = 0.8) {
    scores <- matrix(0, ncol(test), length(results))
    best <- integer(ncol(test))
    delta <- numeric(ncol(test))

    for (i in seq_len(ncol(test))) {
        res <- lapply(results, function(r) r[i])
        cur.markers <- mapply(markers, res, FUN=function(m, l) m[[l]], SIMPLIFY=FALSE)

        keep <- mapply(labels, res, FUN=function(l, r) l==r, SIMPLIFY=FALSE) 
        origins <- rep(seq_along(keep), vapply(keep, sum, 0L))
        all.refs <- mapply(refs, keep, FUN=function(R, k) R[,k,drop=FALSE], SIMPLIFY=FALSE)
        new.ref <- do.call(cbind, all.refs)

        remarkers <- vector("list", length(cur.markers))
        for (j1 in seq_along(remarkers)) {
            remarkers[[j1]] <- vector("list", length(cur.markers))
            for (j2 in seq_along(remarkers[[j1]])) {
                if (j1 == j2) {
                    remarkers[[j1]][[j2]] <- sort(unique(unlist(cur.markers[[j1]])))
                } else {
                    remarkers[[j1]][[j2]] <- integer(0)
                }
            }
        }

        # Turning off fine-tuning so that we get a straightforward calculation of correlations.
        # Also setting top = -1 to avoid truncating the marker list - we want to use the union here.
        out <- run_singlepp(test[,i,drop=FALSE], new.ref, markers=remarkers, labels=origins, fine_tune=FALSE, quantile = quantile, top = -1) 

        scores[i,] <- out$scores
        best[i] <- out$best
        delta[i] <- out$delta
    }

    list(scores = scores, best = best, delta = delta)
}


