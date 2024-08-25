#' @export
naive_integrate <- function(test, results, refs, labels, markers, quantile = 0.8, fine.tune = TRUE, tune.thresh = 0.05) {
    scores <- matrix(0, ncol(test), length(results))
    best <- integer(ncol(test))
    delta <- numeric(ncol(test))

    for (i in seq_len(ncol(test))) {
        cur.markers <- vector("list", length(refs))
        for (r in seq_along(refs)) {
            curbest <- results[[r]][i]
            cur.markers[[r]] <- sort(unique(unlist(markers[[r]][[curbest]])))
        }

        common <- sort(unique(unlist(cur.markers)))
        curtest <- superslice(test, common, i, drop=TRUE)
        collected <- numeric(length(refs))

        for (r in seq_along(refs)) {
            curbest <- results[[r]][i]
            keep <- labels[[r]] == curbest
            curref <- superslice(refs[[r]], common, keep, drop=FALSE)
            corrs <- missing.cor(curref, curtest)
            collected[r] <- stats::quantile(corrs, prob=quantile)
        }

        scores[i,] <- collected

        if (!fine.tune) {
            best[i] <- which.max(collected)
            delta[i] <- diff(sort(-collected)[1:2])
        } else {
            prediction <- which.max(collected)
            tuned <- which(collected >= max(collected) - tune.thresh)

            while (length(tuned) > 1 && length(tuned) < length(collected)) {
                cur.markers <- vector("list", length(tuned))
                for (r in tuned) {
                    curbest <- results[[r]][i]
                    cur.markers[[r]] <- sort(unique(unlist(markers[[r]][[curbest]])))
                }

                common <- sort(unique(unlist(cur.markers)))
                curtest <- superslice(test, common, i, drop=TRUE)
                collected <- numeric(length(tuned))

                for (t in seq_along(tuned)) {
                    r <- tuned[t]
                    curbest <- results[[r]][i]
                    keep <- labels[[r]] == curbest
                    curref <- superslice(refs[[r]], common, keep, drop=FALSE)
                    corrs <- missing.cor(curref, curtest)
                    collected[t] <- stats::quantile(corrs, prob=quantile)
                }

                prediction <- tuned[which.max(collected)]
                tuned <- tuned[which(collected >= max(collected) - tune.thresh)]
            }

            best[i] <- prediction
            delta[i] <- diff(sort(-collected)[1:2])
        }
    }

    list(scores = scores, best = best, delta = delta)
}

missing.cor <- function(x, y) {
    output <- numeric(ncol(x))
    for (i in seq_len(ncol(x))) {
        curx <- x[,i]
        keep <- !is.na(curx) & !is.na(y)
        output[i] <- stats::cor(curx[keep], y[keep], method="spearman")
    }
    output
}

superslice <- function(x, i, j, drop=FALSE) {
    i0 <- if (is.character(i)) match(i, rownames(x)) else i
    x[i0,j,drop=drop]
}
