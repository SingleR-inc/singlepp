#' @export
#' @importFrom stats cor quantile
#' @importFrom utils head
naive_single <- function(test, ref, labels, markers, top=20, quantile = 0.8, fine.tune = TRUE, tune.thresh = 0.05) {
    if (!is.null(rownames(test))) {
        # Intersecting ahead of everything else, so that 'top' has the intended effect.
        common <- intersect(rownames(ref), rownames(test))
        test <- test[common,,drop=FALSE]
        ref <- ref[common,,drop=FALSE]

        for (i in seq_along(markers)) {
            for (j in seq_along(markers[[i]])) {
                m <- match(markers[[i]][[j]], common)
                markers[[i]][[j]] <- m[!is.na(m)]
            }
        }
    }

    y <- split(seq_along(labels), labels)
    names(y) <- NULL

    for (i in seq_along(markers)) {
        for (j in seq_along(markers[[i]])) {
            markers[[i]][[j]] <- head(markers[[i]][[j]], top)
        }
    }

    collected <- matrix(0, ncol(test), length(y))
    genes <- sort(unique(unlist(markers)))

    for (x in seq_along(y)) {
        corrs <- stats::cor(ref[genes,y[[x]]], test[genes,], method="spearman")
        collected[,x] <- apply(corrs, 2, FUN=stats::quantile, prob=quantile)
    }

    if (fine.tune) {
        references <- lapply(y, function(i) ref[,i,drop=FALSE])
        best <- integer(ncol(test))
        delta <- numeric(ncol(test))
        de.info <- do.call(cbind, markers)

        for (i in seq_len(ncol(test))) {
            info <- .fine_tune_cell(i, test, collected, references, de.info, quantile = quantile, tune.thresh = tune.thresh)
            best[i] <- info$best
            delta[i] <- info$delta
        }
    } else {
        best <- max.col(collected, ties.method="first")
        delta <- apply(collected, 1, function(x) diff(sort(-x)[1:2]))
    }

    list(scores = collected, best = best, delta = delta)
}

.fine_tune_cell <- function(i, test, scores, references, de.info, quantile, tune.thresh) {
    cur.exprs <- test[,i]
    cur.scores <- scores[i,]
    top.labels <- which(cur.scores >= max(cur.scores) - tune.thresh)
    old.labels <- integer(0)
    full.scores <- cur.scores

    # Need to compare to old.labels, to avoid an infinite loop
    # if the correlations are still close after fine tuning.
    while (length(top.labels) > 1L && !identical(top.labels, old.labels)) {
        all.combos <- expand.grid(top.labels, top.labels)
        common <- Reduce(union, de.info[as.matrix(all.combos)])
        cur.scores <- .compute_label_scores_manual(common, top.labels, cur.exprs, references, quantile=quantile)

        old.labels <- top.labels
        keep <- !is.na(cur.scores)
        cur.scores <- cur.scores[keep]
        top.labels <- old.labels[keep]
        full.scores <- cur.scores

        keep <- cur.scores >= max(cur.scores) - tune.thresh
        cur.scores <- cur.scores[keep]
        top.labels <- top.labels[keep]
    }

    if (length(top.labels)==1L) {
        label <- top.labels
    } else if (length(top.labels)==0L) {
        label <- NA_integer_
    } else {
        label <- top.labels[which.max(cur.scores)]
    }
    list(best=label, delta=diff(sort(-full.scores))[1])
}

#' @importFrom stats cor quantile
.compute_label_scores_manual <- function(common, top.labels, cur.exprs, references, quantile) {
    cur.exprs <- cur.exprs[common]
    cur.scores <- numeric(length(top.labels))

    for (u in seq_along(top.labels)) {
        ref <- references[[top.labels[[u]]]]
        ref <- as.matrix(ref[common,,drop=FALSE])
        cur.cor <- cor(cur.exprs, ref, method="spearman")
        cur.scores[u] <- quantile(cur.cor, probs=quantile)
    }

    cur.scores
}
