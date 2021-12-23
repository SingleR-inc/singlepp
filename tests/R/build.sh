# This pulls in the same external libraries and makes them available to Rcpp's
# compilation. We don't use RcppEigen because the version there is too old to
# contain the features used by irlba.

set -e
set -u

for lib in aarand annoy kmeans knncolle tatami
do
    ln -s ../../build/_deps/${lib}-src/include/${lib}
done

ln -s ../../build/_deps/hnswlib-src/hnswlib 
ln -s ../../include/singlepp 
