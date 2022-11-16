import logging
import numpy as np

logger = logging.getLogger("searcher")
logger.setLevel(logging.INFO)


def pairwise_sparse_jaccard_distance(X, Y=None):
    """
    Computes the Jaccard distance between two sparse matrices or between all pairs in
    one sparse matrix.

    Args:
        X (scipy.sparse.csr_matrix): A sparse matrix.
        Y (scipy.sparse.csr_matrix, optional): A sparse matrix.

    Returns:
        numpy.ndarray: A similarity matrix.
    """

    if Y is None:
        Y = X

    assert X.shape[1] == Y.shape[1]

    X, Y = (
        X.astype(bool).astype(int),
        Y.astype(bool).astype(int),
    )

    intersect = X.dot(Y.T)

    x_sum, y_sum = X.sum(axis=1).A1, Y.sum(axis=1).A1
    xx, yy = np.meshgrid(x_sum, y_sum)
    union = (xx + yy).T - intersect

    return (1 - intersect / union).A
