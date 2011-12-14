import numpy as np
from numpy.linalg import eig
from scipy.linalg import svd, qr

from ID_dec import compute_ID


def row_extraction_svd(A, Q, j=100):
    """
    Computes the SVD via Row Extraction

    Parameters
    ----------
    A: array

    Q: array

    Returns
    -------
    U, S, V.T
    """
    # FIXME find J and X by computing an ID on Q
    idxs, X = compute_ID(Q)
    Aj = A[idxs, :]
    R, W = qr(Aj)
    Z = np.dot(X, R)
    U, S, V = svd(Z)
    V = np.dot(W, V.T)
    return U, S, V.T


def direct_svd(A, Q):
    """
    Performs a direct SVD

    Parameters
    ----------

    Returns
    -------
    """
    B = np.dot(Q.T, A)
    U, S, V = svd(B)
    return np.dot(Q, U), S, V


def direct_eigenvalue_dec(A, Q):
    B = np.dot(np.dot(Q.T, A), Q)
    L, V = eig(B)
    return L, np.dot(Q, V)
