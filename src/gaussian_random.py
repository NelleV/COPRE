import numpy as np
from scipy.linalg import svd, qr

from sklearn.externals.joblib import Memory

from decomp import randomized_range_finder

mem = Memory(cachedir='.')

# Size of the matrix
m, n = 500, 500
mu, sigma = 1, 1

gaussian = np.load('gaussian.npy')
U_ground_truth, S_ground_truth, V_ground_truth = svd(gaussian)


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


# Algo 1 -> QR factorization and algo 5.1
def algo_1(A):
    """
    Computes the first algorithm

    Step A: rank revealing QR, suing wolumn pivoting and HouseHolder reflector
    Step B: algo 5.1 (direct SVD)

    Parameters
    ----------
    A: array

    Returns
    -------
    U, S, V : SVD decomposition
    """
    Q, R = qr(A, mode='qr')

    return direct_svd(A, Q)


# Algo 2
def algo_2(A):
    """
    Computes the second algorithm

    Step A: Direct range finder
    Step B: Direct SVD

    Parameters
    ----------
    A: array

    Returns
    -------
    U, S, V : SVD decomposition

    """
    Q, R = randomized_range_finder(A)
    return direct_svd(A, Q)


def random_permutation_matrix(n, truncated=None):
    """
    Draw a random permutation matrix of size (n, n)

    Parameters
    ----------
    n: int
       size of the matrix to return

    truncated: int, default None
        if not None, truncates the random permutation

    Returns
    -------
    A: random permutation matrix of shape (n, n)
    """
    A = np.identity(n)
    np.random.shuffle(A)
    if truncated is not None:
        A = A[:truncated]
    return A


def UDFT(n):
    """
    Computes the unitary DFT matrix of size n

    Parameters
    ----------
    n: int
        size of the matrix to compute
    """
    p = np.arange(0, n).repeat(n).reshape((n, n))
    F = np.sqrt(n) * np.exp(-2 * 1j * np.pi * p * p.T / n)
    return F


def random_complex_diag(n):
    """
    Computes a (n, n) diagonal matrix where entries are randomly distributed
    on the complex united circle

    Parameters
    ----------
    n: int
        size of the matrix to compute
    """
    a = np.random.normal(size=(n,)) + 1j * np.random.normal(size=(n,))
    return np.diag(a / np.linalg.norm(a))


def givens_rotation(n, i, j, theta):
    """
    Computes the Givens Rotation matrix

    Parameters
    ----------
    n: int
        size of the Givens matrix to compute

    i: int
        first index

    j: int
        second index

    theta: int
        angle of the Givens rotation
    """
    givens_rotation = np.identity(n)
    givens_rotation[i, i] = np.cos(theta)
    givens_rotation[j, j] = np.cos(theta)
    givens_rotation[i, j] = np.sin(theta)
    givens_rotation[j, i] = - np.sin(theta)
    return givens_rotation


U1, S1, V1 = mem.cache(algo_1)(gaussian)
U2, S2, V2 = mem.cache(algo_2)(gaussian)
