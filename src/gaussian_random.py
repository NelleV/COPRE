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


U1, S1, V1 = mem.cache(algo_1)(gaussian)

# Algo 2
def algo_2(A):
    """
    Computes the second algorithm

    Step A: Direct range finder
    Step B: Direct SVD
    """
    Q, R = randomized_range_finder(A)
    U, S, V = direct_svd(A, Q)
