import numpy as np
from scipy.linalg import svd, qr

from sklearn.datasets import fetch_olivetti_faces
from sklearn.externals.joblib import Memory

from stage_A import randomized_range_finder, fast_randomized_range_finder
from stage_B import direct_svd, row_extraction_svd
from draw import draw_plots, error
from test_image import get_weight_lena

mem = Memory(cachedir='.')

# Size of the matrix
mu, sigma = 1, 1
#gaussian = np.random.normal(size=(m, n))
#gaussian = np.load('gaussian.npy')
gaussian = get_weight_lena()
n, m = gaussian.shape
U_ground_truth, S_ground_truth, V_ground_truth = svd(gaussian)


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
def algo_2(A, l=8):
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
    Q, R = randomized_range_finder(A, l=l)
    return direct_svd(A, Q)


def algo_3(A, l=8, method='srft', verbose=False):
    """
    Computes the third algorithm
    """
    if verbose:
        print "Computing Fast Randomized Range Finder"
    Q, _ = fast_randomized_range_finder(A, l=l, method=method)
    if verbose:
        print "Compute direct SVD"
#    return row_extraction_svd(A, Q)
    return direct_svd(A, Q)


U1, S1, V1 = mem.cache(algo_1)(gaussian)
U2, S2, V2 = mem.cache(algo_2)(gaussian, l=100)
U3, S3, V3 = mem.cache(algo_3)(gaussian, l=100)

draw_plots([S_ground_truth[:100], S1[:100], S2[:100], S3[:100]],
     legend=('Ground truth', 'Algo1', 'Algo2', 'Algo3'),
     logscale=False)
draw_plots([error(S_ground_truth[:100], S1[:100]),
      error(S_ground_truth[:100], S2[:100]),
      error(S_ground_truth[:100], S3[:100])],
      legend=('Error 1', 'Error 2', 'Error 3'))
