import numpy as np
from scipy import misc

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.externals.joblib import Memory

mem = Memory(cachedir='.')


def get_weight_lena(sigma=50000.):
    """
    Calculates the weight matrix of a small patch of lena's image

    Parameters
    ----------
    sigma: int

    Returns
    -------
    """
    lena = misc.lena()

    lena = lena[:30:, :30]
    patches = []

    for i in range(lena.shape[0] - 7):
        for j in range(lena.shape[1] - 7):
            patches.append(lena[i:i + 6, j:j + 6].flatten())
    dist = np.exp(- euclidean_distances(patches, patches) ** 2 / sigma ** 2)
    thres = dist.copy()
    thres.sort(axis=1)
    thres = thres[:, ::-1][:, 6]

    dist[dist < thres] = 0
    return dist


def hermitian(W):
    """
    """
    m, n = W.shape
    D = np.sqrt(np.diag(1. / W.sum(axis=0)))
    return np.dot(np.dot(D, W), D)


# Get a large, noisy, and sparse matrice
W = mem.cache(get_weight_lena)()
A = mem.cache(hermitian)(W)
