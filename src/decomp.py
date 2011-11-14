import numpy as np
import scipy as sp


def randomized_range_finder(X, l=8):
    """
    Algorithm 4.1

    params
    -------
        X, ndarray
    """
    m , n = X.shape
    mu, sigma = 2, 0.5
    v = np.random.normal(mu, sigma, (m, l))
    Y = np.dot(X, v)
    Q, R = np.linalg.qr(Y)
    return Q, R
   

def adaptative_randomized_range_finder(X, l):
    pass


if __name__ == "__main__":
    X = sp.misc.lena()

    calc = X > 125
    X[calc] = 0


