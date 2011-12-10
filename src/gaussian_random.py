import numpy as np
from scipy.linalg import svd, qr
from matplotlib import pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.externals.joblib import Memory


mem = Memory(cachedir='.')

# Size of the matrix
mu, sigma = 1, 1
#gaussian = np.random.normal(size=(m, n))
gaussian = np.load('gaussian.npy')
#gaussian = fetch_olivetti_faces()['data']
n, m = gaussian.shape
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


def algo_3(A, l=8, method='srft'):
    """
    Computes the third algorithm
    """
    Q, _ = fast_randomized_range_finder(A, l=l, method=method)
    return direct_svd(A, Q)


def randomized_range_finder(X, l=8):
    """
    Algorithm 4.1

    params
    -------
        X, ndarray
    """
    m, n = X.shape
    mu, sigma = 1, 1
    v = np.random.normal(mu, sigma, (n, l))
    Y = np.dot(X, v)
    Q, R = np.linalg.qr(Y)
    return Q, R


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
        A = A[:, :truncated]
    return A


def UDFT(n):
    """
    Computes the unitary discrete transform (DFT) matrix of size n

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


    Returns
    -------
    givens_rotation: matrix
        Givens rotation i, j with angle theta
    """
    givens_rotation = np.identity(n)
    givens_rotation[i, i] = np.cos(theta)
    givens_rotation[j, j] = np.cos(theta)
    givens_rotation[i, j] = np.sin(theta)
    givens_rotation[j, i] = - np.sin(theta)
    return givens_rotation


def SRFT(n, l):
    """
    Computes a subsampled random Fourier Transform matrix

    Parameters
    ----------
    n: int
        number of lines of the matrix

    l: int
        number of lines of the matrix

    Returns
    -------
    array (n, l)
        subsampled random fourier transform matrix
    """
    D = random_complex_diag(n)
    F = UDFT(n)
    R = random_permutation_matrix(n, truncated=l)
    return np.sqrt(n / l) * np.dot(np.dot(D, F), R)


def chain_random_givens_rotation(n):
    """
    Computes a chain of random Givens rotation Matrices

    Parameters:
    -----------
    n: int
        size of the chain of random givens matrices

    Returns:
    --------
    array (n, n)
        Chain of Random Givens Rotation Matrices
    """

    P = random_permutation_matrix(n)
    for i in xrange(n - 1):
        P = np.dot(P, givens_rotation(n, i, i + 1, np.random.normal()))
    return P


def givensSRFT(n, l):
    """
    Computes a subsampled random Fourier Transform matrix

    Parameters
    ----------
    n: int
        number of lines of the matrix

    l: int
        number of lines of the matrix

    Returns
    -------

    """

    D1 = random_complex_diag(n)
    D2 = random_complex_diag(n)
    D3 = random_complex_diag(n)

    F = UDFT(n)
    R = random_permutation_matrix(n, truncated=l)

    theta1 = chain_random_givens_rotation(n)
    theta2 = chain_random_givens_rotation(n)

    a = np.dot(np.dot(np.dot(D3, theta2), D2), theta1)
    return np.dot(np.dot(np.dot(a, D1), F), R)


def fast_randomized_range_finder(a, l=100, method='srft'):
    """
    """
    m, n = a.shape
    if method == 'srft':
        o = SRFT(n, l)
    elif method == 'gsrft':
        o = USRFT(n, l)
    else:
        print "Unknown method"
    Y = np.dot(a, o)
    return qr(Y, mode='qr')


def draw(elements, legend=None):
    """
    Draw elements
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    for element in elements:
        ax.plot(element)
    if legend:
        ax.legend(legend)
    fig.show()


def error(ground_truth, S):
    """
    Calculate the error
    """
    err = np.sqrt((ground_truth - S) ** 2)
    return err

U1, S1, V1 = mem.cache(algo_1)(gaussian)
U2, S2, V2 = mem.cache(algo_2)(gaussian, l=100)
U3, S3, V3 = mem.cache(algo_3)(gaussian, l=100)

draw([S_ground_truth[:100], S1[:100], S2[:100], S3[:100]],
     legend=('Ground truth', 'Algo1', 'Algo2', 'Algo3'))
draw([error(S_ground_truth[:100], S1[:100]),
      error(S_ground_truth[:100], S2[:100]),
      error(S_ground_truth[:100], S3[:100])],
      legend=('Error 1', 'Error 2', 'Error 3'))
