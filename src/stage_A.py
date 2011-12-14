import numpy as np
from scipy.linalg import qr


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


def fast_randomized_range_finder(a, l=100, method='srft', verbose=False):
    """
    """
    m, n = a.shape
    if method == 'srft':
        if verbose:
            print "Computing SRFT matrix"
        o = SRFT(n, l)
    elif method == 'gsrft':
        if verbose:
            print "Computing GSRFT matrix"
        o = givensSRFT(n, l)
    else:
        print "Unknown method"
    Y = np.dot(a, o)
    print "Computing QR decomposition"
    return qr(Y, mode='qr')


def randomized_power_iteration(A, l=100, q=4):
    """
    Performs the Randomized Power Iteration

    Algorithm 4.3

    Parameters
    ----------
    A: array
        matrix to perform the iteration upon

    l: int

    q: int
        number of power iteration to perform

    Returns
    -------
        Q, R: array, array
    """
    m, n = A.shape
    l = 100
    q = 4
    o = np.random.normal(size=(n, l))

    Y = np.dot(np.dot(np.dot(A, A.T) ** q, A), o)
    Q, _ = qr(Y)
    return Q


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
