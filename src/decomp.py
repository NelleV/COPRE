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
    mu, sigma = 1, 1
    v = np.random.normal(mu, sigma, (m, l))
    Y = np.dot(X, v)
    Q, R = np.linalg.qr(Y)
    return Q, R
   

def adaptative_randomized_range_finder(X, r=10, tol=1e-10):
    m , n = X.shape
    mu, sigma = 1, 1
    v = np.random.normal(mu, sigma, (m, l))
    Y = np.dot(X, v)
    j = 0
    while np.sqrt(Y[j, j+r] ** 2).max() > tol / (10 * np.sqrt(2. / np.pi)):
        j += 1
    


if __name__ == "__main__":
    X = sp.misc.lena()
    calc = X > 125
    X[calc] = 0

    Q_1, R_1 = randomized_range_finder(X, l=10)

    m , n = X.shape
    mu, sigma = 1, 1
    l = 10
    v = np.random.normal(mu, sigma, (m, l))
    Y = np.dot(X, v).T
    Q = None
    r = 10
    tol = 1e-10
    max_iter= 50
    for it in range(max_iter):
        if np.sqrt((Y ** 2).sum(axis=0)).max() < tol / (10 * np.sqrt(2. / np.pi)):
            print "break at iteration %d" % it
            break
        if Q is not None:
            Y[0, :] = np.dot((np.identity(m) - np.dot(Q, Q.T)), Y[0, :])
        q = Y[0, :] / np.sqrt((Y[0, :]**2).sum())
        if Q is None:
            Q = q.copy()
            Q = Q.reshape(q.shape[0], 1)
        else:
            Q = np.concatenate((Q, q), axis=1)
        o = np.random.normal(mu, sigma, (m,))
        Y[-1, :] = np.dot((np.identity(m) - np.dot(Q, Q.T)), np.dot(X, o))
        for i in range(0, r - 1):
            Y[i, :] = Y[i + 1, :] - Q[i + 1] *  np.dot(Q[i + 1], Y[i + 1, :])

