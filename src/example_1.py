from numpy.linalg import svd

from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.externals.joblib import Memory

from stage_A import randomized_power_iteration
from stage_B import direct_svd
from draw import draw_plots, error


mem = Memory(cachedir='.')
data = fetch_olivetti_faces()['data']
# Let's normalise that
data /= (data ** 2).sum(axis=0)


def compute_eig(A, q=0, l=100):
    """
    Compute Eigenvalue using Randomized Power Iteration and direct eigenvalue
    decomposition

    Parameters
    ----------
    """
    Q = randomized_power_iteration(A, l=l, q=q)
    _, e, _ = direct_svd(A, Q)
    e.sort()
    return e[::-1]


def compute_all(A, max_iter=10, verbose=False):
    e0 = 0
    e1 = 0
    e2 = 0
    e3 = 0
    for it in range(max_iter):
        if verbose:
            print "Compute iteration %d" % it
        print "q = 0"
        e0 += compute_eig(A, q=0)
        print "q = 1"
        e1 += compute_eig(A, q=1)
        print "q = 2"
        e2 += compute_eig(A, q=2)
        print "q = 3"
        e3 += compute_eig(A, q=3)

    e0 /= max_iter
    e1 /= max_iter
    e2 /= max_iter
    e3 /= max_iter
    return e0, e1, e2, e3

print "Computing eigenfaces"
_, gt, _ = mem.cache(svd)(data)
gt.sort()
gt = gt[::-1]

print "computing ..."
e0, e1, e2, e3 = mem.cache(compute_all)(data, max_iter=1, verbose=True)

pca = PCA(n_components=100)
pca.fit(data)
p0 = pca.explained_variance_

rpca = RandomizedPCA(n_components=100, iterated_power=1)
rpca.fit(data)
p1 = rpca.explained_variance_

rpca = RandomizedPCA(n_components=100, iterated_power=2)
rpca.fit(data)
p2 = rpca.explained_variance_

rpca = RandomizedPCA(n_components=100, iterated_power=3)
rpca.fit(data)
p3 = rpca.explained_variance_

draw_plots([gt[:100], e0[:100], e1[:100], e2[:100], e3[:100]],
           legend=('grountruth', 'q=0', 'q=1', 'q=2', 'q=3'))
draw_plots([error(gt[:100], e0[:100]),
           error(gt[:100], e1[:100]),
            error(gt[:100], e2[:100]),
            error(gt[:100], e3[:100])],
            legend=('Error 1', 'Error 2', 'Error 3', 'Error 4'))

draw_plots([p0[:100], p1[:100], p2[:100], p3[:100]],
           legend=('grountruth', 'q=0', 'q=1', 'q=2', 'q=3'))
draw_plots([error(p0[:100], p1[:100]),
           error(p0[:100], p2[:100]),
            error(p0[:100], p3[:100])],
            legend=('Error 1', 'Error 2', 'Error 3'))
