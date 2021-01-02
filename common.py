import numpy as np
import skfuzzy as fuzz
from sklearn.preprocessing import minmax_scale


def to_6D_space(X):
    dX = X[1:, :] - X[:-1, :]
    X = np.hstack([X[1:, :], dX])
    return minmax_scale(X)


def to_concepts(X, centroids):
    X_fuzzy, *_ = fuzz.cluster.cmeans_predict(X.T, centroids, 2, error=0.005, maxiter=1000, init=None)
    return X_fuzzy.T
