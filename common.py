import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.preprocessing import minmax_scale


def windows(X, width, skip_last):
    """Generate rolling windows of data."""
    ret = []
    n = X.shape[0]
    for i in range(n - width + 1 - skip_last):
        window = X[i:i + width, :]
        ret.append([tuple(x) for x in window[:]])
    return np.array(ret)


def windows_to_inputs(X_windows, num_concepts, skip_last):
    """Split windowed data into separate inputs for each feature."""
    n = X_windows.shape[0]
    return {f"concept_{i}": X_windows[:(-1 if skip_last else n), :, i] for i in range(num_concepts)}


def expand_derivatives(X):
    """Add derivatives to M x N array of data, making it (M - 1) x 2N and normalizing it to [0, 1]."""
    dX = X[1:, :] - X[:-1, :]
    X = np.hstack([X[1:, :], dX])
    return minmax_scale(X)


def to_concepts(X, centroids):
    """Map an array of data to concept space using the provided cluster centers."""
    X_fuzzy, *_ = fuzz.cluster.cmeans_predict(X.T, centroids, 2, error=0.005, maxiter=1000, init=None)
    return X_fuzzy.T


def load_file(path):
    """Read and normalize data from a CSV file."""
    df = pd.read_csv(path, header=None)
    return minmax_scale(df.values)
