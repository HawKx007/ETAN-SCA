import numpy as np

def zscore_fit(X: np.ndarray):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    return mu, sd

def zscore_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return (X - mu) / sd

def describe_data(X, y):
    return {
        "X_shape": tuple(X.shape),
        "y_shape": tuple(y.shape),
        "y_min": int(np.min(y)),
        "y_max": int(np.max(y)),
        "num_classes": int(len(np.unique(y))),
    }