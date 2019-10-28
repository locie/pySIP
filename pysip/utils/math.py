import numpy as np
from scipy.linalg import solve_triangular


def log1p_exp(a):
    """Calculates the log of 1 plus the exponential of the specified value without overflow"""
    if a > 0.0:
        return a + np.log1p(np.exp(-a))
    return np.log1p(np.exp(a))


def log_sum_exp(a, b):
    """Robust sum on log-scale"""
    if np.isneginf(a):
        return b
    if np.isposinf(a) and np.isposinf(b):
        return np.inf
    if a > b:
        return a + log1p_exp(b - a)
    return b + log1p_exp(a - b)


def cholesky_inverse(matrix):
    """Borrowed from numpyro at
    https://github.com/pyro-ppl/numpyro/blob/master/numpyro/distributions/util.py
    """
    tril_inv = np.swapaxes(np.linalg.cholesky(matrix[..., ::-1, ::-1])[..., ::-1, ::-1], -2, -1)
    identity = np.broadcast_to(np.identity(matrix.shape[-1]), tril_inv.shape)
    return solve_triangular(tril_inv, identity, lower=True)


def r_squared(y, yhat):
    """R² metric"""
    return 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)


def rmse(y, yhat):
    """Root Mean Squared Erro"""
    return np.linalg.norm(y - yhat) / np.sqrt(len(y))


def mae(y, yhat):
    """Mean Absolute Error"""
    return np.mean(np.abs(y - yhat))


def mad(y, yhat):
    """Maximum Absolute difference"""
    return np.max(np.abs(y - yhat))


def ned(y, yhat):
    """Normalized Euclidean Distance"""
    return np.linalg.norm(y - yhat) / (np.linalg.norm(y) + np.linalg.norm(yhat))


def smape(y, yhat):
    """Symmetric Mean Absolute Error between 0% and 100%"""
    return 100.0 * np.mean(np.abs(y - yhat) / np.add(np.abs(y), np.abs(yhat)))


def fit(y, yhat):
    """Fit percentage"""
    return 1.0 - np.linalg.norm(y - yhat, 2) / np.linalg.norm(y - np.mean(y), 2)


def nearest_cholesky(A) -> np.ndarray:
    """nearest positive semi definite Cholesky decomposition for symmetric matrices

    Returns:
        Upper triangular Cholesky factor of `A`
    """
    X = (A + A.T) / 2.0 + np.eye(A.shape[0]) * 1e-8
    w, v = np.linalg.eigh(X)
    return np.linalg.qr(np.diag([w ** 0.5 if w > 0 else 0 for w in w]) @ v.T, 'r')


def time_series_pca(X: np.ndarray, verbose: bool = False):
    """Principal Component Analysis reduction of multiple time series into one component

    Args:
        X: Array of time series stacked by columns, e.g. X.shape = (n, nc) where n and nc
            are respectively the length and the number of time series
        verbose: Print the percentage of variance explained by the first component

    Returns:
        The first component of the PCA
    """
    w, v = np.linalg.eigh(np.cov((X - X.mean(axis=0)) / X.std(axis=0), rowvar=False))
    idx = w.argsort()[::-1]
    w, v = w[idx], v[:, idx]
    if verbose:
        print(f"explain {np.squeeze(w[0] / w.sum()):.3f} % of the variance")
    return X @ (v[:, 0] / v[:, 0].sum())
