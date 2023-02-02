import warnings
import numpy as np

from typing import Tuple
from scipy.linalg import LinAlgError, LinAlgWarning, ldl, solve_triangular
from numpy.linalg import lstsq


def log1p_exp(a: float):
    """Calculates the log of 1 plus the exponential of the specified value without overflow"""
    if a > 0.0:
        return a + np.log1p(np.exp(-a))
    return np.log1p(np.exp(a))


def log_sum_exp(a: float, b: float):
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


def rmse(y, yhat):
    """Root Mean Squared Error"""
    return np.linalg.norm(y - yhat) / np.sqrt(len(y))


def mae(y, yhat):
    """Mean Absolute Error"""
    return np.mean(np.abs(y - yhat))


def mad(y, yhat):
    """Maximum Absolute Difference"""
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


def nearest_cholesky(m, method='ldl') -> np.ndarray:
    """nearest positive semi definite Cholesky decomposition

    Returns:
        Upper triangular Cholesky factor of `A`
    """
    if not m.any():
        return m

    x = (m + m.T) / 2.0
    if method == 'ldl':
        lu, d, _ = ldl(x, lower=True, hermitian=True)
        return np.diag([np.sqrt(w) if w > 0 else 0 for w in d.diagonal()]) @ lu.T
    elif method == 'eigen':
        eigvals, v = np.linalg.eigh(x)
        return np.linalg.qr(np.diag([np.sqrt(w) if w > 0 else 0 for w in eigvals]) @ v.T, 'r')
    elif method == 'jitter':
        Ix = np.eye(x.shape[0])
        jitter = 1e-9
        while jitter < 1.0:
            try:
                chol = np.linalg.cholesky(x + jitter * Ix).T
                return chol
            except (LinAlgError, LinAlgWarning):
                jitter *= 10.0
    else:
        raise ValueError('`method` must be set to `ldl`, `eigen` or `jitter`')


def diff_upper_cholesky(R: np.ndarray, dS: np.ndarray) -> np.ndarray:
    """Forward differentiation of the upper Cholesky decomposition

    .. math::

        S = R^T R

    Args:
        R: Upper triangular Cholesky factor of the symmetric matrix S
        dS: Derivative of the symmetric matrix S

    Returns:
        dR: Derivative of the upper triangular Cholesky factor R

    Reference:
        Iain Murray. Differentiation of the Cholesky decomposition, February 2016
        https://homepages.inf.ed.ac.uk/imurray2/pub/16choldiff/choldiff.pdf
    """

    def Phi(x):
        x = np.triu(x)
        x[np.diag_indices_from(x)] *= 0.5
        return x

    try:
        right_ = solve_triangular(R, dS.T, trans='T').T
    except (LinAlgError, LinAlgWarning):
        right_ = np.transpose(lstsq(R.T, dS.T, rcond=-1)[0])
    finally:
        try:
            left_ = solve_triangular(R, right_, trans='T')
        except (LinAlgError, LinAlgWarning):
            left_ = lstsq(R.T, right_, rcond=-1)[0]
    return Phi(left_) @ R


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
        print(f"explain {np.squeeze(w[0] / w.sum()):.3e} % of the variance")
    return X @ (v[:, 0] / v[:, 0].sum())
