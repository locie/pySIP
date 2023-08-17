"""tests for linear algebra utilities"""
import numpy as np
import pytest
from scipy.linalg import LinAlgError

from pysip.utils.math import diff_upper_cholesky, nearest_cholesky


@pytest.mark.parametrize("N", [5, 10, 25, 50])
def test_nearest_upper_cholesky(N):

    S = np.cov(np.random.randn(N, 2 * N))
    S = (S + S.T) / 2.0
    Is = np.eye(S.shape[0])
    jitter = 1e-10
    while jitter < 1.0:
        try:
            S += jitter * Is
            upper_chol = np.linalg.cholesky(S).T
            break
        except LinAlgError:
            jitter *= 10.0

    nearest_upper_chol = nearest_cholesky(S)

    assert np.allclose(upper_chol.T @ upper_chol, S)
    assert np.allclose(nearest_upper_chol.T @ nearest_upper_chol, S)


@pytest.mark.parametrize("N", [5, 10, 25, 50])
def test_diff_upper_cholesky(N):
    S = np.cov(np.random.randn(N, 2 * N))
    dS = np.cov(np.random.randn(N, 2 * N))
    S = (S + S.T) / 2.0
    Is = np.eye(S.shape[0])
    jitter = 1e-10
    while jitter < 1.0:
        try:
            S += jitter * Is
            upper_chol = np.linalg.cholesky(S).T
            break
        except LinAlgError:
            jitter *= 10.0

    def fd_upper_cholesky(S, dS):
        hh = 1e-5
        R1 = np.linalg.cholesky(S - dS * hh / 2.0).T
        R2 = np.linalg.cholesky(S + dS * hh / 2.0).T
        return (R2 - R1) / hh

    dR_fd = fd_upper_cholesky(S, dS)

    R = nearest_cholesky(S)
    dR = diff_upper_cholesky(R, dS)

    assert np.allclose(upper_chol.T @ upper_chol, S)
    assert np.allclose(R.T @ R, S)
    assert np.allclose(dR, dR_fd)
