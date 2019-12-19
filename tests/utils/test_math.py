"""tests for linear algebra utilities"""
import pytest
import numpy as np
from scipy.linalg import LinAlgError
from pysip.utils import nearest_cholesky


def test_nearest_cholesky():
    dim = 3
    A = np.abs(np.random.uniform(0.1, 1, (dim, dim)))
    A = (A + A.T) / 2.0

    n = A.shape[0]
    jitter = 1e-14
    while jitter < 1.0:
        try:
            A += jitter * np.eye(n)
            chol = np.linalg.cholesky(A).T
            break
        except (LinAlgError, RuntimeError):
            jitter *= 10.0

    nchol = nearest_cholesky(A)

    assert np.allclose(chol.T @ chol, A)
    assert np.allclose(nchol.T @ nchol, A)
