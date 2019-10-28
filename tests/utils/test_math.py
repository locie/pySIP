"""tests for linear algebra utilities"""
import pytest
import numpy as np
from pysip.utils import nearest_cholesky


def test_nearest_cholesky():
    dim = 3
    A = np.abs(np.random.uniform(0.1, 1, (dim, dim)))
    A = (A + A.T + np.eye(dim)) / 2

    nchol = nearest_cholesky(A)
    chol = np.linalg.cholesky(A).T

    assert np.allclose(chol.T @ chol, A)
    assert np.allclose(nchol.T @ nchol, A)
    assert np.allclose(np.abs(nchol), np.abs(chol))
