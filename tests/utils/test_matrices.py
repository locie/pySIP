"""tests for linear algebra utilities"""
import pytest
import numpy as np

from bopt.utils.matrices import pseudo_cholesky, MultivariateNormal


def test_pseudo_cholesky():
    dim = 3
    A = np.random.random((dim, dim))
    A += A.T + np.eye(dim)
    A /= 2

    pchol = pseudo_cholesky(A, 1e-16)
    chol = (np.linalg.cholesky(A).T)

    assert np.allclose(chol.T @ chol, A)
    assert np.allclose(pchol.T @ pchol, A)
    assert np.allclose(np.abs(pchol), np.abs(chol))


def test_multivariate():
    mvn = MultivariateNormal(2)
    x = [0.3, 0.6]
    assert mvn.log_pdf(x) == pytest.approx(-2.0628770664093454)
