import numpy as np
import pytest
from scipy.linalg import eig, eigvals, expm, inv

from pysip.statespace.discretization import (
    eig_2x2,
    eigvals_2x2,
    eigvals_3x3,
    expm_2x2,
    inv_2x2,
    inv_3x3,
)


def random_real_eig(n):
    """n-dimensional random matrix with real eigenvalues"""
    D = np.diag(np.random.uniform(1e-4, 1.0, n))
    Q, _ = np.linalg.qr(np.random.random((n, n)))
    return Q @ D @ Q.T


@pytest.fixture
def random_2x2():
    return random_real_eig(2)


@pytest.fixture
def random_3x3():
    return random_real_eig(3)


def test_inv_2x2(random_2x2):
    assert np.allclose(inv(random_2x2), inv_2x2(random_2x2))


def test_inv_3x3(random_3x3):
    assert np.allclose(inv(random_3x3), inv_3x3(random_3x3))


def test_eigvals_2x2(random_2x2):
    assert np.allclose(np.sort(eigvals(random_2x2)), np.sort(eigvals_2x2(random_2x2)))


def test_eigvals_3x3(random_3x3):
    assert np.allclose(np.sort(eigvals(random_3x3)), np.sort(eigvals_3x3(random_3x3)))


def test_eig_2x2(random_2x2):
    X = random_2x2
    w_truth, v_truth = eig(X)
    w_truth_sorted = np.sort(w_truth)
    v_truth_sorted = v_truth[:, w_truth.argsort()]

    w, v = eig_2x2(X)
    w_sorted = np.sort(w)
    v_sorted = v[:, w.argsort()]

    assert np.allclose(X, v_truth @ np.diag(w_truth) @ inv_2x2(v_truth))
    assert np.allclose(X, v @ np.diag(w) @ inv_2x2(v))

    assert np.allclose(w_truth_sorted, w_sorted)
    assert np.allclose(np.abs(v_truth_sorted), np.abs(v_sorted))


def test_expm_2x2(random_2x2):
    assert np.allclose(expm(random_2x2), expm_2x2(random_2x2))
