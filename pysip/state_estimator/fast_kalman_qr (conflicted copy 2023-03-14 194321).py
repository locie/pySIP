from copy import deepcopy
import math
from typing import NamedTuple
import warnings
from numba.core.errors import NumbaPerformanceWarning
import numpy as np
import pandas as pd

from ..statespace.base import StateSpace

from .base import BayesianFilter


def solve_tril_inplace(A, b):
    for i in range(b.shape[0]):
        b[i] = (b[i] - A[i, :i] @ b[:i]) / A[i, i]
    return b


def solve_triu_inplace(A, b):
    for i in range(b.shape[0]):
        b[i] = (b[i] - A[i, i + 1 :] @ b[i + 1 :]) / A[i, i]
    return b


def _nb_update(C, D, R, x, P, u, y, _Arru):
    ny, nx = C.shape
    _Arru = np.zeros((nx + ny, nx + ny))
    _Arru[:ny, :ny] = R
    _Arru[ny:, :ny] = P @ C.T
    _Arru[ny:, ny:] = P
    _, r_fact = np.linalg.qr(_Arru)
    S = r_fact[:ny, :ny]
    if ny == 1:
        e = (y - C @ x - D @ u) / S
        x += r_fact[:1, 1:].T * e
    else:
        e = solve_triu_inplace(S, y - C @ x - D @ u)
        x += r_fact[:ny, ny:].T @ e
    P = r_fact[ny:, ny:]
    return x, P, e, S


def _nb_predict(A, B0, B1, Q, x, P, u, u1):
    _, r = np.linalg.qr(np.vstack((P @ A.T, Q)))
    x = A @ x + B0 @ u + B1 @ u1
    return x, r


def _nb_kalman_step(x, P, u, u1, y, C, D, R, Q, A, B0, B1, _Arru):
    x, P, e, S = _nb_update(C, D, R, x, P, u, y, _Arru)
    x, P = _nb_predict(A, B0, B1, Q, x, P, u, u1)
    return x, P, e, S


def _nb_log_likelihood(x, u, dtu, y, C, D, R, P, Q, A, B0, B1):
    n_timesteps = y.shape[1]
    ny, nx = C.shape
    _Arru = np.zeros((nx + ny, nx + ny))
    log_likelihood = 0.5 * n_timesteps * math.log(2.0 * math.pi)
    for i in range(n_timesteps):
        Q_i = Q[i]
        A_i = A[i]
        B0_i = B0[i]
        B1_i = B1[i]
        y_i = y[:, i]
        u_i = u[:, i].reshape(-1, 1)
        dtu_i = dtu[:, i].reshape(-1, 1)
        x, P, e, S = _nb_kalman_step(
            x, P, u_i, dtu_i, y_i, C, D, R, Q_i, A_i, B0_i, B1_i, _Arru
        )
        if ny == 1:
            log_likelihood += math.log(abs(S)) + 0.5 * e**2
        else:
            log_likelihood += np.linalg.slogdet(S)[1] + 0.5 * (e.T @ e)[0, 0]
    return log_likelihood


try:
    from numba import jit_module
    jit_module(nopython=True, nogil=True, cache=True)
except ImportError:
    warnings.warn("Numba not installed, using pure python implementation")


class KalmanQR(BayesianFilter):
    @staticmethod
    def log_likelihood(
        ss: StateSpace,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
    ):
        x = deepcopy(ss.x0)
        P = deepcopy(ss.P0)

        C = ss.C
        D = ss.D
        R = ss.R
        Q, A, B0, B1 = zip(*dt.apply(ss.discretization).to_list())
        print(Q)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            return _nb_log_likelihood(x, u, dtu, y, C, D, R, P, Q, A, B0, B1)
