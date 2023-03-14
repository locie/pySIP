from copy import deepcopy
import math
from typing import NamedTuple
import warnings
from numba.core.errors import NumbaPerformanceWarning
import numpy as np
from numba import njit

from .base import BayesianFilter


@njit
def solve_tril_inplace(A, b):
    for i in range(b.shape[0]):
        b[i] = (b[i] - A[i, :i] @ b[:i]) / A[i, i]
    return b


@njit
def solve_triu_inplace(A, b):
    for i in range(b.shape[0]):
        b[i] = (b[i] - A[i, i + 1 :] @ b[i + 1 :]) / A[i, i]
    return b


@njit
def _nb_update(C, D, R, x, P, u, y, _Arru):
    ny, nx = C.shape
    _Arru = np.zeros((nx + ny, nx + ny))
    _Arru[:ny, :ny] = R
    _Arru[ny:, :ny] = P @ C.T
    _Arru[ny:, ny:] = P
    _, r_fact = np.linalg.qr(_Arru)
    S = r_fact[:ny, :ny]
    if ny == 1:
        e = ((y - C @ x - D @ u) / S)
        x += r_fact[:1, 1:].T * e
    else:
        e = solve_triu_inplace(S, y - C @ x - D @ u)
        x += r_fact[:ny, ny:].T @ e
    P = r_fact[ny:, ny:]
    return x, P, e, S


@njit
def _nb_predict(A, B0, B1, Q, x, P, u, u1):
    _, r = np.linalg.qr(np.vstack((P @ A.T, Q)))
    x = A @ x + B0 @ u + B1 @ u1
    return x, r


@njit
def _nb_log_likelihood(x, u, dtu, y, C, D, R, P, Q, A, B0, B1):
    n_timesteps = y.shape[1]
    ny, nx = C.shape
    _Arru = np.zeros((nx + ny, nx + ny))
    log_likelihood = 0.5 * n_timesteps * math.log(2.0 * math.pi)
    for i in range(n_timesteps):
        y_i = np.asfortranarray(y[:, i])
        u_i = np.asfortranarray(u[:, i]).reshape(-1, 1)
        dtu_i = np.asfortranarray(dtu[:, i]).reshape(-1, 1)
        x, P, e, S = _nb_update(C, D, R, x, P, u_i, y_i, _Arru)
        x, P = _nb_predict(A, B0, B1, Q, x, P, u_i, dtu_i)
        if ny == 1:
            log_likelihood += math.log(abs(S)) + 0.5 * e**2
        else:
            log_likelihood += np.linalg.slogdet(S)[1] + 0.5 * (e.T @ e)[0, 0]
    return log_likelihood


class FastKalmanQR(BayesianFilter):
    @staticmethod
    def log_likelihood(
        ssm: NamedTuple,
        index: np.ndarray,
        u: np.ndarray,
        u1: np.ndarray,
        y: np.ndarray,
        pointwise=False,
    ):
        x = deepcopy(ssm.x0)
        P = deepcopy(ssm.P0)

        C = ssm.C
        D = ssm.D
        R = ssm.R
        A = ssm.A[0]
        B0 = ssm.B0[0]
        B1 = ssm.B1[0]
        Q = ssm.Q[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            return _nb_log_likelihood(
                x, u, u1, y, C, D, R, P, Q, A, B0, B1
            )