from copy import deepcopy
from functools import lru_cache
import math
from typing import NamedTuple
import warnings
from numba.core.errors import NumbaPerformanceWarning
import numpy as np
import pandas as pd

from ..statespace.base import StateSpace

from .base import BayesianFilter


# def _solve_tril_inplace(A, b):
#     for i in range(b.shape[0]):
#         b[i] = (b[i] - A[i, :i] @ b[:i]) / A[i, i]
#     return b


class SSM(NamedTuple):
    C: np.ndarray
    D: np.ndarray
    R: np.ndarray
    Q: np.ndarray
    A: np.ndarray
    B0: np.ndarray
    B1: np.ndarray


def _solve_triu_inplace(A, b):
    for i in range(b.shape[0]):
        b[i] = (b[i] - A[i, i + 1 :] @ b[i + 1 :]) / A[i, i]
    return b


def update(C, D, R, x, P, u, y, _Arru):
    ny, _ = C.shape
    _Arru[:ny, :ny] = R
    _Arru[ny:, :ny] = P @ C.T
    _Arru[ny:, ny:] = P
    _, r_fact = np.linalg.qr(_Arru)
    S = r_fact[:ny, :ny]
    if ny == 1:
        e = (y - C @ x - D @ u) / S[0, 0]
        x += r_fact[:1, 1:].T * e
    else:
        e = _solve_triu_inplace(S, y - C @ x - D @ u)
        x += r_fact[:ny, ny:].T @ e
    P = r_fact[ny:, ny:]
    return x, P, e, S


def predict(A, B0, B1, Q, x, P, u, dtu):
    _, r = np.linalg.qr(np.vstack((P @ A.T, Q)))
    x = A @ x + B0 @ u + B1 @ dtu
    return x, r


def kalman_step(x, P, u, dtu, y, ssm, _Arru):
    if ~np.isnan(y).any():
        x, P, _, _ = update(ssm.C, ssm.D, ssm.R, x, P, u, y, _Arru)
    x, P = predict(ssm.A, ssm.B0, ssm.B1, ssm.Q, x, P, u, dtu)
    return x, P


def unpack_step(y, u, dtu, ssm, i):
    return (
        y[i].reshape(-1, 1),
        u[i].reshape(-1, 1),
        dtu[i].reshape(-1, 1),
        SSM(
            ssm.C,
            ssm.D,
            ssm.R,
            ssm.Q[:, :, i],
            ssm.A[:, :, i],
            ssm.B0[:, :, i],
            ssm.B1[:, :, i],
        ),
    )


def log_likelihood(x0, P0, u, dtu, y, ssm):
    x = x0
    P = P0
    n_timesteps = y.shape[0]
    ny, nx = ssm.C.shape
    _Arru = np.zeros((nx + ny, nx + ny))
    log_likelihood = 0.5 * n_timesteps * math.log(2.0 * math.pi)
    for i in range(n_timesteps):
        y_i, u_i, dtu_i, ssm_i = unpack_step(y, u, dtu, ssm, i)
        if ~np.isnan(y_i).any():
            x, P, e, S = update(ssm_i.C, ssm_i.D, ssm_i.R, x, P, u_i, y_i, _Arru)
            if ny == 1:
                log_likelihood += math.log(abs(S[0, 0])) + 0.5 * e[0, 0]**2
            else:
                log_likelihood += np.linalg.slogdet(S)[1] + 0.5 * (e.T @ e)[0, 0]
        x, P = predict(ssm_i.A, ssm_i.B0, ssm_i.B1, ssm_i.Q, x, P, u_i, dtu_i)
    return log_likelihood


def filtering(x0, P0, u, dtu, y, ssm):
    x = x0
    P = P0
    n_timesteps = y.shape[0]
    ny, nx = ssm.C.shape
    _Arru = np.zeros((nx + ny, nx + ny))
    yield x, P, 0, 0
    for i in range(n_timesteps):
        y_i, u_i, dtu_i, ssm_i = unpack_step(y, u, dtu, ssm, i)
        x, P = kalman_step(x, P, u_i, dtu_i, y_i, ssm_i, _Arru)
        yield x, P


# try:
#     from numba import jit_module

#     jit_module(nopython=True, nogil=True, cache=True)
# except ImportError:
#     warnings.warn("Numba not installed, using pure python implementation")


class KalmanQR(BayesianFilter):
    @staticmethod
    def log_likelihood(
        ss: StateSpace,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
    ):
        ss.update_continuous_ssm()
        discretization = lru_cache(ss.discretization)
        x0 = deepcopy(ss.x0)
        P0 = deepcopy(ss.P0)

        C = ss.C
        D = ss.D
        R = ss.R
        A, B0, B1, Q = map(np.dstack, zip(*dt.apply(discretization).to_list()))
        u = u.to_numpy()
        dtu = dtu.to_numpy()
        y = y.to_numpy()
        ssm = SSM(C, D, R, Q, A, B0, B1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            return log_likelihood(x0, P0, u, dtu, y, ssm)
