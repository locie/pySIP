from copy import deepcopy
from functools import lru_cache
import math
from typing import NamedTuple, Sequence
import warnings
from numba.core.errors import NumbaPerformanceWarning
import numpy as np
import pandas as pd

from ..statespace.base import StateSpace

from .base import BayesianFilter


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


def unpack_ssm(ssm, i):
    return SSM(
        ssm.C,
        ssm.D,
        ssm.R,
        ssm.Q[:, :, i],
        ssm.A[:, :, i],
        ssm.B0[:, :, i],
        ssm.B1[:, :, i],
    )


def log_likelihood(x0, P0, u, dtu, y, ssm):
    x = x0
    P = P0
    n_timesteps = y.shape[0]
    ny, nx = ssm.C.shape
    _Arru = np.zeros((nx + ny, nx + ny))
    log_likelihood = 0.5 * n_timesteps * math.log(2.0 * math.pi)
    for i in range(n_timesteps):
        y_i = np.ascontiguousarray(y[i]).reshape(-1, 1)
        u_i = np.ascontiguousarray(u[i]).reshape(-1, 1)
        dtu_i = np.ascontiguousarray(dtu[i]).reshape(-1, 1)
        ssm_i = unpack_ssm(ssm, i)
        if ~np.isnan(y_i).any():
            x, P, e, S = update(ssm_i.C, ssm_i.D, ssm_i.R, x, P, u_i, y_i, _Arru)
            if ny == 1:
                log_likelihood += math.log(abs(S[0, 0])) + 0.5 * e[0, 0] ** 2
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
    for i in range(n_timesteps):
        y_i = np.ascontiguousarray(y[i]).reshape(-1, 1)
        u_i = np.ascontiguousarray(u[i]).reshape(-1, 1)
        dtu_i = np.ascontiguousarray(dtu[i]).reshape(-1, 1)
        ssm_i = unpack_ssm(ssm, i)
        x, P = kalman_step(x, P, u_i, dtu_i, y_i, ssm_i, _Arru)
        yield x, P


def smoothing(x0, P0, u, dtu, y, ssm):
    # optim TODO: use a proper container to save prior / filtered states
    x = x0
    P = P0
    n_timesteps = y.shape[0]
    ny, nx = ssm.C.shape
    _Arru = np.zeros((nx + ny, nx + ny))
    xp = np.empty((n_timesteps, nx, 1))
    Pp = np.empty((n_timesteps, nx, nx))
    xf = np.empty((n_timesteps, nx, 1))
    Pf = np.empty((n_timesteps, nx, nx))
    for i in range(n_timesteps):
        y_i = np.ascontiguousarray(y[i]).reshape(-1, 1)
        u_i = np.ascontiguousarray(u[i]).reshape(-1, 1)
        dtu_i = np.ascontiguousarray(dtu[i]).reshape(-1, 1)
        ssm_i = unpack_ssm(ssm, i)
        xp[i], Pp[i] = x, P
        if ~np.isnan(y).any():
            x, P, _, _ = update(ssm_i.C, ssm_i.D, ssm_i.R, x, P, u_i, y_i, _Arru)
        xf[i], Pf[i] = x, P
        x, P = predict(ssm_i.A, ssm_i.B0, ssm_i.B1, ssm_i.Q, x, P, u_i, dtu_i)

    for i in range(n_timesteps - 2, -1, -1):
        G = np.linalg.solve(Pp[i + 1], ssm.A[:, :, i] @ Pf[i]).T
        xf[i, :, :] += G @ (xf[i + 1, :, :] - xp[i + 1, :, :])
        Pf[i, :, :] += G @ (Pf[i + 1, :, :] - Pp[i + 1, :, :]) @ G.T
    return xf, Pf


def simulation_step(x, u, dtu, ssm):
    y = ssm.C @ x - ssm.D @ u + ssm.R @ np.random.randn(*x.shape)
    x = ssm.A @ x + ssm.B0 @ u + ssm.B1 @ dtu + ssm.Q @ np.random.randn(*x.shape)
    return y, x


def simulate(x0, u, dtu, ssm):
    x = x0
    n_timesteps = u.shape[0]
    for i in range(n_timesteps):
        # u_i, dtu_i, ssm_i = unpack_step((u, dtu), ssm, i)
        u_i = np.ascontiguousarray(u[i]).reshape(-1, 1)
        dtu_i = np.ascontiguousarray(dtu[i]).reshape(-1, 1)
        ssm_i = unpack_ssm(ssm, i)
        y, x = simulation_step(x, u_i, dtu_i, ssm_i)
        yield y, x


def simulate_output(x0, P0, u, dtu, y, ssm):
    x = x0
    P = P0
    n_timesteps = y.shape[0]
    ny, nx = ssm.C.shape
    _Arru = np.zeros((nx + ny, nx + ny))
    for i in range(n_timesteps):
        # y_i, u_i, dtu_i, ssm_i = unpack_step((y, u, dtu), ssm, i)
        y_i = np.ascontiguousarray(y[i]).reshape(-1, 1)
        u_i = np.ascontiguousarray(u[i]).reshape(-1, 1)
        dtu_i = np.ascontiguousarray(dtu[i]).reshape(-1, 1)
        ssm_i = unpack_ssm(ssm, i)
        x, P = kalman_step(x, P, u_i, dtu_i, y_i, ssm_i, _Arru)
        y_m = ssm_i.C @ x
        y_std = np.sqrt(ssm_i.C @ P.T @ P @ ssm_i.C.T) + ssm_i.R
        yield x, P, y_m, y_std


# All above will be jitter by numba, if available. Otherwise, the pure python / numpy
# implementation will be used.
try:
    from numba import jit_module

    jit_module(nopython=True, nogil=True, cache=True)
except ImportError:
    warnings.warn("Numba not installed, using pure python implementation")


class KalmanQR(BayesianFilter):
    @staticmethod
    def _proxy_params(
        ss: StateSpace,
        dt: pd.Series,
        vars: Sequence[pd.DataFrame],
    ):
        ss.update_continuous_ssm()
        # use lru to avoid re_computation of discretization for identical dt
        caches_discretization_routine = lru_cache(ss.discretization)
        x0 = deepcopy(ss.x0)
        P0 = deepcopy(ss.P0)

        C = ss.C
        D = ss.D
        R = ss.R
        # apply discretization for all timesteps. LRU avoids re-computation
        A, B0, B1, Q = map(
            np.dstack, zip(*dt.apply(caches_discretization_routine).to_list())
        )
        vars = [var.to_numpy() for var in vars]
        ssm = SSM(C, D, R, Q, A, B0, B1)
        return tuple([x0, P0, *vars, ssm])

    @staticmethod
    def log_likelihood(
        ss: StateSpace,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
    ):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            x0, P0, u, dtu, y, ssm = KalmanQR._proxy_params(ss, dt, (u, dtu, y))
            return log_likelihood(x0, P0, u, dtu, y, ssm)

    @staticmethod
    def filtering(
        ss: StateSpace,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            x0, P0, u, dtu, y, ssm = KalmanQR._proxy_params(ss, dt, (u, dtu, y))
            yield from filtering(x0, P0, u, dtu, y, ssm)

    @staticmethod
    def smoothing(
        ss: StateSpace,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            x0, P0, u, dtu, y, ssm = KalmanQR._proxy_params(ss, dt, (u, dtu, y))
            return smoothing(x0, P0, u, dtu, y, ssm)

    @staticmethod
    def simulate(
        ss: StateSpace,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            x0, _, u, dtu, ssm = KalmanQR._proxy_params(ss, dt, (u, dtu))
            yield from simulate(x0, u, dtu, ssm)

    @staticmethod
    def simulate_output(
        ss: StateSpace,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            x0, P0, u, dtu, y, ssm = KalmanQR._proxy_params(ss, dt, (u, dtu, y))
            yield from simulate_output(x0, P0, u, dtu, y, ssm)
