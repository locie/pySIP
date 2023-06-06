import math
import warnings
from copy import deepcopy
from functools import lru_cache
from typing import Sequence

import numpy as np
import pandas as pd
from numba.core.errors import NumbaPerformanceWarning

from ..statespace.base import StateSpace, States
from .base import BayesianFilter

def _solve_triu_inplace(A, b):
    for i in range(b.shape[0]):
        b[i] = (b[i] - A[i, i + 1 :] @ b[i + 1 :]) / A[i, i]
    return b


def _update(C, D, R, x, P, u, y, _Arru):
    ny, _ = C.shape
    _Arru[:ny, :ny] = R
    _Arru[ny:, :ny] = P @ C.T
    _Arru[ny:, ny:] = P
    _, r_fact = np.linalg.qr(_Arru)
    S = r_fact[:ny, :ny]
    if ny == 1:
        k = (y - C @ x - D @ u) / S[0, 0]
        x = x + r_fact[:1, 1:].T * k
    else:
        k = _solve_triu_inplace(S, y - C @ x - D @ u)
        x = x + r_fact[:ny, ny:].T @ k
    P = r_fact[ny:, ny:]
    return x, P, k, S


def _predict(A, B0, B1, Q, x, P, u, dtu):
    _, r = np.linalg.qr(np.vstack((P @ A.T, Q)))
    x = A @ x + B0 @ u + B1 @ dtu
    return x, r


def _kalman_step(x, P, u, dtu, y, states, _Arru):
    if ~np.isnan(y).any():
        x, P, _, _ = _update(states.C, states.D, states.R, x, P, u, y, _Arru)
    x, P = _predict(states.A, states.B0, states.B1, states.Q, x, P, u, dtu)
    return x, P


def _unpack_states(states, i):
    return States(
        states.C,
        states.D,
        states.R,
        states.A[:, :, i],
        states.B0[:, :, i],
        states.B1[:, :, i],
        states.Q[:, :, i],
    )


def _log_likelihood(x0, P0, u, dtu, y, states):
    x = x0
    P = P0
    n_timesteps = y.shape[0]
    ny, nx = states.C.shape
    _Arru = np.zeros((nx + ny, nx + ny))
    log_likelihood = 0.5 * n_timesteps * math.log(2.0 * math.pi)
    for i in range(n_timesteps):
        y_i = np.ascontiguousarray(y[i]).reshape(-1, 1)
        u_i = np.ascontiguousarray(u[i]).reshape(-1, 1)
        dtu_i = np.ascontiguousarray(dtu[i]).reshape(-1, 1)
        states_i = _unpack_states(states, i)
        if ~np.isnan(y_i).any():
            x, P, k, S = _update(
                states_i.C, states_i.D, states_i.R, x, P, u_i, y_i, _Arru
            )
            if ny == 1:
                log_likelihood += math.log(abs(S[0, 0])) + 0.5 * k[0, 0] ** 2
            else:
                log_likelihood += np.linalg.slogdet(S)[1] + 0.5 * (k.T @ k)[0, 0]
        x, P = _predict(
            states_i.A, states_i.B0, states_i.B1, states_i.Q, x, P, u_i, dtu_i
        )
    return log_likelihood


def _filtering(x0, P0, u, dtu, y, states):
    x = x0
    P = P0
    n_timesteps = y.shape[0]
    ny, nx = states.C.shape
    _Arru = np.zeros((nx + ny, nx + ny))
    for i in range(n_timesteps):
        y_i = np.ascontiguousarray(y[i]).reshape(-1, 1)
        u_i = np.ascontiguousarray(u[i]).reshape(-1, 1)
        dtu_i = np.ascontiguousarray(dtu[i]).reshape(-1, 1)
        states_i = _unpack_states(states, i)
        x, P = _kalman_step(x, P, u_i, dtu_i, y_i, states_i, _Arru)
        yield x, P


def _smoothing(x0, P0, u, dtu, y, states):
    # optim TODO: use a proper container to save prior / filtered states
    x = x0
    P = P0
    n_timesteps = y.shape[0]
    ny, nx = states.C.shape
    _Arru = np.zeros((nx + ny, nx + ny))
    xp = np.empty((n_timesteps, nx, 1))
    Pp = np.empty((n_timesteps, nx, nx))
    xf = np.empty((n_timesteps, nx, 1))
    Pf = np.empty((n_timesteps, nx, nx))
    for i in range(n_timesteps):
        y_i = np.ascontiguousarray(y[i]).reshape(-1, 1)
        u_i = np.ascontiguousarray(u[i]).reshape(-1, 1)
        dtu_i = np.ascontiguousarray(dtu[i]).reshape(-1, 1)
        states_i = _unpack_states(states, i)
        xp[i], Pp[i] = x, P
        if ~np.isnan(y).any():
            x, P, _, _ = _update(
                states_i.C, states_i.D, states_i.R, x, P, u_i, y_i, _Arru
            )
        xf[i], Pf[i] = x, P
        x, P = _predict(
            states_i.A, states_i.B0, states_i.B1, states_i.Q, x, P, u_i, dtu_i
        )

    for i in range(n_timesteps - 2, -1, -1):
        G = np.linalg.solve(Pp[i + 1], states.A[:, :, i] @ Pf[i]).T
        xf[i, :, :] += G @ (xf[i + 1, :, :] - xp[i + 1, :, :])
        Pf[i, :, :] += G @ (Pf[i + 1, :, :] - Pp[i + 1, :, :]) @ G.T
    return xf, Pf


def _simulation_step(x, u, dtu, states):
    y = states.C @ x - states.D @ u + states.R @ np.random.randn(*x.shape)
    x = (
        states.A @ x
        + states.B0 @ u
        + states.B1 @ dtu
        + states.Q @ np.random.randn(*x.shape)
    )
    return y, x


def _simulate(x0, u, dtu, states):
    x = x0
    n_timesteps = u.shape[0]
    for i in range(n_timesteps):
        # u_i, dtu_i, states_i = unpack_step((u, dtu), states, i)
        u_i = np.ascontiguousarray(u[i]).reshape(-1, 1)
        dtu_i = np.ascontiguousarray(dtu[i]).reshape(-1, 1)
        states_i = _unpack_states(states, i)
        y, x = _simulation_step(x, u_i, dtu_i, states_i)
        yield y, x


def _simulate_output(x0, P0, u, dtu, y, states):
    x = x0
    P = P0
    n_timesteps = y.shape[0]
    ny, nx = states.C.shape
    _Arru = np.zeros((nx + ny, nx + ny))
    for i in range(n_timesteps):
        # y_i, u_i, dtu_i, states_i = unpack_step((y, u, dtu), states, i)
        y_i = np.ascontiguousarray(y[i]).reshape(-1, 1)
        u_i = np.ascontiguousarray(u[i]).reshape(-1, 1)
        dtu_i = np.ascontiguousarray(dtu[i]).reshape(-1, 1)
        states_i = _unpack_states(states, i)
        x, P = _kalman_step(x, P, u_i, dtu_i, y_i, states_i, _Arru)
        y_m = states_i.C @ x
        y_std = np.sqrt(states_i.C @ P.T @ P @ states_i.C.T) + states_i.R
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
        states = States(C, D, R, A, B0, B1, Q)
        return tuple([x0, P0, *vars, states])

    @staticmethod
    def predict(
        ss: StateSpace,
        x: np.ndarray,
        P: np.ndarray,
        u: np.ndarray,
        dtu: np.ndarray,
        dt: float,
    ):
        ss.update_continuous_states()
        x, P = deepcopy(x), deepcopy(P)
        A, B0, B1, Q = ss.discretization(dt)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            return _predict(A, B0, B1, Q, x, P, u, dtu)

    @staticmethod
    def update(
        ss: StateSpace,
        x: np.ndarray,
        P: np.ndarray,
        u: np.ndarray,
        y: np.ndarray,
    ):
        ss.update_continuous_states()
        x, P = deepcopy(x), deepcopy(P)
        C = ss.C
        D = ss.D
        R = ss.R
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            return _update(C, D, R, x, P, u, y)

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
            x0, P0, u, dtu, y, states = KalmanQR._proxy_params(ss, dt, (u, dtu, y))
            return _log_likelihood(x0, P0, u, dtu, y, states)

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
            x0, P0, u, dtu, y, states = KalmanQR._proxy_params(ss, dt, (u, dtu, y))
            _filtering(x0, P0, u, dtu, y, states)

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
            x0, P0, u, dtu, y, states = KalmanQR._proxy_params(ss, dt, (u, dtu, y))
            return _smoothing(x0, P0, u, dtu, y, states)

    @staticmethod
    def simulate(
        ss: StateSpace,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            x0, _, u, dtu, states = KalmanQR._proxy_params(ss, dt, (u, dtu))
            yield from _simulate(x0, u, dtu, states)

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
            x0, P0, u, dtu, y, states = KalmanQR._proxy_params(ss, dt, (u, dtu, y))
            yield from _simulate_output(x0, P0, u, dtu, y, states)
