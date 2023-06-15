import math
from typing import NamedTuple
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from numba.core.errors import NumbaPerformanceWarning

from ..statespace.base import States
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


class StepResult(NamedTuple):
    x: np.ndarray
    P: np.ndarray
    k: np.ndarray
    S: np.ndarray


def _kalman_step(x, P, u, dtu, y, states, _Arru):
    if ~np.isnan(y).any():
        x, P, k, S = _update(states.C, states.D, states.R, x, P, u, y, _Arru)
    else:
        k = np.full((states.C.shape[0], 1), np.nan)
        S = np.full((states.C.shape[0], states.C.shape[0]), np.nan)
    x, P = _predict(states.A, states.B0, states.B1, states.Q, x, P, u, dtu)
    return x, P, k, S


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
    dtype = states.A.dtype
    _Arru = np.zeros((nx + ny, nx + ny), dtype=dtype)
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
                Si = S[0, 0]
                log_likelihood += (
                    math.log(abs(Si.real) + abs(Si.imag)) + 0.5 * k[0, 0] ** 2
                )
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
    dtype = states.A.dtype
    _Arru = np.zeros((nx + ny, nx + ny), dtype=dtype)
    x_res = np.empty((n_timesteps, nx, 1), dtype=dtype)
    P_res = np.empty((n_timesteps, nx, nx), dtype=dtype)
    k_res = np.empty((n_timesteps, ny, 1), dtype=dtype)
    S_res = np.empty((n_timesteps, ny, ny), dtype=dtype)
    for i in range(n_timesteps):
        y_i = np.ascontiguousarray(y[i]).reshape(-1, 1)
        u_i = np.ascontiguousarray(u[i]).reshape(-1, 1)
        dtu_i = np.ascontiguousarray(dtu[i]).reshape(-1, 1)
        states_i = _unpack_states(states, i)
        x, P, k, S = _kalman_step(x, P, u_i, dtu_i, y_i, states_i, _Arru)
        x_res[i] = x
        P_res[i] = P
        k_res[i] = k
        S_res[i] = S
    return x_res, P_res, k_res, S_res


def _smoothing(x0, P0, u, dtu, y, states):
    # optim TODO: use a proper container to save prior / filtered states
    x = x0
    P = P0
    n_timesteps = y.shape[0]
    ny, nx = states.C.shape
    dtype = states.A.dtype
    _Arru = np.zeros((nx + ny, nx + ny), dtype=dtype)
    xp = np.empty((n_timesteps, nx, 1), dtype=dtype)
    Pp = np.empty((n_timesteps, nx, nx), dtype=dtype)
    xf = np.empty((n_timesteps, nx, 1), dtype=dtype)
    Pf = np.empty((n_timesteps, nx, nx), dtype=dtype)
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


def _simulate(x0, u, dtu, states):
    x = x0
    n_timesteps = u.shape[0]
    ny, nx = states.C.shape
    dtype = states.A.dtype
    y_res = np.empty((n_timesteps, ny, 1), dtype=dtype)
    x_res = np.empty((n_timesteps, nx, 1), dtype=dtype)
    for i in range(n_timesteps):
        # u_i, dtu_i, states_i = unpack_step((u, dtu), states, i)
        u_i = np.ascontiguousarray(u[i]).reshape(-1, 1)
        dtu_i = np.ascontiguousarray(dtu[i]).reshape(-1, 1)
        states_i = _unpack_states(states, i)
        y_res[i] = states_i.C @ x - states_i.D @ u_i
        x_res[i] = (
            states_i.A @ x
            + states_i.B0 @ u_i
            + states_i.B1 @ dtu_i
            + states_i.Q @ np.random.randn(nx, 1)
        )
    y_res += states.R @ np.random.randn(n_timesteps, ny, 1)
    return y_res, x_res


def _estimate_output(x0, P0, u, dtu, y, states):
    x = x0
    P = P0
    n_timesteps = y.shape[0]
    ny, nx = states.C.shape
    dtype = states.A.dtype
    _Arru = np.zeros((nx + ny, nx + ny), dtype=dtype)
    y_res = np.empty((n_timesteps, ny, 1), dtype=dtype)
    x_res = np.empty((n_timesteps, nx, 1), dtype=dtype)
    P_res = np.empty((n_timesteps, nx, nx), dtype=dtype)
    y_std_res = np.empty((n_timesteps, ny, 1), dtype=dtype)

    for i in range(n_timesteps):
        # y_i, u_i, dtu_i, states_i = unpack_step((y, u, dtu), states, i)
        y_i = np.ascontiguousarray(y[i]).reshape(-1, 1)
        u_i = np.ascontiguousarray(u[i]).reshape(-1, 1)
        dtu_i = np.ascontiguousarray(dtu[i]).reshape(-1, 1)
        states_i = _unpack_states(states, i)
        x, P, _, _ = _kalman_step(x, P, u_i, dtu_i, y_i, states_i, _Arru)
        y_m = states_i.C @ x
        y_std = np.sqrt(states_i.C @ P.T @ P @ states_i.C.T) + states_i.R
        y_res[i] = y_m
        x_res[i] = x
        P_res[i] = P
        y_std_res[i] = y_std
    return y_res, x_res, P_res, y_std_res


# All above will be jitter by numba, if available. Otherwise, the pure python / numpy
# implementation will be used.
try:
    from numba import jit_module

    jit_module(nopython=True, nogil=True, cache=True)
except ImportError:
    warnings.warn("Numba not installed, using pure python implementation")


class KalmanQR(BayesianFilter):
    def predict(
        self,
        x: np.ndarray,
        P: np.ndarray,
        u: np.ndarray,
        dtu: np.ndarray,
        dt: float,
    ):
        self.ss.update_continuous_states()
        x, P = deepcopy(x), deepcopy(P)
        A, B0, B1, Q = self.ss.discretization(dt)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            return _predict(A, B0, B1, Q, x, P, u, dtu)

    def update(
        self,
        x: np.ndarray,
        P: np.ndarray,
        u: np.ndarray,
        y: np.ndarray,
    ):
        self.ss.update_continuous_states()
        x, P = deepcopy(x), deepcopy(P)
        C = self.ss.C
        D = self.ss.D
        R = self.ss.R
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            return _update(C, D, R, x, P, u, y)

    def log_likelihood(
        self,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            x0, P0, u, dtu, y, states = self._proxy_params(dt, (u, dtu, y))
            return _log_likelihood(x0, P0, u, dtu, y, states)

    def filtering(
        self,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            x0, P0, u, dtu, y, states = self._proxy_params(dt, (u, dtu, y))
            return _filtering(x0, P0, u, dtu, y, states)

    def smoothing(
        self,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            x0, P0, u, dtu, y, states = self._proxy_params(dt, (u, dtu, y))
            return _smoothing(x0, P0, u, dtu, y, states)

    def simulate(
        self,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            x0, _, u, dtu, states = self._proxy_params(dt, (u, dtu))
            return _simulate(x0, u, dtu, states)

    def estimate_output(
        self,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            x0, P0, u, dtu, y, states = self._proxy_params(dt, (u, dtu, y))
            return _estimate_output(x0, P0, u, dtu, y, states)
