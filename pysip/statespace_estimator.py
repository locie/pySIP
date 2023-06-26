import math
from typing import NamedTuple
import warnings
from copy import deepcopy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd
from .statespace.base import StateSpace

from numba.core.errors import NumbaPerformanceWarning


class States(NamedTuple):
    A: np.ndarray
    B0: np.ndarray
    B1: np.ndarray
    C: np.ndarray
    D: np.ndarray
    Q: np.ndarray
    R: np.ndarray


def _solve_triu_inplace(A, b):
    for i in range(b.shape[0]):
        b[i] = (b[i] - A[i, i + 1 :] @ b[i + 1 :]) / A[i, i]
    return b


def _update(
    C, D, R, x, P, u, y, _Arru
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def _predict(A, B0, B1, Q, x, P, u, dtu) -> tuple[np.ndarray, np.ndarray]:
    _, r = np.linalg.qr(np.vstack((P @ A.T, Q)))
    x = A @ x + B0 @ u + B1 @ dtu
    return x, r


def _kalman_step(
    x, P, u, dtu, y, states, _Arru
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if ~np.isnan(y).any():
        x_up, P_up, k, S = _update(states.C, states.D, states.R, x, P, u, y, _Arru)
    else:
        x_up, P_up = x, P
        k = np.full((states.C.shape[0], 1), np.nan)
        S = np.full((states.C.shape[0], states.C.shape[0]), np.nan)
    x_pred, P_pred = _predict(
        states.A, states.B0, states.B1, states.Q, x_up, P_up, u, dtu
    )
    return x_up, P_up, k, S, x_pred, P_pred


def _unpack_states(states, i) -> States:
    return States(
        states.A[:, :, i],
        states.B0[:, :, i],
        states.B1[:, :, i],
        states.C,
        states.D,
        states.Q[:, :, i],
        states.R,
    )


def _log_likelihood(x0, P0, u, dtu, y, states) -> float:
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


def _filtering(
    x0, P0, u, dtu, y, states
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        x_up, P_up, k, S, x, P = _kalman_step(x, P, u_i, dtu_i, y_i, states_i, _Arru)
        x_res[i] = x_up
        P_res[i] = P_up.T @ P_up
        k_res[i] = k
        S_res[i] = S
    return x_res, P_res, k_res, S_res


def _smoothing(x0, P0, u, dtu, y, states) -> tuple[np.ndarray, np.ndarray]:
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
        xp[i] = x
        Pp[i] = P.T @ P
        x_up, P_up, _, _, x, P = _kalman_step(x, P, u_i, dtu_i, y_i, states_i, _Arru)
        xf[i] = x_up
        Pf[i] = P_up.T @ P_up

    for i in range(n_timesteps - 2, -1, -1):
        G = np.linalg.solve(Pp[i + 1], states.A[:, :, i] @ Pf[i]).T
        xf[i, :, :] += G @ (xf[i + 1, :, :] - xp[i + 1, :, :])
        Pf[i, :, :] += G @ (Pf[i + 1, :, :] - Pp[i + 1, :, :]) @ G.T
    return xf, Pf


def _simulate(x0, u, dtu, states) -> tuple[np.ndarray, np.ndarray]:
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


def _estimate_output(
    x0, P0, u, dtu, y, states
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        y_i = np.ascontiguousarray(y[i]).reshape(-1, 1)
        u_i = np.ascontiguousarray(u[i]).reshape(-1, 1)
        dtu_i = np.ascontiguousarray(dtu[i]).reshape(-1, 1)
        states_i = _unpack_states(states, i)
        _, _, _, _, x, P = _kalman_step(x, P, u_i, dtu_i, y_i, states_i, _Arru)
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


@dataclass
class BayesianFilter(ABC):
    """Bayesian Filter abstract class

    This class defines the interface for all Bayesian filters. It is not meant to be
    used directly, but should be inherited by all Bayesian filters.

    All the methods defined here are abstract and must be implemented by the
    inheriting class.
    """

    ss: StateSpace

    def _proxy_params(
        self,
        dt: pd.Series,
        vars: Sequence[pd.DataFrame],
    ):
        ss = self.ss
        ss.update()
        # use lru to avoid re_computation of discretization for identical dt
        dts, idx = np.unique(dt, return_inverse=True)
        A = np.zeros((ss.nx, ss.nx, dt.size))
        B0 = np.zeros((ss.nx, ss.nu, dt.size))
        B1 = np.zeros((ss.nx, ss.nu, dt.size))
        Q = np.zeros((ss.nx, ss.nx, dt.size))
        Ai, B0i, B1i, Qi = map(np.dstack, zip(*map(ss.discretization, dts)))
        A[:] = Ai[:, :, idx]
        B0[:] = B0i[:, :, idx]
        B1[:] = B1i[:, :, idx]
        Q[:] = Qi[:, :, idx]

        vars = [var.to_numpy() for var in vars]
        states = States(A, B0, B1, ss.C, ss.D, Q, ss.R)
        return tuple([ss.x0, ss.P0, *vars, states])

    @abstractmethod
    def update(
        self,
        x: np.ndarray,
        P: np.ndarray,
        u: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Update the state and covariance of the current time step.

        Parameters
        ----------
        x : np.ndarray
            State (or endogeneous) vector.
        P : np.ndarray
            Covariance matrix.
        u : np.ndarray
            Output (or exogeneous) vector.
        y : np.ndarray
            Measurement (or observation) vector.

        Returns
        -------
        np.ndarray
            Updated state vector.
        np.ndarray
            Updated covariance matrix.
        np.ndarray
            Kalman gain.
        np.ndarray
            Innovation covariance.
        """
        pass

    @abstractmethod
    def predict(
        self,
        x: np.ndarray,
        P: np.ndarray,
        u: np.ndarray,
        dtu: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict the state and covariance of the next time step.

        Parameters
        ----------
        x : np.ndarray
            State (or endogeneous) vector.
        P : np.ndarray
            Covariance matrix.
        u : np.ndarray
            Output (or exogeneous) vector.
        dtu : np.ndarray
            Time derivative of the output vector.
        dt : float
            Time step.

        Returns
        -------
        np.ndarray
            Predicted state vector.
        np.ndarray
            Predicted covariance matrix.
        """
        ...

    @abstractmethod
    def log_likelihood(
        self,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
    ) -> float:
        """Compute the log-likelihood of the model.

        Parameters
        ----------
        dt : pd.Series
            Time steps.
        u : pd.DataFrame
            Output (or exogeneous) vector.
        dtu : pd.DataFrame
            Time derivative of the output vector.
        y : pd.DataFrame
            Measurement (or observation) vector.

        Returns
        -------
        float
            Log-likelihood of the model.
        """
        ...

    @abstractmethod
    def filtering(
        self,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
    ):
        ...

    @abstractmethod
    def smoothing(
        self,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
    ):
        ...

    @abstractmethod
    def simulate(
        self,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
    ):
        ...

    @abstractmethod
    def estimate_output(
        self,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
    ):
        ...


class KalmanQR(BayesianFilter):
    def predict(
        self,
        x: np.ndarray,
        P: np.ndarray,
        u: np.ndarray,
        dtu: np.ndarray,
        dt: float,
    ):
        """Predict the state and covariance of the next time step.

        Note that this function will not update the statespace model to ensure the
        consistency with the model parameters : to do so, use the `filter.ss.update`
        method.


        Parameters
        ----------
        x : np.ndarray
            State (or endogeneous) vector.
        P : np.ndarray
            Covariance matrix.
        u : np.ndarray
            Output (or exogeneous) vector.
        dtu : np.ndarray
            First order derivative of the output vector.
        dt : float
            Time step between the current and next time step.

        Returns
        -------
        np.ndarray
            Predicted state vector.
        np.ndarray
            Predicted covariance matrix.
        np.ndarray
            Kalman gain.
        np.ndarray
            Innovation covariance.
        """
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
        """Update the state and covariance of the current time step.

        Note that this function will not update the statespace model to ensure the
        consistency with the model parameters : to do so, use the `filter.ss.update`
        method.

        Parameters
        ----------
        x : np.ndarray
            State (or endogeneous) vector.
        P : np.ndarray
            Covariance matrix.
        u : np.ndarray
            Output (or exogeneous) vector.
        y : np.ndarray
            Measurement (or observation) vector.

        Parameters
        ----------
        x : np.ndarray
            State (or endogeneous) vector.
        P : np.ndarray
            Covariance matrix.
        """

        x, P = deepcopy(x), deepcopy(P)
        nx = self.ss.nx
        ny = self.ss.ny
        C = self.ss.C
        D = self.ss.D
        R = self.ss.R
        _Arru = np.zeros((nx + ny, nx + ny))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            return _update(C, D, R, x, P, u, y, _Arru)

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
