"""Bayesian Filter template"""


from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence
import numpy as np
import pandas as pd
from ..statespace.base import StateSpace, States


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
        dtype = self.ss._coerce_dtypes()
        ss.update_continuous_ssm()
        # use lru to avoid re_computation of discretization for identical dt
        dts, idx = np.unique(dt, return_inverse=True)
        A = np.zeros((ss.nx, ss.nx, dt.size), dtype=dtype)
        B0 = np.zeros((ss.nx, ss.nu, dt.size), dtype=dtype)
        B1 = np.zeros((ss.nx, ss.nu, dt.size), dtype=dtype)
        Q = np.zeros((ss.nx, ss.nx, dt.size), dtype=dtype)
        Ai, B0i, B1i, Qi = map(np.dstack, zip(*map(ss.discretization, dts)))
        A[:] = Ai[:, :, idx]
        B0[:] = B0i[:, :, idx]
        B1[:] = B1i[:, :, idx]
        Q[:] = Qi[:, :, idx]

        vars = [var.to_numpy(dtype) for var in vars]
        states = States(ss.C, ss.D, ss.R, A, B0, B1, Q)
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
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
          - **x**: Updated state vector.
          - **P**: Updated covariance matrix.
          - **K**: Kalman gain.
          - **S**: Innovation covariance.
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
            Time step size.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - **x**: Predicted state vector.
            - **P**: Predicted covariance matrix.
        """
        pass

    @abstractmethod
    def log_likelihood(
        self,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
    ) -> float:
        """Compute the log-likelihood of the data given the model.

        Parameters
        ----------
        dt : pd.Series
            Time step sizes.
        u : pd.DataFrame
            Output (or exogeneous) vector.
        dtu : pd.DataFrame
            Time derivative of the output vector.
        y : pd.DataFrame
            Measurement (or observation) vector.

        Returns
        -------
        float
            Log-likelihood of the data given the model.
        """
        pass

    @abstractmethod
    def filtering(
        self,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
    ):
        """Perform filtering using the given model.

        Parameters
        ----------
        dt : pd.Series
            Time step sizes.
        u : pd.DataFrame
            Output (or exogeneous) vector.
        dtu : pd.DataFrame
            Time derivative of the output vector.
        y : pd.DataFrame
            Measurement (or observation) vector.

        Returns
        -------
        TODO

        """
        pass

    @abstractmethod
    def smoothing(
        self,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
    ):
        """Perform smoothing using the given model.

        Parameters
        ----------
        dt : pd.Series
            Time step sizes.
        u : pd.DataFrame
            Output (or exogeneous) vector.
        dtu : pd.DataFrame
            Time derivative of the output vector.
        y : pd.DataFrame
            Measurement (or observation) vector.

        Returns
        -------
        TODO
        """
        pass

    @abstractmethod
    def simulate(
        self,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
    ):
        """ Perform a simulation of the model using the given inputs.

        Parameters
        ----------
        dt : pd.Series
            Time step sizes.
        u : pd.DataFrame
            Output (or exogeneous) vector.
        dtu : pd.DataFrame
            Time derivative of the output vector.

        Returns
        -------
        TODO

        """
        pass

    @abstractmethod
    def estimate_output(
        self,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
    ):
        """ Considering the inputs, the observation and the model, return the estimated
        mean values and standard deviation of the observation.

        Parameters
        ----------
        dt : pd.Series
            Time step sizes.
        u : pd.DataFrame
            Output (or exogeneous) vector.
        dtu : pd.DataFrame
            Time derivative of the output vector.
        y : pd.DataFrame
            Measurement (or observation) vector.

        Returns
        -------
        TODO
        """

        pass
