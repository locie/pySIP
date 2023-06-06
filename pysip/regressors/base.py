"""Regressor template"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, Union

import numpy as np
import pandas as pd

from ..filters.kalman_qr import KalmanQR

from ..filters import BayesianFilter
from ..statespace.base import StateSpace


@dataclass
class BaseRegressor(ABC):
    """Regressor with common methods for frequentist and Baysian regressor

    Parameters
    ----------
    ss : StateSpace()
        State-space model
    bayesian_filter : BayesianFilter()
        Bayesian filter
    time_scale : str
        Time series frequency, e.g. 's': seconds, 'D': days, etc.
    use_jacobian : bool
        Use jacobian adjustement
    use_penalty : bool
        Use penalty function
    """
    ss: StateSpace
    bayesian_filter: BayesianFilter = field(default_factory=KalmanQR)
    time_scale: str = "s"
    use_jacobian: bool = True
    use_penalty: bool = True

    def simulate_output(
        self, dt: np.ndarray, u: np.ndarray, u1: np.ndarray, x0: np.ndarray = None
    ) -> np.ndarray:
        """Stochastic simulation of the state-space model

        Parameters
        ----------
        dt : float
            Sampling time
        u : array_like, shape (n, m)
            Input data
        u1 : array_like, shape (n, m)
            Forward finite difference of the input data
        x0 : array_like, shape (n,)
            Initial state mean different from ``ss.x0``

        Returns
        -------
        y : array_like, shape (n, p)
            Simulated output
        """

        ssm, index = self.ss.get_discrete_ssm(dt)
        return self.filter.simulate(ssm, index, u, u1, x0)

    def estimate_output(
        self,
        dt: np.ndarray,
        u: np.ndarray,
        u1: np.ndarray,
        y: np.ndarray,
        x0: np.ndarray = None,
        P0: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate the output filtered distribution

        Parameters
        ----------
        dt : float
            Sampling time
        u : (N, m) array_like
            Input data
        u1 : (N, m) array_like
            Forward finite difference of the input data
        y : (N, p) array_like
            Output data
        x0 : (n,) array_like
            Initial state mean different from `ss.x0`
        P0 : (n, n) array_like
            Initial state deviation different from `ss.P0`

        Returns
        -------
        ym : (N, p) ndarray
            Filtered output mean
        ystd : (N, p) ndarray
            Filtered output standard deviation
        """

        ssm, index = self.ss.get_discrete_ssm(dt)
        return self.filter.simulate_output(ssm, index, u, u1, y, x0, P0)

    def _estimate_states(
        self,
        dt: np.ndarray,
        u: np.ndarray,
        u1: np.ndarray,
        y: np.ndarray,
        x0: np.ndarray = None,
        P0: np.ndarray = None,
        smooth: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate the state filtered/smoothed distribution

        Parameters
        ----------
        dt : float
            Sampling time
        u : array_like, shape (n_u, n_steps)
            Input data
        u1 : array_like, shape (n_u, n_steps)
            Forward finite difference of the input data
        y : array_like, shape (n_y, n_steps)
            Output data
        x0 : array_like, shape (n_x, )
            Initial state mean different from `ss.x0`
        P0 : array_like, shape (n_x, n_x)
            Initial state deviation different from `ss.P0`
        smooth : bool, optional
            Use RTS smoother

        Returns
        -------
        x : array_like, shape (n_x, n_steps)
            Filtered or smoothed state mean
        P : array_like, shape (n_x, n_x, n_steps)
            Filtered or smoothed state covariance
        """

        ssm, index = self.ss.get_discrete_ssm(dt)
        if smooth:
            return self.filter.smoothing(ssm, index, u, u1, y, x0, P0)
        return self.filter.filtering(ssm, index, u, u1, y, x0, P0)[:2]

    def _eval_log_likelihood(
        self,
        dt: np.ndarray,
        u: np.ndarray,
        u1: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Evaluate the negative log-likelihood

        Parameters
        ----------
        dt : float
            Sampling time
        u : array_like, shape (n_u, n_steps)
            Input data
        u1 : array_like, shape (n_u, n_steps)
            Forward finite difference of the input data
        y : array_like, shape (n_y, n_steps)
            Output data

        Returns
        -------
        log_likelihood : float
            The negative log-likelihood or the predictive density evaluated for each
            observation.
        """

        return self.filter.log_likelihood(self.ss, dt, u, u1, y)

    def _eval_log_posterior(
        self,
        eta: np.ndarray,
        dt: np.ndarray,
        u: np.ndarray,
        u1: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Evaluate the negative log-posterior

        Parameters
        ----------
        eta : array_like, shape (n_eta, )
            Unconstrained parameters
        dt : float
            Sampling time
        u : array_like, shape (n_u, n_steps)
            Input data
        u1 : array_like, shape (n_u, n_steps)
            Forward finite difference of the input data
        y : array_like, shape (n_y, n_steps)
            Output data

        Returns
        -------
        log_posterior : float
            The negative log-posterior
        """

        self.ss.parameters.eta = eta
        log_likelihood = self._eval_log_likelihood(dt, u, u1, y)
        log_posterior = log_likelihood - self.ss.parameters.prior

        return log_posterior

    @abstractmethod
    def fit(
        self,
        df: pd.DataFrame,
        outputs: Union[str, list],
        inputs: Union[str, list] = None,
        options: dict = None,
    ):
        """Fit the state-space model

        Args:
            df: Training data
            outputs: Output name(s)
            inputs: Inputs names(s)
            options: See options for frequentist and Bayesian regressor

        Returns:
            Fit object

        Parameters
        ----------
        df : pandas.DataFrame
            Training data
        outputs : str or list of str
            Output name(s)
        inputs : str or list of str, optional
            Input name(s)
        options : dict, optional
            See options for frequentist and Bayesian regressor
        """
        pass

    @property
    def parameters(self):
        return self.ss.parameters