"""Bayesian Filter template"""


from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
from ..statespace.base import StateSpace


class BayesianFilter(metaclass=ABCMeta):
    """Bayesian Filter abstract class

    This class defines the interface for all Bayesian filters. It is not meant to be
    used directly, but should be inherited by all Bayesian filters.

    All the methods defined here are abstract and must be implemented by the
    inheriting class.
    """

    @abstractmethod
    def update(
        ss: StateSpace,
        x: np.ndarray,
        P: np.ndarray,
        u: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Update the state and covariance of the current time step.

        Parameters
        ----------
        ss : StateSpace
            State space model.
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
        ss: StateSpace,
        x: np.ndarray,
        P: np.ndarray,
        u: np.ndarray,
        dtu: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict the state and covariance of the next time step.

        Parameters
        ----------
        ss : StateSpace
            State space model.
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
        ss: StateSpace,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
    ) -> float:
        """Compute the log-likelihood of the data given the model.

        Parameters
        ----------
        ss : StateSpace
            State space model.
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
        ss: StateSpace,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
    ):
        """Perform filtering using the given model.

        Parameters
        ----------
        ss : StateSpace
            State space model.
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
        ss: StateSpace,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
    ):
        """Perform smoothing using the given model.

        Parameters
        ----------
        ss : StateSpace
            State space model.
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
        ss: StateSpace,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
    ):
        """ Perform a simulation of the model using the given inputs.

        Parameters
        ----------
        ss : StateSpace
            State space model.
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
    def simulate_output(
        ss: StateSpace,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
    ):
        """ Considering the inputs, the observation and the model, return the estimated
        mean values and standard deviation of the observation.

        Parameters
        ----------
        ss : StateSpace
            State space model.
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
