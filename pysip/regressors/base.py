"""Regressor template"""
from numbers import Real
from typing import Tuple, Union

import numpy as np
import pandas as pd

from ..state_estimator import BayesianFilter
from ..statespace.base import StateSpace


def _check_data(dt, u, dtu, y):
    for df in [dt, u, dtu]:
        if not np.all(np.isfinite(df)):
            raise ValueError(f"{df} contains undefinite values")
    if not np.all(np.isnan(y[~np.isfinite(y)])):
        raise TypeError("The output vector must contains numerical values or numpy.nan")


def _prepare_data(
    df: pd.DataFrame,
    inputs: Union[str, list],
    outputs: Union[str, list],
    time_scale: str = "s",
):
    time = df.index.to_series()
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a dataframe")
    time_scale = pd.to_timedelta(1, time_scale)

    # diff and forward-fill the nan-last value
    dt = time.diff().shift(-1)
    dt.iloc[-1] = dt.iloc[-2]
    if isinstance(df.index, pd.DatetimeIndex):
        dt = dt / time_scale
    else:
        dt = dt.astype(float)

    u = pd.DataFrame(df[inputs])

    # diff and ffill the nan-last value wit 0
    dtu = u.diff().shift(-1) / dt.to_numpy()[:, None]
    dtu.iloc[-1, :] = 0

    y = pd.DataFrame(df[outputs])

    _check_data(dt, u, dtu, y)
    return dt, u, dtu, y


def _init_parameters(
    ss, n_init: int = 1, method: str = "unconstrained", hpd: float = 0.95
) -> np.ndarray:
    """Random initialization of the parameters

    Args:
        n_init: Number of random initialization
        method:
            - **unconstrained**: Uniform draw between [-1, 1] in the uncsontrained
            space
            - **prior**: Uniform draw from the prior distribution
            - **zero**: Set the unconstrained parameters to 0
            - **fixed**: The current parameter values are used
            - **value**: Uniform draw between the parameter value +/- 25%
        hpd: Highest Prior Density to draw sample from (True for unimodal
            distribution)

    Returns:
        eta0: Array of unconstrained parameters of shape (n_par, n_init), where
            n_par is the
        number of free parameters and n_init the number of random initialization
    """

    if not isinstance(n_init, int) or n_init <= 0:
        raise TypeError("`n_init` must an integer greater or equal to 1")

    available_methods = ["unconstrained", "prior", "zero", "fixed", "value"]
    if method not in available_methods:
        raise ValueError(f"`method` must be one of the following {available_methods}")

    if not isinstance(hpd, Real) or not 0.0 < hpd <= 1.0:
        raise ValueError("`hpd` must be between ]0, 1]")

    n_par = len(ss.parameters.eta_free)
    if method == "unconstrained":
        eta0 = np.random.uniform(-1, 1, (n_par, n_init))
    elif method == "zero":
        eta0 = np.zeros((n_par, n_init))
    else:
        eta0 = np.zeros((n_par, n_init))
        for n in range(n_init):
            if method == "prior":
                ss.parameters.prior_init(hpd=hpd)
            elif method == "value":
                value = np.asarray(ss.parameters.theta_sd)
                lb = value - 0.25 * value
                ub = value + 0.25 * value
                ss.parameters.theta_sd = np.random.uniform(lb, ub)

            eta0[:, n] = ss.parameters.eta_free

    return np.squeeze(eta0)


class BaseRegressor:
    """Regressor with common methods for frequentist and Baysian regressor

    Args:
        ss: StateSpace()
        bayesian_filter: BayesianFilter()
        time_scale: Time series frequency, e.g. 's': seconds, 'D': days, etc.
            Works only for pandas.DataFrame with DateTime index
        use_jacobian: Use jacobian adjustement
        use_penalty: Use penalty function
    """

    def __init__(
        self,
        ss: StateSpace,
        bayesian_filter: BayesianFilter,
        time_scale: str,
        use_jacobian: bool,
        use_penalty: bool,
    ):

        self.ss = ss
        self.filter: BayesianFilter = bayesian_filter()
        self._time_scale = time_scale
        self._use_jacobian = use_jacobian
        self._use_penalty = use_penalty

    def _simulate_output(
        self, dt: np.ndarray, u: np.ndarray, u1: np.ndarray, x0: np.ndarray = None
    ) -> np.ndarray:
        """Stochastic simulation of the state-space model

        Args:
            dt: Sampling time
            u: Input data
            u1: Forward finite difference of the input data
            x0: Initial state mean different from `ss.x0`

        Returns:
            Simulated output
        """

        ssm, index = self.ss.get_discrete_ssm(dt)
        return self.filter.simulate(ssm, index, u, u1, x0)

    def _estimate_output(
        self,
        dt: np.ndarray,
        u: np.ndarray,
        u1: np.ndarray,
        y: np.ndarray,
        x0: np.ndarray = None,
        P0: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate the output filtered distribution

        Args:
            dt: Sampling time
            u: Input data
            u1: Forward finite difference of the input data
            y: Output data
            x0: Initial state mean different from `ss.x0`
            P0: Initial state deviation different from `ss.P0`

        Returns:
            2-element tuple containing
                - Filtered output mean
                - Filtered output standard deviation
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

        Args:
            dt: Sampling time
            u: Input data
            u1: Forward finite difference of the input data
            y: Output data
            x0: Initial state mean different from `ss.x0`
            P0: Initial state deviation different from `ss.P0`
            smooth: Use RTS smoother

        Returns:
            2-element tuple containing
                - Filtered or smoothed state mean
                - Filtered or smoothed state covariance
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

        Args:
            dt: Sampling time
            u: Input data
            u1: Forward finite difference of the input data
            y: Output data
            pointwise: Return the negative log-likelihood for each time instants

        Returns:
            The negative log-likelihood or the predictive density evaluated for each
            observation
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

        Args:
            eta: Unconstrained parameters
            dt: Sampling time
            u: Input data
            u1: Forward finite difference of the input data
            y: Output data

        Returns:
            The negative log-posterior
        """

        self.ss.parameters.eta = eta
        log_likelihood = self._eval_log_likelihood(dt, u, u1, y)
        log_posterior = log_likelihood - self.ss.parameters.prior

        return log_posterior

    def _prepare_data(
        self,
        df: pd.DataFrame,
        inputs: Union[str, list],
        outputs: Union[str, list],
    ) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare the data

        This method converts data from the public methods (DataFrame) to the private
        methods (array)

        Args:
            df: Input/output data with a time index
            inputs: Input name(s)
            outputs: Output name(s)
            tnew: Interpolated/extrapolated time index

        Returns:
            5-elements tuple containing
                - **dt**: Sampling time
                - **u**: Input data
                - **u1**: Forward finite difference of the input data
                - **y**: Output data
                - **index_back**: Index corresponding to the new time array `tnew`

        Notes:
            The use of `tnew` is discouraged at the moment; DateTimeIndex are not yet
            supported.
        """
        return _prepare_data(df, inputs, outputs)

    def _init_parameters(
        self, n_init: int = 1, method: str = "unconstrained", hpd: float = 0.95
    ) -> np.ndarray:
        """Random initialization of the parameters

        Args:
            n_init: Number of random initialization
            method:
              - **unconstrained**: Uniform draw between [-1, 1] in the uncsontrained
                space
              - **prior**: Uniform draw from the prior distribution
              - **zero**: Set the unconstrained parameters to 0
              - **fixed**: The current parameter values are used
              - **value**: Uniform draw between the parameter value +/- 25%
            hpd: Highest Prior Density to draw sample from (True for unimodal
              distribution)

        Returns:
            eta0: Array of unconstrained parameters of shape (n_par, n_init), where
              n_par is the
            number of free parameters and n_init the number of random initialization
        """
        return _init_parameters(
            self.ss, n_init=n_init, method=method, hpd=hpd
        )

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

        Raise:
            Must be overriden
        """
        raise NotImplementedError
