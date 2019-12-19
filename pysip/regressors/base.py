"""Regressor template"""
from typing import Tuple, Union
from numbers import Real
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from ..state_estimator import BayesianFilter
from ..statespace.base import StateSpace


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
        self.filter = bayesian_filter()
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
        self, dt: np.ndarray, u: np.ndarray, u1: np.ndarray, y: np.ndarray, pointwise: bool = False
    ) -> Union[float, np.ndarray]:
        """Evaluate the negative log-likelihood

        Args:
            dt: Sampling time
            u: Input data
            u1: Forward finite difference of the input data
            y: Output data
            pointwise: Return the negative log-likelihood for each time instants

        Returns:
            The negative log-likelihood or the predictive density evaluated for each observation
        """

        ssm, index = self.ss.get_discrete_ssm(dt)
        return self.filter.log_likelihood(ssm, index, u, u1, y, pointwise)

    def _eval_dlog_likelihood(
        self, dt: np.ndarray, u: np.ndarray, u1: np.ndarray, y: np.ndarray
    ) -> Union[float, np.ndarray]:
        """Evaluate the negative log-likelihood and the gradient

        Args:
            dt: Sampling time
            u: Input data
            u1: Forward finite difference of the input data
            y: Output data

        Returns:
            2-element tuple containing
                - The negative log-likelihood
                - The gradient of the negative log-likelihood
        """

        ssm, dssm, index = self.ss.get_discrete_dssm(dt)
        return self.filter.dlog_likelihood(ssm, dssm, index, u, u1, y)

    def _eval_log_posterior(
        self, eta: np.ndarray, dt: np.ndarray, u: np.ndarray, u1: np.ndarray, y: np.ndarray
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
        log_likelihood = self._eval_log_likelihood(dt, u, u1, y, pointwise=False)
        log_posterior = log_likelihood - self.ss.parameters.prior

        if self._use_jacobian:
            log_posterior -= self.ss.parameters.theta_log_jacobian

        if self._use_penalty:
            log_posterior += self.ss.parameters.penalty

        return log_posterior

    def _eval_dlog_posterior(
        self, eta: np.ndarray, dt: np.ndarray, u: np.ndarray, u1: np.ndarray, y: np.ndarray
    ) -> Union[float, np.ndarray]:
        """Evaluate the negative log-posterior and the gradient

        Args:
            eta: Unconstrained parameters
            dt: Sampling time
            u: Input data
            u1: Forward finite difference of the input data
            y: Output data

        Returns:
            2-element tuple containing
                - The negative log-posterior
                - The gradient of the negative log-posterior
        """

        self.ss.parameters.eta = eta
        log_likelihood, dlog_likelihood = self._eval_dlog_likelihood(dt, u, u1, y)
        log_posterior = log_likelihood - self.ss.parameters.prior
        dlog_posterior = dlog_likelihood * self.ss.parameters.scale - self.ss.parameters.d_prior
        dlog_posterior *= self.ss.parameters.theta_jacobian

        if self._use_jacobian:
            log_posterior -= self.ss.parameters.theta_log_jacobian
            dlog_posterior -= self.ss.parameters.theta_dlog_jacobian

        if self._use_penalty:
            log_posterior += self.ss.parameters.penalty
            dlog_posterior += self.ss.parameters.d_penalty

        return log_posterior, dlog_posterior

    def _prepare_data(
        self,
        df: pd.DataFrame,
        inputs: Union[str, list],
        outputs: Union[str, list],
        tnew: [np.ndarray, pd.Series] = None,
    ) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare the data

        This method converts data from the public methods (DataFrame) to the private methods (array)

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
            The use of `tnew` is discouraged at the moment; DateTimeIndex are not yet supported.
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError('`df` must be a dataframe')

        if inputs is not None:
            if not isinstance(inputs, (str, list)):
                raise TypeError('the input name(s) must be a string or a list of strings')

            if isinstance(inputs, str):
                inputs = [inputs]

            if self.ss.nu != len(inputs):
                raise ValueError(f'The model {self.ss.name} requires {self.ss.nu} inputs')

        if tnew is None:
            if isinstance(df.index, pd.DatetimeIndex):
                dt = np.diff(df.index) / np.timedelta64(1, self._time_scale)
            else:
                dt = np.diff(df.index.astype(np.float64))
            index_back = ()
            n = df.index.shape[0]
        else:
            if isinstance(df.index, pd.DatetimeIndex):
                t = df.index.astype(np.int64) // 1e9
            else:
                t = df.index.astype(np.float64)
            tbase = t.copy()

            if isinstance(tnew, pd.DatetimeIndex):
                tp = tnew.astype(np.int64) // 1e9
            else:
                tp = tnew.astype(np.float64)

            t, index, index_back = np.unique(np.append(tbase, tp), True, True)
            dt = np.diff(t)
            n = t.shape[0]

        # Compute sampling time
        if n > 1:
            dt = np.append(dt, dt[-1])

        if not np.isfinite(dt).all():
            raise ValueError('The time vector contains undefinite values')

        if np.any(dt < 0):
            raise ValueError('The time vector is not monotonically increasing')

        # Input array
        u1 = np.zeros((self.ss.nu, n))
        if self.ss.nu != 0:
            u = np.atleast_2d(df[inputs].T)

            if not np.isfinite(u).all():
                raise ValueError('The input vector contains undefinite values')

            if self.ss.hold_order == 1:
                if tnew is None:
                    u1[:, :-1] = np.diff(u) / dt[:-1]
                else:
                    u1 = np.zeros((self.ss.nu, tbase.shape[0]))
                    u1[:, :-1] = np.diff(u) / np.diff(tbase)
            if tnew is not None:
                u = interp1d(tbase, u, kind='previous')(t)
                u1 = interp1d(tbase, u1, kind='previous')(t)
        else:
            u = u1

        # Output array
        if outputs is None:
            y = np.full((self.ss.ny, n), np.nan, np.float)
        else:
            if not isinstance(outputs, (str, list)):
                raise TypeError("the output names must be a string or a list")

            if isinstance(outputs, str):
                outputs = [outputs]

            if self.ss.ny != len(outputs):
                raise TypeError(f'The model {self.ss.name} requires {self.ss.ny} outputs')

            y = np.atleast_2d(df[outputs].T)
            if not np.isnan(y[~np.isfinite(y)]).all():
                raise TypeError('The output vector must contains numerical values or numpy.nan')

            if tnew is not None:
                y = np.append(y, np.full((self.ss.ny, tp.shape[0]), np.nan, np.float), axis=1)
                y = y[:, index]

        return dt, u, u1, y, index_back

    def _init_parameters(
        self, n_init: int = 1, method: str = 'unconstrained', hpd: float = 0.95
    ) -> np.ndarray:
        """Random initialization of the parameters

        Args:
            n_init: Number of random initialization
            method:
                - **unconstrained**: Uniform draw between [-1, 1] in the uncsontrained space
                - **prior**: Uniform draw from the prior distribution
                - **zero**: Set the unconstrained parameters to 0
                - **fixed**: The current parameter values are used
                - **value**: Uniform draw between the parameter value +/- 25%
            hpd: Highest Prior Density to draw sample from (True for unimodal distribution)

        Returns:
            eta0: Array of unconstrained parameters of shape (n_par, n_init), where n_par is the
            number of free parameters and n_init the number of random initialization
        """

        if not isinstance(n_init, int) or n_init <= 0:
            raise TypeError('`n_init` must an integer greater or equal to 1')

        available_methods = ['unconstrained', 'prior', 'zero', 'fixed', 'value']
        if method not in available_methods:
            raise ValueError(f'`method` must be one of the following {available_methods}')

        if not isinstance(hpd, Real) or not 0.0 < hpd <= 1.0:
            raise ValueError('`hpd` must be between ]0, 1]')

        n_par = len(self.ss.parameters.eta_free)
        if method == 'unconstrained':
            eta0 = np.random.uniform(-1, 1, (n_par, n_init))
        elif method == 'zero':
            eta0 = np.zeros((n_par, n_init))
        else:
            eta0 = np.zeros((n_par, n_init))
            for n in range(n_init):
                if method == 'prior':
                    self.ss.parameters.prior_init(hpd=hpd)
                elif method == 'value':
                    value = np.asarray(self.ss.parameters.theta_sd)
                    lb = value - 0.25 * value
                    ub = value + 0.25 * value
                    self.ss.parameters.theta_sd = np.random.uniform(lb, ub)

                eta0[:, n] = self.ss.parameters.eta_free

        return np.squeeze(eta0)

    def fit(
        self,
        df: pd.DataFrame,
        outputs: Union[str, list],
        inputs: [str, list] = None,
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
