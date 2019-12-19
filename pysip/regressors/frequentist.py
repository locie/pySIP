"""Frequentist regressor"""
from typing import Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .base import BaseRegressor
from ..state_estimator import BayesianFilter, Kalman_QR
from ..statespace.base import StateSpace
from ..utils.statistics import ttest


class FreqRegressor(BaseRegressor):
    """Frequentist Regressor

    Args:
        ss: StateSpace()
        bayesian_filter: BayesianFilter()
        time_scale: Time series frequency, e.g. 's': seconds, 'D': days, etc.
            Works only for pandas.DataFrame with DateTime index
    """

    def __init__(
        self, ss: StateSpace, bayesian_filter: BayesianFilter = Kalman_QR, time_scale: str = 's'
    ):
        super().__init__(ss, bayesian_filter, time_scale, False, True)

    def fit(
        self,
        df: pd.DataFrame,
        outputs: Union[str, list],
        inputs: [str, list] = None,
        options: dict = None,
    ) -> Union[pd.DataFrame, pd.DataFrame, dict]:
        """Fit the model

        Args:
            df: Training data
            outputs: Outputs name(s)
            inputs: Inputs name(s)
            options:
                - scipy.minimmize options
                - **init** (str, default=`unconstrained`):
                    - unconstrained: Uniform draw between [-1, 1] in the uncsontrained space
                    - prior: Uniform draw from the prior distribution
                    - zero: Set the unconstrained parameters to 0
                    - fixed: The current parameter values are used
                    - value: Uniform draw between the parameter value +/- 25%
                    - prior_mass (float, default=0.95):
                - **hpd**: (float, default=0.95)
                    Highest Prior Density to draw sample from the prior (True if unimodal)

        Returns:
            3-elements tuple containing
                - **df**: Fit summary
                - **df_corr**: Correlation matrix
                - **results**: Scipy optimize summary
        """

        if options is None:
            options = {}
        else:
            options = dict(options)

        options.setdefault('disp', True)
        options.setdefault('gtol', 1e-4)

        init = options.pop('init', 'fixed')
        hpd = options.pop('hpd', 0.95)
        self.ss.parameters.eta = self._init_parameters(1, init, hpd)
        data = self._prepare_data(df, inputs, outputs, None)[:-1]

        results = minimize(
            fun=self._eval_dlog_posterior,
            x0=self.ss.parameters.eta_free,
            args=data,
            method='BFGS',
            jac=True,
            options=options,
        )

        # inverse jacobian of the transform eta = f(theta)
        inv_jac = np.diag(1.0 / np.array(self.ss.parameters.eta_jacobian))

        # covariance matrix in the constrained space (e.g. theta)
        cov_theta = inv_jac @ results.hess_inv @ inv_jac

        # standard deviation of the constrained parameters
        sig_theta = np.sqrt(np.diag(cov_theta)) * self.ss.parameters.scale
        inv_sig_theta = np.diag(1.0 / np.sqrt(np.diag(cov_theta)))

        # correlation matrix of the constrained parameters
        corr_matrix = inv_sig_theta @ cov_theta @ inv_sig_theta
        pd.set_option('display.float_format', '{:.3e}'.format)
        df = pd.DataFrame(
            index=self.ss.parameters.names_free,
            columns=['\u03B8', '\u03C3(\u03B8)', 'pvalue', '|g(\u03B7)|', '|dpen(\u03B8)|'],
        )
        df.iloc[:, 0] = self.ss.parameters.theta_free
        df.iloc[:, 1] = sig_theta
        df.iloc[:, 2] = ttest(self.ss.parameters.theta_free, sig_theta, data[2].shape[1])
        df.iloc[:, 3] = np.abs(results.jac)
        df.iloc[:, 4] = np.abs(self.ss.parameters.d_penalty)

        df_corr = pd.DataFrame(
            data=corr_matrix,
            index=self.ss.parameters.names_free,
            columns=self.ss.parameters.names_free,
        )

        self.summary_ = df
        self.corr_ = df_corr
        self.results_ = results

        return df, df_corr, results

    def eval_residuals(
        self,
        df: pd.DataFrame,
        outputs: Union[str, list],
        inputs: Union[str, list] = None,
        x0: np.ndarray = None,
        P0: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the standardized residuals

        Args:
            df: Data
            outputs: Outputs name(s)
            inputs: Inputs name(s)
            x0: Initial state mean
            P0: Initial state deviation

        Returns:
            2-element tuple containing
                - **res**: Standardized residuals
                - **res_std**: Residuals deviations
        """

        dt, u, u1, y, *_ = self._prepare_data(df, inputs, outputs, None)
        ssm, index = self.ss.get_discrete_ssm(dt)
        res, res_std = self.filter.filtering(ssm, index, u, u1, y)[2:]

        return res.squeeze(), res_std.squeeze()

    def estimate_states(
        self,
        df: pd.DataFrame,
        outputs: list,
        inputs: list = None,
        x0: np.ndarray = None,
        P0: np.ndarray = None,
        smooth: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate the state filtered/smoothed distribution

        Args:
            df: Data
            inputs: Inputs names
            outputs: Outputs names
            x0: Initial state mean
            P0: Initial state deviation
            smooth: Use smoother

        Returns:
            2-element tuple containing
                - state mean
                - state covariance
        """

        dt, u, u1, y, *_ = self._prepare_data(df, inputs, outputs, None)
        return self._estimate_states(dt, u, u1, y, x0, P0, smooth)

    def eval_log_likelihood(
        self,
        df: pd.DataFrame,
        outputs: Union[str, list],
        inputs: Union[str, list] = None,
        pointwise: bool = False,
    ) -> Union[float, np.ndarray]:
        """Evaluate the negative log-likelihood

        Args:
            df: Data
            outputs: Outputs name(s)
            inputs: Inputs name(s)
            pointwise: Evaluate the log-likelihood pointwise

        Returns:
            Negative log-likelihood or predictive density evaluated point-wise
        """

        dt, u, u1, y, *_ = self._prepare_data(df, inputs, outputs, None)
        return self._eval_log_likelihood(dt, u, u1, y, pointwise)

    def predict(
        self,
        df: pd.DataFrame,
        outputs: Union[str, list] = None,
        inputs: Union[str, list] = None,
        tnew: Union[np.ndarray, pd.Series] = None,
        x0: np.ndarray = None,
        P0: np.ndarray = None,
        smooth: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """State-space model output prediction

        Args:
            df: Data
            outputs: Outputs name(s)
            inputs: Inputs name(s)
            tnew: New time instants
            x0: Initial state mean
            P0: Initial state deviation
            smooth: Use smoother

        Returns:
            2-element tuple containing
                - **y_mean**: Output mean
                - **y_std**: Output deviation
        """

        if self.ss.ny > 1:
            raise NotImplementedError

        dt, u, u1, y, index_back = self._prepare_data(df, inputs, outputs, tnew)
        x, P = self._estimate_states(dt, u, u1, y, x0, P0, smooth)

        # keep only the part corresponding to `tnew`
        if tnew is not None:
            x = x[index_back, :, :]
            P = P[index_back, :, :]
            x = x[-tnew.shape[0] :, :, :]
            P = P[-tnew.shape[0] :, :, :]

        y_mean = self.ss.C @ x
        y_std = np.sqrt(self.ss.C @ P @ self.ss.C.T) + self.ss.R

        return np.squeeze(y_mean), np.squeeze(y_std)
