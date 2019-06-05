from typing import List, Union, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from ..utils.statistics import ttest
from .bayesian_filter import SRKalman
from .mcmc import sMMALA
from ..statespace.base import StateSpace


class Regressor:
    """Regressor for stochastic state-space inference and prediction

    Args:
        statespace: state-space model
    """

    def __init__(self, ss: StateSpace):
        self.ss = ss
        self.filter = SRKalman(len(self.ss._names), self.ss.Nx, self.ss.Nu, self.ss.Ny)

        self._compute_hessian = False
        self._jacobian_adjustment = False
        self._penalty_function = True

    def _estimate_states(self, dt, u, u1, y, x0, P0, smooth):
        """Estimate the state posterior distribution

        Args:
            dt: Sampling time array
            u: Input array
            u1: Derivative input array
            y: Measurement array
            x0: Initial state mean
            P0: Initial state deviation
            smooth: Use smoother

        Returns:
            x: Filtered | Smoothed state distribution mean
            P: Filtered | Smoothed state distribution covariance
        """
        # Update state-space model
        self.ss.update()

        # Discretize LTI state-space model
        idx, Ad, B0d, B1d, Qd, *_ = self.ss.discretization(dt)

        if smooth:
            x, P = self.filter.smoothing(
                idx, Ad, B0d, B1d, Qd, self.ss.C, self.ss.D, self.ss.R,
                x0, P0, dt, u, u1, y
            )
        else:
            x, P, _, _ = self.filter.filtering(
                idx, Ad, B0d, B1d, Qd, self.ss.C, self.ss.D, self.ss.R,
                x0, P0, dt, u, u1, y
            )

        return x, P

    def _eval_log_likelihood(self, dt, u, u1, y, x0, P0):
        """Evaluate the negative logarithm of the likelihood

        Args:
            dt: Sampling time array
            u: Input array
            u1: Derivative input array
            y: Measurement array
            x0: Initial state mean
            P0: Initial state deviation

        Returns:
            Negative log-likelihood
        """
        # Update state-space model
        self.ss.update()

        # Discretize LTI state-space model
        idx, Ad, B0d, B1d, Qd, *_ = self.ss.discretization(dt)

        self.log_likelihood = self.filter.log_likelihood(
            idx, Ad, B0d, B1d, Qd, self.ss.C, self.ss.D, self.ss.R,
            x0, P0, dt, u, u1, y
        )

    def _eval_dlog_likelihood(self, dt, u, u1, y):
        """Evaluate the negative log-likelihood and the gradient

        Args:
            dt: Sampling time array
            u: Input array
            u1: Derivative input array
            y: Measurement array

        Returns:
            Negative log-likelihood
            Gradient negative log-likelihood
        """

        # update state-space model
        self.ss.update()

        # Single loop?
        names = self.ss.parameters.names_free
        dC = np.asarray([self.ss.dC[k] for k in names])
        dD = np.asarray([self.ss.dD[k] for k in names])
        dR = np.asarray([self.ss.dR[k] for k in names])
        dx0 = np.asarray([self.ss.dx0[k] for k in names])
        dP0 = np.asarray([self.ss.dP0[k] for k in names])

        # Discretize augmented LTI state-space model
        (idx, Ad, B0d, B1d, Qd,
         dAd, dB0d, dB1d, dQd) = self.ss.discretization(dt)

        out = self.filter.dlog_likelihood(
            idx, Ad, dAd, B0d, dB0d, B1d, dB1d, Qd, dQd,
            self.ss.C, dC, self.ss.D, dD, self.ss.R, dR,
            self.ss.x0, dx0, self.ss.P0, dP0, dt, u, u1, y
        )

        if self.filter._compute_hessian:
            (self.log_likelihood,
             self.dlog_likelihood,
             self.d2log_likelihood) = out
        else:
            self.log_likelihood, self.dlog_likelihood = out

    def _eval_log_posterior(self, eta, dt, u, u1, y):
        """Evaluate the negative log-posterior

        Args:
            eta: array of unconstrained free parameters
            dt: Sampling time array
            u: Input array
            u1: Derivative input array
            y: Measurement array

        Returns:
            Negative log-posterior

        """
        # update unconstrained parameters
        self.ss.parameters.eta = eta

        # evaluate the negative log-likelihood
        self._eval_log_likelihood(dt, u, u1, y, self.ss.x0, self.ss.P0)

        # negative log-posterior distribution function
        self.log_posterior = self.log_likelihood - self.ss.parameters.prior

        if self._penalty_function:
            self.log_posterior += self.ss.parameters.penalty

        if self._jacobian_adjustment:
            self.log_posterior -= self.ss.parameters.theta_log_jacobian

        return self.log_posterior

    def _eval_dlog_posterior(self, eta, dt, u, u1, y):
        """Evaluate the gradient of the negative log-posterior

        Args:
            eta: array of unconstrained free parameters
            dt: Sampling time array
            u: Input array
            u1: Derivative input array
            y: Measurement array

        Returns:
            Negative log-posterior
            Gradient negative log-posterior
        """
        # update unconstrained parameters
        self.ss.parameters.eta = eta

        # evaluate the negative log-likelihood and the gradient
        self._eval_dlog_likelihood(dt, u, u1, y)

        # negative log-posterior distribution function
        self.log_posterior = self.log_likelihood - self.ss.parameters.prior

        # gradient of the negative log-posterior distribution function
        self.dlog_posterior = self.dlog_likelihood - self.ss.parameters.d_prior

        if self._penalty_function:
            self.log_posterior += self.ss.parameters.penalty
            self.dlog_posterior += self.ss.parameters.d_penalty

        self.dlog_posterior *= self.ss.parameters.theta_jacobian

        if self._jacobian_adjustment:
            self.log_posterior -= self.ss.parameters.theta_log_jacobian
            self.dlog_posterior -= self.ss.parameters.theta_dlog_jacobian

        return self.log_posterior, self.dlog_posterior

    def _eval_d2log_posterior(self, eta, dt, u, u1, y):
        """Evaluate the Hessian approximation of the negative log-posterior

        Args:
            eta: array of unconstrained free parameters
            dt: Sampling time array
            u: Input array
            u1: Derivative input array
            y: Measurement array

        Returns:
            Negative log-posterior
            Gradient negative log-posterior
            Hessian approximation negative log-posterior
        """
        # set flag for computing the FIM
        self.filter._compute_hessian = True

        # update unconstrained parameters
        self.ss.parameters.eta = eta

        # evaluate the negative log-likelihood, the gradient and the FIM
        self._eval_dlog_likelihood(dt, u, u1, y)

        # negative log-posterior distribution function
        self.log_posterior = self.log_likelihood - self.ss.parameters.prior

        # gradient of the negative log-posterior distribution function
        self.dlog_posterior = self.dlog_likelihood - self.ss.parameters.d_prior
        self.dlog_posterior *= self.ss.parameters.theta_jacobian

        # Hessian of the negative log-posterior distribution function
        Jd = np.diag(self.ss.parameters.theta_jacobian)
        self.d2log_posterior = Jd @ (self.d2log_likelihood
                                     - np.diag(self.ss.parameters.d2_prior)) @ Jd

        if self._jacobian_adjustment:
            self.log_posterior -= self.ss.parameters.theta_log_jacobian
            self.dlog_posterior -= self.ss.parameters.theta_dlog_jacobian
            self.d2log_posterior -= np.diag(
                self.ss.parameters.theta_d2log_jacobian)

        return (self.log_posterior, self.dlog_posterior,
                self.d2log_posterior)

    def _prepare_data(self, df, inputs, outputs):
        """Prepare the data for the Regressor() methods

        Args:
            df: Pandas DataFrame time index
            inputs: Inputs names, string / list of strings
            outputs: Outputs names, string / list of strings

        Returns:
            dt: Sampling time array
            u: Input array
            u1: Derivative input
            y: Measurement array
        """
        # Check arguments
        if not isinstance(df, pd.DataFrame):
            raise TypeError("`df` must be a dataframe")

        if inputs is not None and not isinstance(inputs, (str, list)):
            raise TypeError("the input name(s) must be a string or a list")

        if isinstance(inputs, str):
            inputs = [inputs]

        # Compute sampling time
        dt = np.empty(df.index.shape[0])
        if isinstance(df.index, pd.DatetimeIndex):
            dt[:-1] = np.diff(df.index) / np.timedelta64(1, 's')
        else:
            dt[:-1] = np.diff(df.index.astype(np.float64))
        dt[-1] = dt[-2]

        if not np.isfinite(dt).all():
            raise ValueError("The tiie vector contains undefinite values")

        if np.any(dt < 0):
            raise ValueError("The time vector isn't monotonically increasing")

        # Prepare input vector
        if self.ss.Nu == 0 and inputs is not None:
            raise ValueError("The model doesn't support an input vector")

        u1 = np.zeros((self.ss.Nu, df.index.shape[0]))
        if self.ss.Nu != 0:
            if inputs is None or self.ss.Nu != len(inputs):
                raise ValueError(f"{self.ss.Nu} inputs are expected")

            u = np.atleast_2d(df[inputs].T)

            if not np.isfinite(u).all():
                raise ValueError("The input vector contains undefinite values")

            if self.ss.hold_order == 'foh':
                u1[:, :-1] = np.diff(u) / dt[:-1]
        else:
            u = u1

        # Prepare output vector
        if outputs is None:
            y = np.full((self.ss.Ny, df.index.shape[0]), np.nan, np.float)
        else:
            if not isinstance(outputs, (str, list)):
                raise TypeError("the output names must be a string or a list")

            if isinstance(outputs, str):
                outputs = [outputs]

            if self.ss.Ny != len(outputs):
                raise TypeError(f"{self.ss.Ny} outputs are expected")

            y = np.atleast_2d(df[outputs].T)

            if not np.isnan(y[~np.isfinite(y)]).all():
                raise TypeError("The output vector must contains "
                                " numerical values or numpy.nan")

        return dt, u, u1, y

    def fit(self, df: pd.DataFrame, outputs: list, inputs: list = None, options: dict = None) -> None:
        """Fit the state-space model

        Args:
            df: training data
            inputs: inputs names
            outputs: outputs names
            options: a dictionary of solver options for SciPy `minimize`_ function

        .. _minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        """
        if options is None:
            options = {}
        else:
            options = dict(options)

        options.setdefault('disp', True)
        method = options.pop('method', 'BFGS')

        data = self._prepare_data(df, inputs, outputs)

        if method == 'BFGS':
            self._jacobian_adjustment = False
            self._penalty_function = True

            if self.ss.jacobian:
                results = minimize(fun=self._eval_dlog_posterior,
                                   x0=self.ss.parameters.eta_free,
                                   args=data,
                                   method=method,
                                   jac=True,
                                   options=options)
            else:
                results = minimize(fun=self._eval_log_posterior,
                                   x0=self.ss.parameters.eta_free,
                                   args=data,
                                   method=method,
                                   options=options)

            # inverse jacobian of the transform eta = f(theta)
            inv_jac = np.diag(1.0 / np.array(self.ss.parameters.eta_jacobian))

            # covariance matrix in the constrained space (e.g. theta)
            cov_theta = inv_jac @ results.hess_inv @ inv_jac

            # standard deviation of the constrained parameters
            sig_theta = np.sqrt(np.diag(cov_theta))
            inv_sig_theta = np.diag(1.0 / sig_theta)

            # correlation matrix of the constrained parameters
            corr_matrix = inv_sig_theta @ cov_theta @ inv_sig_theta
            pd.set_option('display.float_format', '{:.3e}'.format)
            df = pd.DataFrame(index=self.ss.parameters.names_free,
                              columns=['\u03B8', '\u03C3(\u03B8)', 'pvalue',
                                       '|g(\u03B7)|', '|dpen(\u03B8)|'])
            df.iloc[:, 0] = self.ss.parameters.theta_free
            df.iloc[:, 1] = sig_theta
            df.iloc[:, 2] = ttest(self.ss.parameters.theta_free,
                                  sig_theta, data[2].shape[1])
            df.iloc[:, 3] = np.abs(results.jac)
            df.iloc[:, 4] = np.abs(self.ss.parameters.d_penalty)

            df_corr = pd.DataFrame(data=corr_matrix,
                                   index=self.ss.parameters.names_free,
                                   columns=self.ss.parameters.names_free)

            self.summary_ = df
            self.corr_ = df_corr

            results = df, df_corr

        if method == 'sMMALA':
            self._jacobian_adjustment = True
            self._penalty_function = False

            mh = sMMALA(self._eval_d2log_posterior, self.ss.parameters)
            mh.sampling(*data, options)

            results = mh

        return results

    def eval_residuals(self,
                       df: pd.DataFrame,
                       outputs: list,
                       inputs: list = None,
                       x0: float = None,
                       P0: float = None) -> np.ndarray:
        """Compute the standardized residuals

        Args:
            df: data
            inputs: inputs names
            outputs: outputs names
            x0: Initial state mean
            P0: Initial state deviation

        Returns:
            Standardized residuals
        """
        dt, u, u1, y = self._prepare_data(df, inputs, outputs)

        self.ss.update()

        # Discretize LTI state-space model
        idx, Ad, B0d, B1d, Qd, *_ = self.ss.discretization(dt)

        if x0 is None:
            x0 = self.ss.x0

        if P0 is None:
            P0 = self.ss.P0

        _, _, res, _ = self.filter.filtering(
            idx, Ad, B0d, B1d, Qd, self.ss.C, self.ss.D, self.ss.R,
            x0, P0, dt, u, u1, y
        )

        return res.squeeze()

    def estimate_states(self,
                        df: pd.DataFrame,
                        outputs: list,
                        inputs: list = None,
                        x0: float = None,
                        P0: float = None,
                        smooth: bool = False) -> Tuple[np.ndarray]:
        """Estimate the state posterior distribution

        Args:
            df: data
            inputs: inputs names
            outputs: outputs names
            x0: Initial state mean
            P0: Initial state deviation
            smooth: smoother flag

        Returns:
            2-element tuple containing

            - **x**: state distribution mean
            - **P**: state distribution covariance
        """
        dt, u, u1, y = self._prepare_data(df, inputs, outputs)

        if x0 is None:
            x0 = self.ss.x0

        if P0 is None:
            P0 = self.ss.P0

        x, P = self._estimate_states(dt, u, u1, y, x0, P0, smooth)

        return x, P

    def eval_log_likelihood(self,
                            df: pd.DataFrame,
                            outputs: list,
                            inputs: list = None,
                            x0: float = None,
                            P0: float = None) -> float:
        """Evaluate the negative log-likelihood

        Args:
            df: data
            inputs: inputs names
            outputs: outputs names
            x0: Initial state mean
            P0: Initial state deviation

        Returns:
            Negative log-likelihood
        """
        dt, u, u1, y = self._prepare_data(df, inputs, outputs)

        if x0 is None:
            x0 = self.ss.x0

        if P0 is None:
            P0 = self.ss.P0

        self._eval_log_likelihood(dt, u, u1, y, x0, P0)

    def predict(self,
                df: pd.DataFrame,
                outputs: list = None,
                inputs: list = None,
                tpred: pd.Series = None,
                x0: float = None,
                P0: float = None,
                smooth: bool = False):
        """State-space model output prediction / simulation / interpolation

        The returns values will be either predictive, filtered or smoothed.

        Args:
            df: data
            inputs: inputs names
            outputs: outputs names
            tpred: New time vector
            x0: Initial state mean
            P0: Initial state deviation
            smooth: smoother flag

        Returns:
            2-element tuple containing

                - **y_mean**: Output distribution mean
                - **y_std**: Output distribution deviation
        """
        if self.ss.Ny > 1:
            raise NotImplementedError

        dt, u, u1, y = self._prepare_data(df, inputs, outputs)

        # Change of state initial conditions
        if x0 is None:
            x0 = self.ss.x0

        if P0 is None:
            P0 = self.ss.P0

        if tpred is not None:
            # merge time index
            if isinstance(df.index, pd.DatetimeIndex):
                tf = df.index.astype(np.int64) // 1e9
            else:
                tf = df.index.astype(np.float64)

            if isinstance(tpred, pd.DatetimeIndex):
                tp = tpred.astype(np.int64) // 1e9
            else:
                tp = tpred.astype(np.float64)

            t, index, index_back = np.unique(np.append(tf, tp), True, True)

            dt = np.empty(t.shape[0])
            dt[:-1] = np.diff(t)
            dt[-1] = dt[-2]
            if not np.isfinite(dt).all():
                raise ValueError("The tiie vector contains undefinite values")

            if np.any(dt < 0):
                raise ValueError(
                    "The time vector isn't monotonically increasing")

            # Prepare input
            if self.ss.Nu != 0:
                if self.ss.hold_order == 'foh':
                    u1 = interp1d(tf, u1, kind='previous')(t)
                else:
                    u1 = np.zeros((self.ss.Nu, t.shape[0]))
                u = interp1d(tf, u, kind='previous')(t)
            else:
                u = np.zeros((self.ss.Nu, t.shape[0]))
                u1 = u

            # Prepare output
            ynan = np.full((self.ss.Ny, tp.shape[0]), np.nan, np.float)
            y = np.append(y, ynan, 1)
            y = y[:, index]

        x, P = self._estimate_states(dt, u, u1, y, x0, P0, smooth)

        # keep only the part corresponding to `tpred`
        if tpred is not None:
            x = x[index_back, :, :]
            P = P[index_back, :, :]
            x = x[-tpred.shape[0]:, :, :]
            P = P[-tpred.shape[0]:, :, :]

        # state mean and state standard deviation
        y_mean = self.ss.C @ x
        y_std = np.sqrt(self.ss.C @ P @ self.ss.C.T) + self.ss.R

        return np.squeeze(y_mean), np.squeeze(y_std)
