from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from multiprocessing import cpu_count
from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from scipy.optimize import approx_fprime, minimize

from .params.parameters import Parameters
from .statespace.base import StateSpace
from .statespace_estimator import KalmanQR
from .utils.statistics import ttest


def _make_estimate(theta, data, estimator):
    estimator = deepcopy(estimator)
    estimator.ss.parameters.theta_free = theta
    y_res, *_ = estimator.estimate_output(*data)
    return y_res


class FdiffLoglikeGrad(pt.Op):
    itypes = [pt.dvector]
    otypes = [pt.dvector]

    def __init__(self, reg: Regressor, df, eps=None):
        self.estimator = reg.estimator
        self.data = reg.prepare_data(df)
        self.eps = eps

    def perform(self, _, inputs, outputs):
        def _target(eta) -> float:
            estimator = deepcopy(self.estimator)
            estimator.ss.parameters.theta_free = eta
            return estimator.log_likelihood(*self.data)

        (eta,) = inputs  # this will contain my variables
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outputs[0][0] = approx_fprime(eta, _target, self.eps)


class Loglike(pt.Op):
    itypes = [pt.dvector]
    otypes = [pt.dscalar]

    def __init__(self, reg: Regressor, df, eps=None):
        self.estimator = reg.estimator
        self.data = reg.prepare_data(df)
        self.eps = eps
        self.logpgrad = FdiffLoglikeGrad(reg, df, eps)

    def perform(self, _, inputs, outputs):
        estimator = deepcopy(self.estimator)
        (eta,) = inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.ss.parameters.theta_free = eta
            logl = estimator.log_likelihood(*self.data)
        outputs[0][0] = np.array(logl)

    def grad(self, inputs, g):
        (eta,) = inputs
        return [g[0] * self.logpgrad(eta)]


@dataclass
class Regressor:
    """A regressor use a statespace model and a bayesian filter to predict the states of
    the system given the exogeneous (or boundary) inputs and the initial conditions.

    This base class does not have the ability to estimate the parameters of the
    statespace model : see the derived classes for that.


    Parameters
    ----------
    ss : StateSpace()
        State-space model
    inputs : str or list of str, optional
        Input column names, by default None. If None, they are inferred from the
        state-space model.
    outputs : str or list of str, optional
        Output column names, by default None. If None, they are inferred from the
        state-space model.
    time_scale : str
        Time series frequency, e.g. 's': seconds, 'D': days, etc.
    """

    ss: StateSpace
    inputs: Optional[Union[str, Sequence[str]]] = None
    outputs: Optional[Union[str, Sequence[str]]] = None
    time_scale: str = "s"

    def __post_init__(self):
        self.estimator = KalmanQR(self.ss)
        if self.inputs is None:
            self.inputs = [node.name for node in self.ss.inputs]
        if self.outputs is None:
            self.outputs = [node.name for node in self.ss.outputs]
        self.states = [node.name for node in self.ss.states]
        if isinstance(self.inputs, str):
            self.inputs = [self.inputs]
        if isinstance(self.outputs, str):
            self.outputs = [self.outputs]

    @property
    def parameters(self) -> Parameters:
        return self.ss.parameters

    def prepare_data(self, df, with_outputs=True) -> Tuple[pd.DataFrame, ...]:
        """Prepare data for training

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the data
        with_outputs : bool, optional
            Whether to return the outputs, by default True

        Returns
        -------
        DataFrame:
            time steps
        DataFrame:
            input data
        DataFrame:
            derivative of input data
        DataFrame:
            output data (filled with NaNs if with_outputs=False)
        """
        return self.ss.prepare_data(
            df, self.inputs, self.outputs if with_outputs else False, self.time_scale
        )

    def simulate(self, df: pd.DataFrame) -> np.ndarray:
        """Stochastic simulation of the state-space model

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing the inputs and outputs
        inputs : str or list of str, optional
            Input column names, by default None. If None, the regressor's `inputs`
            attribute is used.
        time_scale : str, optional
            Time series frequency, e.g. 's': seconds, 'D': days, etc., by default None.
            If None, the regressor's `time_scale` attribute is used.

        Returns
        -------
        xarray.Dataset
            Dataset containing the simulated outputs and states
        """

        dt, u, dtu, _ = self.prepare_data(df, with_outputs=False)
        return self.estimator.simulate(dt, u, dtu).to_xarray(
            df.index.name or "time", df.index, self.states, self.outputs
        )

    def estimate_output(
        self,
        df: pd.DataFrame,
        x0: np.ndarray = None,
        P0: np.ndarray = None,
        use_outputs: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate the output filtered distribution

        Parameters
        ----------


        Returns
        -------

        """

        dt, u, dtu, y = self.prepare_data(df, with_outputs=use_outputs)
        return self.estimator.estimate_output(dt, u, dtu, y, x0, P0).to_xarray(
            df.index.name or "time", df.index, self.states, self.outputs
        )

    def estimate_states(
        self,
        df: pd.DataFrame,
        x0: np.ndarray = None,
        P0: np.ndarray = None,
        smooth: bool = False,
        use_outputs: bool = True,
    ) -> xr.Dataset:
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

        dt, u, dtu, y = self.prepare_data(df, with_outputs=use_outputs)

        return (
            self.estimator.smoothing(dt, u, dtu, y, x0, P0)
            if smooth
            else self.estimator.filtering(dt, u, dtu, y, x0, P0)
        ).to_xarray(df.index.name or "time", df.index, self.states, self.outputs)

    def _target(
        self,
        eta: np.ndarray,
        dt: np.ndarray,
        u: np.ndarray,
        dtu: np.ndarray,
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
        dtu : array_like, shape (n_u, n_steps)
            Forward finite difference of the input data
        y : array_like, shape (n_y, n_steps)
            Output data

        Returns
        -------
        log_posterior : float
            The negative log-posterior
        """
        estimator = deepcopy(self.estimator)
        estimator.ss.parameters.eta = eta
        log_likelihood = estimator.log_likelihood(dt, u, dtu, y)
        log_posterior = (
            log_likelihood
            - estimator.ss.parameters.prior
            + estimator.ss.parameters.penalty
        )

        return log_posterior

    def fit(
        self,
        df: pd.DataFrame,
        options: dict = None,
        *,
        init: Literal["unconstrained", "prior", "zero", "fixed", "value"] = "fixed",
        hpd: float = 0.95,
        jac="2-point",
        method="BFGS",
        **minimize_options,
    ) -> Union[pd.DataFrame, pd.DataFrame, dict]:
        """Estimate the parameters of the state-space model.

        Parameters
        ----------
        df : pandas.DataFrame
            Training data
        outputs : str or list of str, optional
            Output name(s)
        inputs : str or list of str, optional
            Input name(s)
        options : dict, optional, deprecated
            Options for the minimization method. You should use named arguments instead.
            Usage of this argument will raise a warning and will be removed in the
            future.
        init : str, optional
            Method to initialize the parameters. Options are:
            - 'unconstrained': initialize the parameters in the unconstrained space
            - 'prior': initialize the parameters using the prior distribution
            - 'zero': initialize the parameters to zero
            - 'fixed': initialize the parameters to the fixed values
            - 'value': initialize the parameters to the given values
        hpd : float, optional
            Highest posterior density interval. Used only when `init='prior'`.
        minimize_options : dict, optional
            Options for the minimization method. See `scipy.optimize.minimize` for
            details. Compared to the original `scipy.optimize.minimize` function, the
            following options are set by default:
            - `method='BFGS'`
            - `jac='3-point'`
            - `disp=True`
            - `gtol=1e-4`

        Returns
        -------
        pandas.DataFrame
            Dataframe with the estimated parameters, their standard deviation, the
            p-value of the t-test and penalty values.
        pandas.DataFrame
            Dataframe with the correlation matrix of the estimated parameters.
        dict
            Results object from the minimization method. See `scipy.optimize.minimize`
            for details.
        """
        if options is not None:
            warnings.warn(
                "Use of the options argument is deprecated and will raise an error in "
                "the future. Prefer using named arguments (kwargs) instead.",
                DeprecationWarning,
            )
            minimize_options.update(options)
        minimize_options = {"disp": True, "gtol": 1e-4} | minimize_options

        self.parameters.eta = self.parameters.init_parameters(1, init, hpd)
        data = self.prepare_data(df)

        results = minimize(
            fun=self._target,
            x0=self.parameters.eta_free,
            args=data,
            method=method,
            jac=jac,
            options=minimize_options,
        )

        self.parameters.eta = results.x
        # inverse jacobian of the transform eta = f(theta)
        inv_jac = np.diag(1.0 / np.array(self.parameters.eta_jacobian))

        # covariance matrix in the constrained space (e.g. theta)
        cov_theta = inv_jac @ results.hess_inv @ inv_jac

        # standard deviation of the constrained parameters
        sig_theta = np.sqrt(np.diag(cov_theta)) * self.parameters.scale
        inv_sig_theta = np.diag(1.0 / np.sqrt(np.diag(cov_theta)))

        # correlation matrix of the constrained parameters
        corr_matrix = inv_sig_theta @ cov_theta @ inv_sig_theta
        df = pd.DataFrame(
            data=np.vstack(
                [
                    self.parameters.theta_free,
                    sig_theta,
                    ttest(self.parameters.theta_free, sig_theta, len(data[0])),
                    np.abs(results.jac),
                    np.abs(self.parameters.d_penalty),
                ]
            ).T,
            columns=["θ", "σ(θ)", "pvalue", "|g(η)|", "|dpen(θ)|"],
            index=self.parameters.names_free,
        )
        df_corr = pd.DataFrame(
            data=corr_matrix,
            index=self.parameters.names_free,
            columns=self.parameters.names_free,
        )

        self.summary_ = df
        self.corr_ = df_corr
        self.results_ = results

        return df, df_corr, results

    def eval_residuals(
        self,
        df: pd.DataFrame,
        x0: np.ndarray = None,
        P0: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the standardized residuals

        Parameters
        ----------
        df : pandas.DataFrame
            Training data
        outputs : str or list of str, optional
            Output name(s)
        inputs : str or list of str, optional
            Input name(s)
        x0 : numpy.ndarray, optional
            Initial state. If not provided, the initial state is taken from the
            state-space model defaults.
        P0 : numpy.ndarray, optional
            Initial state covariance. If not provided, the initial state covariance is
            taken from the state-space model defaults.

        Returns
        -------
        res : numpy.ndarray
            Standardized residuals
        res_std : numpy.ndarray
            Residuals deviations
        """

        dt, u, dtu, y = self.prepare_data(df)
        res = self.estimator.filtering(dt, u, dtu, y, x0, P0).to_xarray(
            df.index.name or "time", df.index, self.states, self.outputs
        )
        return res[["k", "S"]].rename({"k": "residual", "S": "residual_std"})

    def log_likelihood(self, df: pd.DataFrame) -> Union[float, np.ndarray]:
        """Evaluate the negative log-likelihood

        Args:
            df: Data
            outputs: Outputs name(s)
            inputs: Inputs name(s)
            pointwise: Evaluate the log-likelihood pointwise

        Returns:
            Negative log-likelihood or predictive density evaluated point-wise
        """

        dt, u, dtu, y = self.prepare_data(df)
        return self.estimator.log_likelihood(dt, u, dtu, y)

    def log_posterior(
        self,
        df: pd.DataFrame,
    ) -> Union[float, np.ndarray]:
        return (
            self.log_likelihood(df)
            - self.ss.parameters.prior
            + self.ss.parameters.penalty
        )

    def predict(
        self,
        df: pd.DataFrame,
        tnew: Union[np.ndarray, pd.Series] = None,
        x0: np.ndarray = None,
        P0: np.ndarray = None,
        smooth: bool = False,
        use_outputs: bool = False,
    ) -> xr.Dataset:
        """State-space model output prediction

        Parameters
        ----------
        df : pandas.DataFrame
            Training data
        tnew : numpy.ndarray or pandas.Series, optional
            New time instants
        x0 : numpy.ndarray, optional
            Initial state. If not provided, the initial state is taken from the
            state-space model.
        P0 : numpy.ndarray, optional
            Initial state covariance. If not provided, the initial state covariance is
            taken from the state-space model.
        smooth : bool, optional
            If True, the Kalman smoother is used instead of the Kalman filter
        """

        if self.ss.ny > 1:
            raise NotImplementedError
        if tnew is not None:
            itp_df = pd.concat(
                [df, df.reindex(tnew).drop(df.index, errors="ignore")]
            ).sort_index()
            itp_df[self.inputs] = itp_df[self.inputs].interpolate(method="linear")
        else:
            itp_df = df.copy()

        ds = self.estimate_states(
            itp_df, smooth=smooth, use_outputs=use_outputs, x0=x0, P0=P0
        )
        idx_name = df.index.name or "time"
        if tnew is not None:
            ds = ds.sel(**{idx_name: tnew})
        ds["y_mean"] = (
            (idx_name, "outputs"),
            (self.ss.C @ ds.x_predict.values.reshape(-1, self.ss.nx, 1))[..., 0],
        )
        ds["y_std"] = (
            (idx_name, "outputs", "outputs"),
            np.sqrt(self.ss.C @ ds.P_predict.values @ self.ss.C.T) + self.ss.R,
        )
        return ds

    @property
    def pymc_model(reg):
        class PyMCModel(pm.Model):
            def __init__(self, df):
                super().__init__()
                theta = []
                for name, par in zip(
                    reg.parameters.names_free, reg.parameters.parameters_free
                ):
                    # theta.append(pm.Normal(name, par.eta, 1))
                    theta.append(par.prior.pymc_dist(name))
                theta = pt.as_tensor_variable(theta)
                pm.Potential("likelihood", -Loglike(reg, df)(theta))

        return PyMCModel

    def sample(self, df, draws=1000, tune=500, chains=4, **kwargs):
        cores = min(kwargs.pop("cores", cpu_count()), chains)
        with self.pymc_model(df):
            self._trace = pm.sample(
                draws=draws, tune=tune, chains=chains, cores=cores, **kwargs
            )
        return self.trace

    def prior_predictive(self, df, samples=1000, **kwargs):
        with self.pymc_model(df):
            self._prior_trace = pm.sample_prior_predictive(samples=samples, **kwargs)
        parameters = self.trace.prior.to_dataframe().to_numpy()
        data = self.prepare_data(df)
        with ProcessPoolExecutor() as executor:
            results = list(
                executor.map(
                    partial(_make_estimate, data=data, estimator=self.estimator),
                    parameters,
                )
            )
        return results

    def posterior_predictive(self, df, **kwargs):
        parameters = self.trace.posterior.to_dataframe().to_numpy()
        data = self.prepare_data(df)

        with ProcessPoolExecutor() as executor:
            results = list(
                executor.map(
                    partial(_make_estimate, data=data, estimator=self.estimator),
                    parameters,
                )
            )
        return results

    @property
    def prior_trace(self):
        return getattr(self, "_prior_trace", None)

    @property
    def posterior_trace(self):
        return getattr(self, "_posterior_trace", None)

    @property
    def trace(self):
        main_trace = getattr(self, "_trace", None)
        if main_trace is None:
            raise ValueError("Model has not been fitted yet")
        if (prior_trace := self.prior_trace) is not None:
            main_trace.extend(prior_trace)
        if (posterior_trace := self.prior_trace) is not None:
            main_trace.extend(posterior_trace)
        return main_trace
