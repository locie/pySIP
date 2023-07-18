from concurrent.futures import ProcessPoolExecutor
from functools import partial
import warnings
from copy import deepcopy
import numpy as np
from multiprocessing import cpu_count

import pymc as pm
import pytensor.tensor as pt
from scipy.optimize import approx_fprime
from .frequentist import Regressor


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


class BayesRegressor(Regressor):
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

    # def prior_predictive(
    #     self,
    #     df: pd.DataFrame,
    #     inputs: Union[str, list] = None,
    #     n_sim: int = 1000,
    #     hpd: Real = None,
    #     n_cpu: int = -1,
    # ) -> Tuple[dict, dict]:
    #     """Prior predictive distribution

    #     Args:
    #         df: Data
    #         inputs: Input name(s)
    #         n_sim: Number of prior predictive draw
    #         hpd: Highest Prior Density to draw sample from (True for unimodal
    #           distribution)
    #         n_cpu: Number of cpu

    #     Returns:
    #         2-elements tuple containing
    #             - **draws**: Parameter draws
    #             - **ppc**: Prior predictive samples
    #     """
    #     if self.ss.ny > 1:
    #         raise ValueError("Multiple outputs is not yet supported")

    #     if not isinstance(n_sim, int) or n_sim <= 0:
    #         raise TypeError("`n_sim` must be a positive integer")

    #     if not isinstance(n_cpu, int) or (n_cpu != -1 and n_cpu <= 0):
    #         raise TypeError("`n_cpu` must be a strictly positive integer or set to -1")

    #     dt, u, u1, *_ = self._prepare_data(df, inputs, None, None)

    #     ps = np.empty((n_sim, self.ss.parameters.n_par))
    #     ppd = np.empty((n_sim, df.index.shape[0], self.ss.ny))

    #     def prior_pd():
    #         self.ss.parameters.prior_init(hpd)
    #         return self.ss.parameters.theta_free, self._simulate_output(dt, u, u1, None)

    #     pbar = tqdm(range(n_sim), desc="Sampling")
    #     out = Parallel(n_jobs=n_cpu)(delayed(prior_pd)() for _ in pbar)

    #     for n in range(n_sim):
    #         ps[n, :] = out[n][0]
    #         ppd[n, :, :] = out[n][1]

    #     # Safety against duplication of parameter names. Do not use
    #     # self.ss.parameters.names
    #     names = [k for i, k in enumerate(self.ss.names) if self.ss.parameters.free[i]]
    #     draws = {k: ps[:, i][np.newaxis, :] for i, k in enumerate(names)}
    #     ppc = {
    #         k.name: ppd[:, :, i][np.newaxis, :] for i, k in enumerate(self.ss.outputs)
    #     }

    #     return draws, ppc

    # def posterior_predictive(
    #     self,
    #     trace: dict,
    #     df: pd.DataFrame,
    #     inputs: Union[str, list] = None,
    #     outputs: Union[str, list] = None,
    #     n_cpu: int = -1,
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     """Posterior predictive distribution

    #     Args:
    #         trace: Markov Chain traces, trace[key](chain, draw)
    #         df: Data
    #         inputs: Input name(s)
    #         outputs: Output name(s)
    #         n_cpu: Number of cpus

    #     Returns:
    #         2-elements tuple containing
    #             - **ym**: Mean of the predictive distribution
    #             - **ysd**: Standard deviation of the predictive distribution
    #     """
    #     if self.ss.ny > 1:
    #         raise ValueError("Multiple outputs is not yet supported")

    #     if not isinstance(n_cpu, int) or (n_cpu != -1 and n_cpu <= 0):
    #         raise TypeError("`n_cpu` must be a strictly positive integer or set to -1")

    #     dt, u, u1, y, *_ = self._prepare_data(df, inputs, outputs, None)

    #     chain, draw = list(trace.values())[0].shape
    #     n_draws = chain * draw
    #     names = [n for n, f in zip(self.ss.names, self.ss.parameters.free) if f]
    #     samples = self._dict_to_array(trace, names)

    #     def posterior_pd(index):
    #         self.ss.parameters.theta_free = samples[:, index]
    #         return self._estimate_output(dt, u, u1, y, None, None)

    #     pbar = tqdm(range(n_draws), desc="Sampling")
    #     out = Parallel(n_jobs=n_cpu)(delayed(posterior_pd)(i) for i in pbar)

    #     ym = np.empty((n_draws, dt.shape[0]))
    #     ysd = np.empty((n_draws, dt.shape[0]))
    #     for n in range(n_draws):
    #         ym[n], ysd[n] = np.squeeze(out[n])

    #     return ym, ysd

    # def pointwise_log_likelihood(
    #     self,
    #     trace: dict,
    #     df: pd.DataFrame,
    #     outputs: Union[str, list],
    #     inputs: Union[str, list] = None,
    #     n_cpu: int = -1,
    # ) -> dict:
    #     """Point-wise log-likelihood for Pareto Smoothing Importance Sampling

    #     Args:
    #         trace: Markov Chain traces, trace[key](chain, draw)
    #         df: Data
    #         outputs: Output name(s)
    #         inputs: Input name(s)
    #         cpu: Number of cpus

    #     Returns:
    #         Positive log-likelihood evaluated point-wise
    #     """
    #     if not isinstance(n_cpu, int) or (n_cpu != -1 and n_cpu <= 0):
    #         raise TypeError("`n_cpu` must be a strictly positive integer or set to -1")

    #     dt, u, u1, y, *_ = self._prepare_data(df, inputs, outputs, None)
    #     chain, draw = list(trace.values())[0].shape
    #     names = [n for n, f in zip(self.ss.names, self.ss.parameters.free) if f]
    #     samples = self._dict_to_array(trace, names)

    #     def eval_loglik_pw(index):
    #         self.ss.parameters.theta_free = samples[:, index]
    #         return self._eval_log_likelihood(dt, u, u1, y, pointwise=True)

    #     pbar = tqdm(range(chain * draw), desc="Sampling")
    #     out = Parallel(n_jobs=n_cpu)(delayed(eval_loglik_pw)(i) for i in pbar)

    #     pw_loglik = np.empty((chain * draw, y.shape[1]))
    #     for n in range(chain * draw):
    #         pw_loglik[n, :] = -out[n]

    #     return {"log_likelihood": pw_loglik.reshape(chain, draw, y.shape[1])}

    # def posterior_state_distribution(
    #     self,
    #     trace: dict,
    #     df: pd.DataFrame,
    #     outputs: Union[str, list],
    #     inputs: Union[str, list] = None,
    #     smooth: bool = False,
    #     n_cpu: int = -1,
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     """Posterior state filtered/smoothed distribution

    #     Args:
    #         trace: Markov Chain traces, trace[key](chain, draw)
    #         df: Data
    #         outputs: Output name(s)
    #         inputs: Input name(s)
    #         smooth: Use Kalman smoother
    #         cpu: Number of cpus

    #     Returns:
    #         2-element tuple containing
    #             - **xm**: State filtered/smoothed distribution mean
    #             - **xsd**: State filtered/smoothed distribution standard deviation
    #     """
    #     if not isinstance(n_cpu, int) or (n_cpu != -1 and n_cpu <= 0):
    #         raise TypeError("`n_cpu` must be a strictly positive integer or set to -1")

    #     dt, u, u1, y, *_ = self._prepare_data(df, inputs, outputs, None)

    #     chain, draw = list(trace.values())[0].shape
    #     n_draws = chain * draw
    #     samples = self._dict_to_array(trace, self.ss.parameters.names_free)

    #     def state_pd(index):
    #         self.ss.parameters.theta_free = samples[:, index]
    #         x, P = self._estimate_states(dt, u, u1, y, smooth=smooth)
    #         return x, np.sqrt(P.diagonal(0, 1, 2))

    #     pbar = tqdm(range(n_draws), desc="Sampling")
    #     out = Parallel(n_jobs=n_cpu)(delayed(state_pd)(i) for i in pbar)

    #     xm = np.empty((n_draws, dt.shape[0], self.ss.nx))
    #     xsd = np.empty((n_draws, dt.shape[0], self.ss.nx))
    #     for n in range(n_draws):
    #         xm[n] = np.squeeze(out[n][0])
    #         xsd[n] = np.squeeze(out[n][1])

    #     return xm, xsd
