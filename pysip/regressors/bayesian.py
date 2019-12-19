from typing import Tuple, Union
from numbers import Real
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from .base import BaseRegressor
from ..state_estimator import BayesianFilter, Kalman_QR
from ..mcmc.hamiltonian import EuclideanHamiltonian
from ..mcmc.hmc import DynamicHMC, Fit_Bayes
from ..statespace.base import StateSpace


class BayesRegressor(BaseRegressor):
    """Bayesian Regressor

    Args:
        ss: StateSpace()
        bayesian_filter: BayesianFilter()
        time_scale: Time series frequency, e.g. 's': seconds, 'D': days, etc.
            Works only for pandas.DataFrame with DateTimeIndex
    """

    def __init__(
        self, ss: StateSpace, bayesian_filter: BayesianFilter = Kalman_QR, time_scale: str = 's'
    ):
        super().__init__(ss, bayesian_filter, time_scale, True, False)

    def fit(
        self,
        df: pd.DataFrame,
        outputs: Union[str, list],
        inputs: Union[str, list] = None,
        options: dict = None,
    ) -> Fit_Bayes:
        """Bayesian inference of the state-space model

        Args:
            df: Training data
            outputs: Outputs name(s)
            inputs: Inputs name(s)
            options:
                - **stepsize** (float, default=0.25 / n_par**0.25)
                    Step-size of the leapfrog integrator
                - **max_tree_depth** (int, default=10)
                    Maximum tree depth
                - **dH_max** (float, default=1000)
                    Maximum energy change allowed in a trajectory. Larger deviations are considered
                    as diverging transitions.
                - **accp_target** (float, default=0.8):
                    Target average acceptance probability. Valid values are between ]0, 1[
                - **t0** (float, default=10.0):
                    Adaptation iteration offset (primal-dual averaging algorithm parameter)
                - **gamma** (float, default=0.05):
                    Adaptation regularization scale (primal-dual averaging algorithm parameter)
                - **kappa** (float, default=0.75):
                    Adaptation relaxation exponent (primal-dual averaging algorithm parameter)
                - **mu** (float, default=log(10 * stepsize)):
                    Asymptotic mean of the step-size (primal-dual averaging algorithm parameter)
                - **n_cpu**: (int, default=-1):
                    Number of cpu to use. To use all available cpu, set the value to -1,
                    otherwise valid values are between [1, +Inf[
                - **init_buffer**: (int, default=75)
                    Width of initial fast adaptation interval
                - **term_buffer**: (int, default=150)
                    Width of final fast adaptation interval
                - **window**: (int, default=25)
                    Initial width of slow adaptation interval
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
            Fit_Bayes(): An instance summarizing the results from the Bayesian inference
        """
        if options is None:
            options = {}
        else:
            options = dict(options)

        # options is saved in Fit_bayes for reproducible experiments
        options.setdefault('n_draws', 2000)
        options.setdefault('n_chains', 4)
        options.setdefault('n_warmup', 1000)
        options.setdefault('init', 'unconstrained')
        options.setdefault('hpd', 0.95)

        n_draws = options.get('n_draws')
        n_chains = options.get('n_chains')
        n_warmup = options.get('n_warmup')
        init = options.get('init')
        hpd = options.get('hpd')

        dt, u, u1, y, *_ = self._prepare_data(df, inputs, outputs, None)

        # An additional initial position is required for the adaptation
        q0 = self._init_parameters(n_chains, init, hpd)
        V = lambda q: self._eval_log_posterior(q, dt, u, u1, y)
        dV = lambda q: self._eval_dlog_posterior(q, dt, u, u1, y)
        M = np.eye(len(self.ss.parameters.eta_free))

        dhmc = DynamicHMC(EuclideanHamiltonian(V=V, dV=dV, M=M))
        uchains, stats, options = dhmc.sample(q0, n_draws, n_chains, n_warmup, options)

        # Safety against duplication of parameter names. Do not use self.ss.parameters.names
        names = [k for i, k in enumerate(self.ss._names) if self.ss.parameters.free[i]]
        chains = self._array_to_dict(self._inv_transform_chains(uchains), names)

        return Fit_Bayes(chains, stats, options, n_warmup, self.ss.name)

    def prior_predictive(
        self,
        df: pd.DataFrame,
        inputs: Union[str, list] = None,
        n_sim: int = 1000,
        hpd: Real = None,
        n_cpu: int = -1,
    ) -> Tuple[dict, dict]:
        """Prior predictive distribution

        Args:
            df: Data
            inputs: Input name(s)
            n_sim: Number of prior predictive draw
            hpd: Highest Prior Density to draw sample from (True for unimodal distribution)
            n_cpu: Number of cpu

        Returns:
            2-elements tuple containing
                - **draws**: Parameter draws
                - **ppc**: Prior predictive samples
        """
        if self.ss.ny > 1:
            raise ValueError('Multiple outputs is not yet supported')

        if not isinstance(n_sim, int) or n_sim <= 0:
            raise TypeError('`n_sim` must be a positive integer')

        if not isinstance(n_cpu, int) or (n_cpu != -1 and n_cpu <= 0):
            raise TypeError('`n_cpu` must be a strictly positive integer or set to -1')

        dt, u, u1, *_ = self._prepare_data(df, inputs, None, None)

        ps = np.empty((n_sim, self.ss.parameters.n_par))
        ppd = np.empty((n_sim, df.index.shape[0], self.ss.ny))

        def prior_pd():
            self.ss.parameters.prior_init(hpd)
            return self.ss.parameters.theta_free, self._simulate_output(dt, u, u1, None)

        pbar = tqdm(range(n_sim), desc='Sampling')
        out = Parallel(n_jobs=n_cpu)(delayed(prior_pd)() for _ in pbar)

        for n in range(n_sim):
            ps[n, :] = out[n][0]
            ppd[n, :, :] = out[n][1]

        # Safety against duplication of parameter names. Do not use self.ss.parameters.names
        names = [k for i, k in enumerate(self.ss._names) if self.ss.parameters.free[i]]
        draws = {k: ps[:, i][np.newaxis, :] for i, k in enumerate(names)}
        ppc = {k.name: ppd[:, :, i][np.newaxis, :] for i, k in enumerate(self.ss.outputs)}

        return draws, ppc

    def posterior_predictive(
        self,
        trace: dict,
        df: pd.DataFrame,
        inputs: Union[str, list] = None,
        outputs: Union[str, list] = None,
        n_cpu: int = -1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Posterior predictive distribution

        Args:
            trace: Markov Chain traces, trace[key](chain, draw)
            df: Data
            inputs: Input name(s)
            outputs: Output name(s)
            n_cpu: Number of cpus

        Returns:
            2-elements tuple containing
                - **ym**: Mean of the predictive distribution
                - **ysd**: Standard deviation of the predictive distribution
        """
        if self.ss.ny > 1:
            raise ValueError('Multiple outputs is not yet supported')

        if not isinstance(n_cpu, int) or (n_cpu != -1 and n_cpu <= 0):
            raise TypeError('`n_cpu` must be a strictly positive integer or set to -1')

        dt, u, u1, y, *_ = self._prepare_data(df, inputs, outputs, None)

        chain, draw = list(trace.values())[0].shape
        n_draws = chain * draw
        samples = self._dict_to_array(trace, self.ss.parameters.names_free)

        def posterior_pd(index):
            self.ss.parameters.theta_free = samples[:, index]
            return self._estimate_output(dt, u, u1, y, None, None)

        pbar = tqdm(range(n_draws), desc='Sampling')
        out = Parallel(n_jobs=n_cpu)(delayed(posterior_pd)(i) for i in pbar)

        ym = np.empty((n_draws, dt.shape[0]))
        ysd = np.empty((n_draws, dt.shape[0]))
        for n in range(n_draws):
            ym[n], ysd[n] = np.squeeze(out[n])

        return ym, ysd

    def pointwise_log_likelihood(
        self,
        trace: dict,
        df: pd.DataFrame,
        outputs: Union[str, list],
        inputs: Union[str, list] = None,
        n_cpu: int = -1,
    ) -> dict:
        """Point-wise log-likelihood for Pareto Smoothing Importance Sampling

        Args:
            trace: Markov Chain traces, trace[key](chain, draw)
            df: Data
            outputs: Output name(s)
            inputs: Input name(s)
            cpu: Number of cpus

        Returns:
            Positive log-likelihood evaluated point-wise
        """
        if not isinstance(n_cpu, int) or (n_cpu != -1 and n_cpu <= 0):
            raise TypeError('`n_cpu` must be a strictly positive integer or set to -1')

        dt, u, u1, y, *_ = self._prepare_data(df, inputs, outputs, None)
        chain, draw = list(trace.values())[0].shape
        samples = self._dict_to_array(trace, self.ss.parameters.names_free)

        def eval_loglik_pw(index):
            self.ss.parameters.theta_free = samples[:, index]
            return self._eval_log_likelihood(dt, u, u1, y, pointwise=True)

        pbar = tqdm(range(chain * draw), desc='Sampling')
        out = Parallel(n_jobs=n_cpu)(delayed(eval_loglik_pw)(i) for i in pbar)

        pw_loglik = np.empty((chain * draw, y.shape[1]))
        for n in range(chain * draw):
            pw_loglik[n, :] = -out[n]

        return {'log_likelihood': pw_loglik.reshape(chain, draw, y.shape[1])}

    def posterior_state_distribution(
        self,
        trace: dict,
        df: pd.DataFrame,
        outputs: Union[str, list],
        inputs: Union[str, list] = None,
        smooth: bool = False,
        n_cpu: int = -1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Posterior state filtered/smoothed distribution

        Args:
            trace: Markov Chain traces, trace[key](chain, draw)
            df: Data
            outputs: Output name(s)
            inputs: Input name(s)
            smooth: Use Kalman smoother
            cpu: Number of cpus

        Returns:
            2-element tuple containing
                - **xm**: State filtered/smoothed distribution mean
                - **xsd**: State filtered/smoothed distribution standard deviation
        """
        if not isinstance(n_cpu, int) or (n_cpu != -1 and n_cpu <= 0):
            raise TypeError('`n_cpu` must be a strictly positive integer or set to -1')

        dt, u, u1, y, *_ = self._prepare_data(df, inputs, outputs, None)

        chain, draw = list(trace.values())[0].shape
        n_draws = chain * draw
        samples = self._dict_to_array(trace, self.ss.parameters.names_free)

        def state_pd(index):
            self.ss.parameters.theta_free = samples[:, index]
            x, P = self._estimate_states(dt, u, u1, y, smooth=smooth)
            return x, np.sqrt(P.diagonal(0, 1, 2))

        pbar = tqdm(range(n_draws), desc='Sampling')
        out = Parallel(n_jobs=n_cpu)(delayed(state_pd)(i) for i in pbar)

        xm = np.empty((n_draws, dt.shape[0], self.ss.nx))
        xsd = np.empty((n_draws, dt.shape[0], self.ss.nx))
        for n in range(n_draws):
            xm[n] = np.squeeze(out[n][0])
            xsd[n] = np.squeeze(out[n][1])

        return xm, xsd

    def _inv_transform_chains(self, eta_traces: np.ndarray) -> np.ndarray:
        """Transform samples to the constrained space

        Args:
            eta_traces: Unconstrained Markov chain traces of shape (n_chains, n_par, n_draws)

        Returns:
            theta_traces: Constrained Markov chain traces of shape (n_chains, n_par, n_draws)
        """
        if not isinstance(eta_traces, np.ndarray):
            raise TypeError('`eta_traces` must be a numpy array')

        sizes = eta_traces.shape
        if len(sizes) != 3:
            raise ValueError('`eta_traces` must be an array of shape (n_chains, n_par, n_draws)')
        theta_traces = np.zeros(sizes)

        for c in range(sizes[0]):
            for d in range(sizes[2]):
                self.ss.parameters.eta = eta_traces[c, :, d]
                theta_traces[c, :, d] = self.ss.parameters.theta_free

        return theta_traces

    def _array_to_dict(self, chains: np.ndarray, names: Union[str, list]) -> dict:
        """Convert 3d numpy array to dictionary

        Args:
            chains: Numpy array of shape (n_chains, n_par, n_draws)
            names: Parameter names list of size n_par

        Returns:
            Markov chains traces, [n_par keys](n_chains, n_draws)
        """
        if not isinstance(chains, np.ndarray):
            raise TypeError('`chains` must be a numpy array')

        if not isinstance(names, (str, list)):
            raise TypeError('`names` must be a string of a list of strings')

        if isinstance(names, str):
            names = [names]

        if len(names) != chains.shape[1]:
            raise ValueError('The length of `names` does not match `chains.shape[1]`')

        return {k: chains[:, i, :] for i, k in enumerate(names)}

    def _dict_to_array(self, chains: dict, names: Union[str, list]) -> np.ndarray:
        """Convert dict [n_par keys](n_chains, n_draws) to numpy array (n_par, chain * draw)

        Args:
            chains: Markov Chain traces, chains[key](n_chains, n_draws)
            names: Parameter names to convert into numpy array

        Returns:
            Markov Chain traces (n_par, chain * draw)
        """
        if not isinstance(chains, dict):
            raise TypeError('`chains` must be a dictionary')

        if not isinstance(names, (str, list)):
            raise TypeError('`names` must be a string of a list of strings')

        if isinstance(names, str):
            names = [names]

        return np.asarray([v.ravel() for k, v in chains.items() if k in names])
