"""Bayesian regressor"""
from typing import Tuple
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from .base import BaseRegressor
from ..state_estimator import BayesianFilter, Kalman_QR
from ..mcmc.hamiltonian import EuclideanHamiltonian
from ..mcmc.hmc import DynamicHMC, Fit_Bayes
from ..statespace.base import StateSpace
from ..utils.miscellaneous import dict_to_array, array_to_dict


class BayesRegressor(BaseRegressor):
    """Frequentist Regressor inference and prediction

    Args:
        ss: StateSpace instance
        bayesian_filter: BayesianFilter instance
        time_scale: Time series frequency, e.g. 's': seconds, 'D': days, etc.
            Works only for pandas.DataFrame with DateTime index
    """

    def __init__(
        self, ss: StateSpace, bayesian_filter: BayesianFilter = Kalman_QR, time_scale: str = 's'
    ):
        super().__init__(ss, bayesian_filter, time_scale, True, False)

    def fit(self, df: pd.DataFrame, outputs: list, inputs: list = None, options: dict = None):
        """Fit the model

        Args:
            df: Training data
            outputs: Outputs names
            inputs: Inputs names
            options:
                - **init** (str, default=`unconstrained`):
                    `unconstrained`: Uniform draws in the unconstrained space between [-2, 2]
                    `prior`: Uniform draws from the prior distribution
                    `fixed`: Use values set in position.theta
                - **prior_mass** (float, default=0.95):
                    Use **prior_mass** highest prior density for **init**=`prior`
                TODO

        Returns:
            BayesFit object
        """
        if options is None:
            options = {}
        else:
            options = dict(options)

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

        dt, u, u1, y, *_ = self._prepare_data(df, inputs, outputs, None, self.time_scale)

        # An additional initial position is required for the adaptation
        q0 = self._init_parameters(n_chains, init, hpd)
        V = lambda q: self._eval_log_posterior(q, dt, u, u1, y)
        dV = lambda q: self._eval_dlog_posterior(q, dt, u, u1, y)
        M = np.eye(len(self.ss.parameters.eta_free))

        dhmc = DynamicHMC(EuclideanHamiltonian(V=V, dV=dV, M=M))
        uchains, stats, options = dhmc.sample(q0, n_draws, n_chains, n_warmup, options)

        # Safety against duplication of parameter names. Do not use self.ss.parameters.names
        names = [k for i, k in enumerate(self.ss._names) if self.ss.parameters.free[i]]
        chains = array_to_dict(self._inv_transform_chains(uchains), names)

        return Fit_Bayes(chains, stats, options, n_warmup, self.ss.name)

    def _inv_transform_chains(self, eta_chains):
        """Transforma the samples from the unconstrained space to the constrained space"""

        sizes = eta_chains.shape
        if len(sizes) != 3:
            raise ValueError('`eta_chains` must be an array of shape (n_chains, n_par, n_draws)')
        n_chains, n_par, n_draws = eta_chains.shape
        theta_chains = np.zeros((n_chains, n_par, n_draws))

        for c in range(n_chains):
            for d in range(n_draws):
                self.ss.parameters.eta = eta_chains[c, :, d]
                theta_chains[c, :, d] = self.ss.parameters.theta_free

        return theta_chains

    def prior_predictive(
        self,
        df: pd.DataFrame,
        inputs: list = None,
        Nsim: int = 100,
        hpd: float = None,
        cpu: int = -1,
    ) -> Tuple[dict]:
        """Prior predictive distribution

        Args:
            df: Data
            inputs: Inputs names
            Nsim: Number of prior predictive draw
            hpd: Prior probability mass around the mode
            cpu: Number of cpu to use

        Returns:
            2-element tuple containing
                - **draws**: Sampled parameters from the prior distribution
                - **ppc**: Prior predictive distribution
        """
        if not isinstance(Nsim, int) or Nsim <= 0:
            raise TypeError('`Nsim` must be a positive integer')

        if not isinstance(cpu, int):
            raise TypeError('`cpu` must be an integer > 0 or equal to -1')

        dt, u, u1, *_ = self._prepare_data(df, inputs, None, None, self.time_scale)

        ps = np.empty((Nsim, len(self.ss.parameters.theta_free)))
        ppd = np.empty((Nsim, df.index.shape[0], self.ss.ny))
        divergence = np.empty((Nsim))

        def prior_pd():
            self.ss.parameters.prior_init(hpd)
            try:
                y = self._simulate_output(dt, u, u1, None)
                div = 0
            except (RuntimeError, RuntimeWarning):
                div = 1
                y = np.zeros((df.index.shape[0], self.ss.ny))
            return self.ss.parameters.theta_free, y, div

        pbar = tqdm(range(Nsim), desc='Prior predictive')
        out = Parallel(n_jobs=cpu)(delayed(prior_pd)() for _ in pbar)

        for n in range(Nsim):
            ps[n, :] = out[n][0]
            ppd[n, :, :] = out[n][1]
            divergence[n] = out[n][2]

        # Safety against duplication of parameter names. Do not use self.ss.parameters.names
        names = [k for i, k in enumerate(self.ss._names) if self.ss.parameters.free[i]]
        draws = {k: ps[:, i][np.newaxis, :] for i, k in enumerate(names)}
        ppc = {k.name: ppd[:, :, i][np.newaxis, :] for i, k in enumerate(self.ss.outputs)}

        return draws, ppc, divergence

    def posterior_predictive(
        self,
        trace: dict,
        df: pd.DataFrame,
        inputs: list = None,
        x0: np.ndarray = None,
        cpu: int = -1,
    ) -> dict:
        """Posterior predictive distribution

        Args:
            trace: Dictionary of Markov Chain traces, trace[key](chain, draw)
            df: Data
            inputs: Inputs names
            x0: Initial state mean (chain * draw, Nx, 1)
            cpu: Number of cpu to use

        Returns:
            Posterior predictive distribution
        """
        if not isinstance(cpu, int):
            raise TypeError('`cpu` must be an integer > 0 or equal to -1')

        dt, u, u1, *_ = self._prepare_data(df, inputs, None, None, self.time_scale)

        chain, draw = list(trace.values())[0].shape
        T = df.index.shape[0]

        trace = dict_to_array(trace, self.ss.parameters.names_free)
        draw_stack = chain * draw

        ppd = np.empty((draw_stack, T, self.ss.ny))

        def posterior_pd(index):
            self.ss.parameters.theta_free = trace[:, index]
            if x0 is not None:
                x0s = x0[index, :, :]
            else:
                x0s = None
            return self._simulate_output(dt, u, u1, x0s)

        pbar = tqdm(range(draw_stack), desc='Posterior predictive')
        out = Parallel(n_jobs=cpu)(delayed(posterior_pd)(i) for i in pbar)

        for n in range(draw_stack):
            ppd[n, :, :] = out[n]

        return {k.name: ppd[:, :, i][np.newaxis, :] for i, k in enumerate(self.ss.outputs)}

    def pointwise_log_likelihood(
        self,
        trace: np.ndarray,
        df: pd.DataFrame,
        inputs: list = None,
        outputs: list = None,
        cpu: int = -1,
    ) -> dict:
        """"Compute pointwise log-likelihood from the posterior distribution

        Args:
            trace: Dictionary of Markov Chain traces, trace[key](chain, draw)
            df: Data
            inputs: Inputs names
            outputs: Outputs names
            cpu: Number of cpu to use

        Returns:
            Positive log-likelihood evaluated point-wise

        Notes:
            The output can be transformed to an arviz.InferenceData with
            arviz.from_dict(sample_stats={log_likelihood: `returns`}) which
            allows to use arviz.loo and arviz.waic.
            See https://arviz-devs.github.io/arviz/
        """
        if not isinstance(cpu, int):
            raise TypeError('`cpu` must be an integer > 0 or equal to -1')

        dt, u, u1, y, *_ = self._prepare_data(df, inputs, outputs, None, self.time_scale)
        chain, draw = list(trace.values())[0].shape

        names = [k for k in trace.keys() if k != 'lp_']
        trace_arr = dict_to_array(trace, names)

        # point-wise log-likelihood
        pw_loglik = np.empty((chain * draw, y.shape[1]))

        def eval_loglik_pw(index):
            self.ss.parameters.theta_free = trace_arr[:, index]
            return self._eval_log_likelihood(dt, u, u1, y, pointwise=True)

        pbar = tqdm(range(chain * draw), desc='Point-wise log-likelihood')
        out = Parallel(n_jobs=cpu)(delayed(eval_loglik_pw)(i) for i in pbar)

        for n in range(chain * draw):
            pw_loglik[n, :] = -out[n]

        return {'log_likelihood': pw_loglik.reshape(chain, draw, y.shape[1])}

    def posterior_state_distribution(
        self,
        trace: dict,
        df: pd.DataFrame,
        inputs: list = None,
        outputs: list = None,
        smooth: bool = False,
        cpu: int = -1,
    ) -> Tuple[np.ndarray]:
        """Posterior state filtered/smoothed distribution

        Args:
            trace: Dictionary of Markov Chain traces, trace[key](chain, draw)
            df: Data
            inputs: Inputs names
            outputs: Outputs names
            smooth: Use smoother
            cpu: Number of cpu to use

        Returns:
            2-element tuple containing
                - **x_stack**: State filtered/smoothed distribution mean
                - **P_stack**: State filtered/smoothed distribution deviation

        TODO
            add x0, P0
        """
        dt, u, u1, y, *_ = self._prepare_data(df, inputs, outputs, None, self.time_scale)

        chain, draw = list(trace.values())[0].shape
        T = df.index.shape[0]

        names = [k for k in trace.keys() if k != 'lp_']
        trace = dict_to_array(trace, names)
        draw_stack = chain * draw

        x_stack = np.empty((draw_stack, T, self.ss.nx, 1))
        P_stack = np.empty((draw_stack, T, self.ss.nx, self.ss.nx))

        def state_pd(index):
            self.ss.parameters.theta_free = trace[:, index]
            return self._estimate_states(dt, u, u1, y, smooth=smooth)

        if smooth:
            desc = 'state smoothed distribution:'
        else:
            desc = 'state filtered distribution:'

        pbar = tqdm(range(draw_stack), desc=desc)
        out = Parallel(n_jobs=cpu)(delayed(state_pd)(i) for i in pbar)

        for n in range(draw_stack):
            x_stack[n, :, :, :] = out[n][0]
            P_stack[n, :, :, :] = out[n][1]

        return x_stack, P_stack
