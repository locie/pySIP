"""Standalone file for Dynamic Hamiltonian Monte Carlo"""
from typing import Tuple, Iterable, Union, NamedTuple
from numbers import Real
from collections import namedtuple, defaultdict
from copy import deepcopy
from datetime import datetime
from joblib import Parallel, delayed
import tqdm.auto as tqdm
import numpy as np
import pandas as pd
from scipy.linalg import LinAlgError

from .hamiltonian import EuclideanHamiltonian
from .adaptation import DualAveraging, WindowedAdaptation, WelfordCovEstimator, CovAdaptation
from ..utils.math import log_sum_exp, cholesky_inverse

State = namedtuple('State', 'q, p V dV dK H')


class IntegrationError(RuntimeError):
    """Leapfrog integration error"""

    pass


class DynamicHMC:
    """Dynamic Hamiltonian Monte Carlo Sampler with Multinomial sampling

    Args:
        hamiltonian: Hamiltonian system with Euclidean-Gaussian kinetic energies

    References:
    Betancourt, M., 2017. A conceptual introduction to Hamiltonian Monte Carlo.
    arXiv preprint arXiv:1701.02434.
    """

    def __init__(self, hamiltonian: EuclideanHamiltonian):
        self._hamiltonian = hamiltonian
        self._dH_max = None
        self._max_tree_depth = None
        self._accp_target = None
        self._n_warmup = None

    def sample(
        self,
        q: np.ndarray,
        n_draws: int = 2000,
        n_chains: int = 4,
        n_warmup: int = 1000,
        options: dict = None,
    ) -> Tuple[np.ndarray, dict, pd.DataFrame]:
        """Draw samples from the target distribution

        Args:
            q: Initial position variables (e.g. initial parameters)
            n_draws: Number of samples
            n_chains: Number of Markov chains
            n_warmup: Number of iterations to use for adapting the step size and the mass matrix
            options:
                - **stepsize** (float, default=0.25 / n_par**0.25)
                    Step-size of the leapfrog integrator
                - **max_tree_depth** (int, default=10)
                    Maximum tree depth
                - **dH_max** (float, default=1000)
                    Maximum energy change allowed in a trajectory. Larger
                    deviations are considered as diverging transitions.
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

        Returns:
            chains: Markov chain traces
            stats: Hamiltonian transition statistics
            df: Hamiltonian sampler diagnostic
        """

        if not isinstance(q, np.ndarray):
            raise TypeError('The initial position variable `q` must be a numpy array')

        if not isinstance(n_draws, int) and n_draws <= 0:
            raise TypeError('The number of samples `n_draws` must be an integer greater than 0')

        if not isinstance(n_chains, int) and n_chains <= 0:
            raise TypeError('The number of chains `n_chains` must be an integer greater than 0')

        if not isinstance(n_warmup, int):
            raise TypeError('`n_warmup` must be an integer')

        if options is not None and not isinstance(options, dict):
            raise TypeError('`options must be a dictionary`')

        if len(q.shape) == 1:
            if n_chains != 1:
                raise ValueError(f'The position array `q` must be an 1d array')
        else:
            if q.shape[1] != n_chains:
                raise ValueError(f'The position array `q` must have {n_chains} columns')

        if n_warmup < 1000:
            raise ValueError(f'`n_warmup` must be at least 1000 iterations')
        self._n_warmup = n_warmup

        # Check and  unpack options
        if options is None:
            options = {}
        else:
            options = dict(options)

        # Set default values and return the dict of options
        options.setdefault('stepsize', 0.25 / q.shape[0] ** 0.25)
        options.setdefault('dH_max', 1000)
        options.setdefault('max_tree_depth', 10)
        options.setdefault('n_cpu', -1)

        stepsize = options.get('stepsize')
        if not isinstance(stepsize, Real) or stepsize <= 0:
            raise TypeError('`stepsize` must be a strictly positive real number')

        self._dH_max = options.get('dH_max')
        if not isinstance(self._dH_max, Real) or self._dH_max <= 0:
            raise TypeError('`dH_max` must be a strictly positive real number')

        self._max_tree_depth = options.get('max_tree_depth')
        if not isinstance(self._max_tree_depth, int) or self._max_tree_depth <= 0:
            raise TypeError('`max_tree_depth` must a strictly positive integer')

        n_cpu = options.get('n_cpu')
        if not isinstance(n_cpu, int) or (n_cpu != -1 and n_cpu <= 0):
            raise TypeError('`n_cpu` must be a strictly positive integer or set to -1')

        # The dictionary of options is saved in the returned fit object
        options.setdefault('accp_target', 0.8)
        self._accp_target = options.get('accp_target')
        options.setdefault('t0', 10)
        t0 = options.get('t0')
        options.setdefault('gamma', 0.05)
        gamma = options.get('gamma')
        options.setdefault('kappa', 0.75)
        kappa = options.get('kappa')
        options.setdefault('mu', np.log(10.0 * stepsize))
        mu = options.get('mu')
        options.setdefault('init_buffer', 75)
        init_buffer = options.get('init_buffer')
        options.setdefault('term_buffer', 150)
        term_buffer = options.get('term_buffer')
        options.setdefault('window', 25)
        window = options.get('window')

        # Adaptation methods
        step_adapter = DualAveraging(self._accp_target, t0, gamma, kappa, mu)
        estimator = WelfordCovEstimator(q.shape[0], False)
        schedule = WindowedAdaptation(self._n_warmup, init_buffer, term_buffer, window)
        cov_adapter = CovAdaptation(estimator, schedule)

        # Sampling
        if n_chains == 1:
            pbar = tqdm.trange(n_draws, desc='Sampling', dynamic_ncols=True)
            samples, _stats = self._sample_chain(q, pbar, stepsize, step_adapter, cov_adapter)
            # The number of chains must be a dimension
            samples = samples[None, :, :]
            stats = {k: np.asarray(v)[None, :] for k, v in _stats.items()}
        else:
            pbar = [
                tqdm.trange(n_draws, desc=f'Chain {n}', dynamic_ncols=True, position=n)
                for n in range(n_chains)
            ]

            job = Parallel(n_jobs=n_cpu)(
                delayed(self._sample_chain)(q[:, n], pbar[n], stepsize, step_adapter, cov_adapter)
                for n in range(n_chains)
            )

            # unpack parallel chains
            keys = job[0][1].keys()
            stats = {k: np.zeros((n_chains, n_draws)) for k in keys}
            samples = np.zeros((n_chains, q.shape[0], n_draws))
            for i, (c, d) in enumerate(job):
                samples[i, :, :] = c
                for k, v in d.items():
                    stats[k][i, :] = v

        return samples, stats, options

    def _sample_chain(
        self,
        q: np.ndarray,
        pbar: tqdm.trange,
        stepsize: float,
        step_adapter: DualAveraging,
        cov_adapter: CovAdaptation,
    ) -> Tuple[np.ndarray, dict]:
        """Sample a chain

        Args:
            q: Initial position variables
            pbar: Progress bar
            stepsize: Initial step-size of the leapfrog integrator
            step_adapter: Dual averaging algorithm for learning the step-size
            cov_adapter: Windowed Welford covariance estimator

        Returns:
            chain: Sampled chain
            stats: Transition statistics
        """

        samples = np.zeros((q.shape[0], len(pbar.iterable)))
        stats = defaultdict(list)
        # stepsize = self._find_reasonable_stepsize(q, stepsize)

        for i in pbar:
            # HMC step
            q, s = self._hmc_step(q, stepsize)

            # Save samples and statistics
            samples[:, i] = q
            for key, value in s.items():
                stats[key].append(value)

            # do adaptation
            if i <= self._n_warmup - 1:
                stepsize = step_adapter.learn(s['accept_prob'])
                update, cov = cov_adapter.learn(q)
                if update:
                    self._hamiltonian.cholM = cholesky_inverse(cov)
                    stepsize = step_adapter.adapted_step_size
                    # stepsize = self._find_reasonable_stepsize(q, stepsize)
                    # step_adapter.restart(mu=np.log(10.0 * stepsize))

                # End of adaptation
                if i == self._n_warmup - 1:
                    stepsize = step_adapter.adapted_step_size

        return samples, stats

    def _hamiltonian_state(self, q: np.ndarray, p: np.ndarray) -> NamedTuple:
        """Compute Hamiltonian state at a point in phase space

        Args:
            q: Position in phase space
            p: Momentum in phase space

        Return:
            Hamiltonian state
        """

        V, dV = self._hamiltonian.dV(q)
        H = V + self._hamiltonian.K(p)
        dK = self._hamiltonian.dK(p)

        return State(q, p, V, dV, dK, H)

    def _hmc_step(self, q: np.ndarray, stepsize: float) -> Tuple[np.ndarray, dict]:
        """Dynamic Hamiltonian Monte Carlo step

        Args:
            q: Position in phase space
            stepsize: Step-size of the leapfrog integrator

        Returns:
            state_n.q: Next position
            stats: Statistic from the Hamiltonian transition
        """

        # Draw momentum
        p = self._hamiltonian.sample_p()

        # Compute Hamiltonian state
        state = self._hamiltonian_state(q, p)

        # Hamiltonian transition
        state_n, stats = self._transition(state, stepsize)

        return state_n.q, stats

    def _transition(self, state: NamedTuple, stepsize: float) -> Tuple[NamedTuple, dict]:
        """Dynamic Hamiltonian Monte Carlo transition with multinomial sampling

        Args:
            state: Hamiltonian state at the current point in phase space
            stepsize: Step-size of the leapfrog integrator

        Returns:
            state_n: Next Hamiltonian state
            stats: Trajectory statistics
                - **accept_prob**: Average acceptance probability accross trajectory
                - **leapfrog_steps**: Number of leapfrog steps
                - **diverging**: Flag for diverging transition
                - **energy**: Hamiltonian at the next state
                - **tree_depth**: Depth of the binary tree
                - **max_tree_depth**: Flag for maximum integration steps
                - **stepsize**: Step-size of the leapfrog integrator
                - **potential**: Potential energy (negative log-posterior)
        """

        state_n, state_l, state_r = deepcopy(state), deepcopy(state), deepcopy(state)

        H0 = state.H.copy()
        sum_p = state.p.copy()
        sum_w = 0.0
        stats = {
            'leapfrog_steps': 0,
            'accept_prob': 0.0,
            'max_tree_depth': False,
            'diverging': False,
            'potential': state_n.V,
            'stepsize': stepsize,
        }

        for depth in range(self._max_tree_depth):
            sum_p_s = np.zeros(state.p.shape)
            sum_w_s = -np.inf

            if np.random.uniform() > 0.5:
                terminate_s, _, state_r, state_p, sum_w_s = self._build_tree(
                    depth, 1, state_r, stepsize, H0, sum_p_s, sum_w_s, stats
                )
            else:
                terminate_s, _, state_l, state_p, sum_w_s = self._build_tree(
                    depth, -1, state_l, stepsize, H0, sum_p_s, sum_w_s, stats
                )

            if terminate_s:
                break

            if sum_w_s > sum_w:
                state_n = state_p
            else:
                if np.random.uniform() < np.exp(sum_w_s - sum_w):
                    state_n = state_p

            sum_w = log_sum_exp(sum_w, sum_w_s)
            sum_p += sum_p_s

            if np.dot(state_l.dK, sum_p) < 0 or np.dot(state_r.dK, sum_p) < 0:
                break
        else:
            stats['max_tree_depth'] = True

        if stats['leapfrog_steps'] > 0:
            stats['accept_prob'] /= stats['leapfrog_steps']

        stats['energy'] = state_n.H
        stats['potential'] = state_n.V
        stats['tree_depth'] = depth

        return state_n, stats

    def _build_tree(
        self,
        depth: int,
        direction: int,
        state: NamedTuple,
        stepsize: float,
        H0: float,
        sum_p: float,
        sum_w: float,
        stats: dict,
    ) -> Tuple[bool, NamedTuple, NamedTuple, NamedTuple, float]:
        """Recursively build the binary tree

        Args:
            depth: Depth of the desired subtree
            direction: direction in time to build subtree [-1 or 1]
            state: State proposed from the subtree
            stepsize: Step-size of the leapfrog integrator
            H0: Hamiltonian of the initial state
            sum_p: Summed momentum accross trajectory
            sum_w: Summed weight accross trajectory
            stats: Trajectory statistics
                - **accept_prob**: Summed acceptance probability accross trajectory
                - **leapfrog_steps**: Summed number of leapfrog integrations
                - **diverging**: Flag for diverging transition

        Returns:
            5-elements tuple with
                stop: Flag indicating a divergent transition or a U-turn
                state_inner: State of the inner subtree
                state_outer: State of the outer subtree
                state_proposed: State of the proposed subtree
                sum_w: Summed weight accross trajectory
        """

        # Base case recursion
        if depth == 0:
            try:
                state = self._leapfrog_step(state, direction, stepsize)

                dH = state.H - H0
                if np.isnan(dH):
                    dH = np.inf

                if dH > self._dH_max:
                    stats['diverging'] = True
                    state = None
                    terminate = True
                else:
                    sum_p += state.p
                    sum_w = log_sum_exp(sum_w, -dH)
                    if -dH > 0.0:
                        stats['accept_prob'] += 1
                    else:
                        stats['accept_prob'] += np.exp(-dH)
                    stats['leapfrog_steps'] += 1
                    terminate = False

            except (LinAlgError, RuntimeError):
                terminate = True
                state = None

            return terminate, state, state, state, sum_w

        # build the inner subtree
        sum_p_i = np.zeros(state.p.shape[0])
        sum_w_i = -np.inf

        terminate_i, state_i, state, state_pi, sum_w_i = self._build_tree(
            depth - 1, direction, state, stepsize, H0, sum_p_i, sum_w_i, stats
        )

        if terminate_i:
            return True, None, None, None, None

        # build the outer subtree
        sum_p_o = np.zeros(state.p.shape[0])
        sum_w_o = -np.inf

        terminate_o, _, state_o, state_po, sum_w_o = self._build_tree(
            depth - 1, direction, state, stepsize, H0, sum_p_o, sum_w_o, stats
        )

        if terminate_o:
            return True, None, None, None, None

        # Sample from combined inner and outer subtree
        sum_w_s = log_sum_exp(sum_w_i, sum_w_o)
        sum_w = log_sum_exp(sum_w, sum_w_s)

        if sum_w_o > sum_w_s:
            state_p = state_po
        else:
            if np.random.uniform() < np.exp(sum_w_o - sum_w_s):
                state_p = state_po
            else:
                state_p = state_pi

        sum_p_s = sum_p_i + sum_p_o
        sum_p += sum_p_s

        stop_s = np.dot(state_i.dK, sum_p_s) < 0 or np.dot(state_o.dK, sum_p_s) < 0

        return stop_s, state_i, state_o, state_p, sum_w

    def _leapfrog_step(self, state: NamedTuple, direction: int, stepsize: float) -> NamedTuple:
        """Explicit leapfrog integrator for separable Hamiltonian systems

        Args:
            state: State of the Hamiltonian at the current point in phase space
            direction: Time direction for integration, e.g forward=1, backward=-1
            stepsize: Step-size of the leapfrog integrator

        Returns:
            State of the Hamiltonian a step further in the trajectory
        """
        q = deepcopy(state.q)
        p = deepcopy(state.p)

        dt = direction * stepsize
        p -= 0.5 * dt * state.dV
        q += dt * self._hamiltonian.dK(p)
        V, dV = self._hamiltonian.dV(q)
        p -= 0.5 * dt * dV

        # Required for u-turn criterion and building the binary tree
        dK = self._hamiltonian.dK(p)
        H = V + self._hamiltonian.K(p)

        return State(q, p, V, dV, dK, H)

    def _find_reasonable_stepsize(self, q: np.ndarray, stepsize: float) -> float:
        """Find a reasonable step-size by heuristic tuning

        Args:
            q: Position variable in phase space
            stepsize: Initial step-size

        Returns:
            stepsize: Heuristic adaptation of the step-size
        """
        if stepsize < 1e-16 or stepsize > 1e7:
            return

        log_target = np.log(self._accp_target)
        V, dV = self._hamiltonian.dV(q)
        p = self._hamiltonian.sample_p()
        H = V + self._hamiltonian.K(p)
        state0 = State(q, p, V, dV, None, H)
        state1 = self._leapfrog_step(state0, 1, stepsize)
        dH = state0.H - state1.H
        if np.isnan(dH):
            dH = -np.inf
        direction = 2 * (dH > log_target) - 1

        while True:
            p = self._hamiltonian.sample_p()
            H = V + self._hamiltonian.K(p)
            state0 = State(q, p, V, dV, None, H)
            state1 = self._leapfrog_step(state0, direction, stepsize)
            dH = state0.H - state1.H
            if np.isnan(dH):
                dH = -np.inf

            if direction == 1 and not dH > log_target:
                break
            elif direction == -1 and not dH < log_target:
                break
            else:
                stepsize *= 2.0 ** direction

            if stepsize < 1e-16:
                raise RuntimeError('No acceptably small step-size could be found')
            if stepsize > 1e7:
                raise RuntimeError('The step-size diverged to an unacceptable large value')

        return stepsize


class Fit_Bayes:
    """Fit summary returned by DynamicHMC

    Args:
        chains: Markov Chains Traces
        stats: Hamiltonian transition statistics
        options: Sampler options
        n_warmup: Number of iterations used as warmup
        model: Model name
    """

    def __init__(self, chains: dict, stats: dict, options: dict, n_warmup: int, model: str = None):

        if not isinstance(chains, (dict, np.ndarray)):
            raise TypeError('`chains` must be a dictionary or a numpy array')

        if isinstance(chains, np.ndarray):
            if len(chains.shape) < 3:
                raise TypeError('`chains` must be 3-d numpy array')
            self._chains = {'x' + str(i): chains[:, i, :] for i in range(chains.shape[1])}
        else:
            self._chains = chains

        if not isinstance(stats, dict):
            raise TypeError('`stats` must be a dictionary')

        if not isinstance(options, dict):
            raise TypeError('`options` must be a dictionary')

        if model is not None and not isinstance(model, str):
            raise TypeError('`model` must be a string')

        self._stats = stats
        self._options = options
        self.n_warmup = n_warmup
        self._n_chains, self._n_draws = stats['tree_depth'].shape
        self._chains['lp_'] = self._stats.pop('potential')
        self._datetime = datetime.now()
        if model is None:
            self._model = 'unknown'
        else:
            self._model = model

    def __repr__(self):
        """Return information of the fitting"""
        return (
            f'\nmodel: {self._model}\nnumber of chains: {self._n_chains}\n'
            f'number of draws: {self._n_draws}\nnumber of draws for warm-up: {self._n_warmup}'
            f'\ndate: {self._datetime.strftime("%d/%m/%Y")}\n'
            f'time: {self._datetime.strftime("%H:%M:%S")}\n'
        )

    @property
    def n_warmup(self) -> int:
        """Number of iterations used as warmup"""
        return self._n_warmup

    @n_warmup.setter
    def n_warmup(self, x):
        """Set the number of iterations for the warm-up"""
        if not isinstance(x, int) or x < 0:
            raise ValueError('`n_warmup` must be a strictly positive integer')
        self._n_warmup = x

    @property
    def posterior(self) -> dict:
        '''Return the Markov chain traces with without warm-up'''
        return {k: v[:, self._n_warmup :] for k, v in self._chains.items()}

    @property
    def stats(self) -> dict:
        '''Return the Markov chain traces with without warm-up'''
        return {k: v[:, self._n_warmup :] for k, v in self._stats.items()}

    def get_options(self, keys: Union[str, list] = None) -> dict:
        """Return the dictionary of options or only the options given in `keys`

        Args:
            keys: List of options (keys)

        Returns:
            d: Dictionary of options
        """
        if keys is not None and not isinstance(keys, (list, str)):
            raise TypeError('`keys` must be a string or a list of strings')

        if isinstance(keys, str):
            keys = [keys]

        if keys is None:
            d = self._options
        else:
            d = {}
            for k in keys:
                if k not in self._options.keys():
                    raise ValueError('{k} is not an available option')
                d[k] = self._options[k]
        return d

    @property
    def diagnostic(self) -> pd.DataFrame:
        """Sampler diagnostic

        Returns:
            df: Sampler diagnostic

        Notes:
            EBFMI (Energy Bayesian Fraction of Missing Information) values below 0.3 indicates
            that momentum resampling will ineffciently explore the energy level sets.
        """
        df = pd.DataFrame(
            index=['Chain ' + str(i) for i in range(self._n_chains)],
            data={
                'ebfmi': np.square(np.diff(self.stats['energy'], axis=1)).mean(axis=1)
                / self.stats['energy'].var(axis=1),
                'mean accept_prob': self.stats['accept_prob'].mean(axis=1),
                'mean tree_depth': self.stats['tree_depth'].mean(axis=1),
                'mean leapfrog_steps': self.stats['leapfrog_steps'].mean(axis=1),
                'sum diverging': self.stats['diverging'].sum(axis=1).astype('int'),
                'sum max_tree_depth': self.stats['max_tree_depth'].sum(axis=1).astype('int'),
                'stepsize': self.stats['stepsize'].mean(axis=1),
            },
        )
        return df
