"""Module for Hamiltonian Monte Carlo Adaptation"""
import numpy as np

__ALL__ = [
    "DualAveraging",
    "WelfordCovEstimator",
    "WindowedAdaptation",
    "CovAdaptation",
]


class DualAveraging:
    """Nesterov Primal Dual Averaging for step-size adapation"""

    def __init__(
        self,
        acc_prob_target: float = 0.8,
        t0: float = 10.0,
        gamma: float = 0.05,
        kappa: float = 0.75,
        mu: float = 0.5,
    ):

        if not isinstance(acc_prob_target, float):
            raise TypeError("`acc_prob_target must a float`")
        if acc_prob_target < 0.6:
            raise ValueError("`acc_prob_target` must be >= 0.6")

        if not isinstance(t0, (int, float)):
            raise TypeError("`t0 must an integer or a float`")
        if t0 < 0:
            raise ValueError("`t0` must be > 0")

        if not isinstance(gamma, (int, float)):
            raise TypeError("`gamma must an integer or a float`")
        if gamma <= 0:
            raise ValueError("`gamma` must be > 0")

        if not isinstance(kappa, float):
            raise TypeError("`kappa must a float`")
        if not 0.0 < kappa < 1.0:
            raise ValueError("`kappa` must be between ]0, 1[")

        self._target = acc_prob_target
        self._t0 = t0
        self._gamma = gamma
        self._kappa = kappa
        self.restart(mu)

    def learn(self, acc_prob: float):
        """Adapt the step-size to match the targeted acceptance probability

        Args:
            acc_prob: Mean acceptance probability from the HMC transition
        """
        if acc_prob > 1:
            acc_prob = 1
        self._counter += 1
        w = 1.0 / (self._counter + self._t0)
        self._hbar = (1.0 - w) * self._hbar + w * (self._target - acc_prob)
        log_e = self._mu - self._hbar * np.sqrt(self._counter) / self._gamma
        z = self._counter ** (-self._kappa)
        self._log_ebar = (1.0 - z) * self._log_ebar + z * log_e

        return np.exp(log_e)

    @property
    def adapted_step_size(self):
        """Return the estimated step-size"""
        return np.exp(self._log_ebar)

    def restart(self, mu: float = None):
        """Reset the adaptation"""
        self._counter = 0
        self._hbar = 0.0
        self._log_ebar = 0.0
        if mu is not None:
            if not isinstance(mu, (int, float)):
                raise TypeError("`mu must an integer or a float`")
            self._mu = mu


class WelfordCovEstimator:
    """Welford's accumulator for sequentially estimating the sample covariance matrix.
    This method is used for estimating the inverse mass matrix in the Hamiltonian
    Monte Carlo sampler.

    Args:
        dimension: Number of dimension
        dense: Estimate the full covariance matrix. By default, only the diagonal
          elements are estimated
        shrinkage: Shrink the estimate towards unity
    """

    def __init__(self, dimension: int = 1, dense: bool = True, shrinkage: bool = True):
        """Initialize the estimator with the samples dimension"""
        if not isinstance(dense, bool):
            raise TypeError("`dense` must be a boolean")

        self._dim = dimension
        self._dense = dense
        self._shrinkage = shrinkage
        self.restart()

    def restart(self):
        """Restart the sample estimator"""
        self._n = 0
        self._mean = np.zeros(self._dim)
        if self._dense:
            self._m2 = np.zeros((self._dim, self._dim))
        else:
            self._m2 = np.zeros(self._dim)

    @property
    def n_sample(self):
        """Number of samples accumulated"""
        return self._n

    @property
    def sample_mean(self):
        """Sample mean"""
        return self._mean

    def add_sample(self, sample: np.ndarray):
        """Update the estimator with a new sample

        Args:
            sample: New sample
        """
        self._n += 1
        pre_diff = sample - self._mean
        self._mean = self._mean + pre_diff / self._n
        post_diff = sample - self._mean
        if self._dense:
            self._m2 = self._m2 + np.outer(post_diff, pre_diff)
        else:
            self._m2 = self._m2 + post_diff * pre_diff

    def get_covariance(self):
        """Sample covariance matrix"""
        cov = self._m2 / (self._n - 1)
        if self._shrinkage:
            scaled_cov = cov * (self._n / (self._n + 5))
            shrinkage = 1e-3 * (5 / (self._n + 5))
            if self._dense:
                cov = scaled_cov + shrinkage * np.identity(self._mean.shape[0])
            else:
                cov = scaled_cov + shrinkage
        return cov


class WindowedAdaptation:
    """Stan window adaptation scheme

    Args:
        n_adapt: Length of the adaptation
        init_buffer: Width of initial fast adaptation interval
        term_buffer: Width of final fast adaptation interval
        window: Initial width of slow adaptation interval
    """

    def __init__(
        self,
        n_adapt: int = 1000,
        init_buffer: int = 75,
        term_buffer: int = 50,
        window: int = 25,
    ):

        if not isinstance(n_adapt, int) and n_adapt <= 0:
            raise TypeError("`n_adapt` must be a positive integer")

        if not isinstance(init_buffer, int) and init_buffer > 0:
            raise TypeError("`init_buffer` must be a positive integer")

        if not isinstance(term_buffer, int) and term_buffer > 0:
            raise TypeError("`term_buffer` must be a positive integer")

        if not isinstance(window, int) and window > 0:
            raise TypeError("`window` must be a positive integer")

        if init_buffer + window + term_buffer > n_adapt:
            raise ValueError(
                f"There are not enougth adaptation iterations `n_adapt` to"
                f" fit the three stages of adaptation as currently, e.g."
                f" `init_buffer` + `window` + `term_buffer` ="
                f" {init_buffer + window + term_buffer}"
            )
        self._n_adapt = n_adapt
        self._init_buffer = init_buffer
        self._term_buffer = term_buffer
        self._base_window = window
        self._last_window = self._n_adapt - self._term_buffer - 1
        self._next_window = None
        self.restart()

    def restart(self):
        """Restart the windows"""
        self._counter = 0
        self._window_size = self._base_window
        self._next_window = self._init_buffer + self._window_size - 1

    def increment_counter(self):
        """Increment the window counter to keep track of the window schedule"""
        self._counter += 1

    @property
    def adaptation_window(self):
        """Check if we are in an adaptation window"""
        return (
            (self._counter >= self._init_buffer)
            & (self._counter < self._n_adapt - self._term_buffer)
            & (self._counter != self._n_adapt)
        )

    @property
    def end_adaptation_window(self):
        """Check if we are at the end of the current window"""
        return (self._counter == self._next_window) & (self._counter != self._n_adapt)

    def compute_next_window(self):
        """Compute the next slow adaptation window by doubling the size of the previous
        one"""
        if self._next_window == self._last_window:
            return

        self._window_size *= 2
        self._next_window = self._counter + self._window_size

        if self._next_window == self._last_window:
            return

        next_window_boundary = self._next_window + 2 * self._window_size

        if next_window_boundary >= self._n_adapt - self._term_buffer:
            self._next_window = self._last_window


class CovAdaptation:
    """Windowed (co)variance adaptation

    Args:
        estimator: A sample (co)variance estimator
        schedule: Windowed schedule
    """

    def __init__(self, estimator: WelfordCovEstimator, schedule: WindowedAdaptation):
        self._estimator = estimator
        self._schedule = schedule

    def learn(self, q: np.array):
        """Compute the sample covariance matrix at the end of the current window"""
        if self._schedule.adaptation_window:
            self._estimator.add_sample(q)

        if self._schedule.end_adaptation_window:
            self._schedule.compute_next_window()
            cov = self._estimator.get_covariance()
            self._estimator.restart()
            self._schedule.increment_counter()

            return True, cov

        self._schedule.increment_counter()
        return False, ()
