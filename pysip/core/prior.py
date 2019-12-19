import numpy as np
from scipy import special, stats, optimize
from typing import Tuple, Callable
from numbers import Real


__ALL__ = ['Normal', 'Gamma', 'Beta', 'InverseGamma', 'LogNormal']


class Prior:
    """Prior class template

    The notation from the following book are used throughout the module:

    Appendix A - Gelman, A., Stern, H.S., Carlin, J.B., Dunson, D.B., Vehtari, A.
    and Rubin, D.B., 2013. Bayesian data analysis. Chapman and Hall/CRC.
    """

    def mean(self) -> float:
        """mean"""
        raise NotImplementedError

    def mode(self) -> float:
        """mode"""
        raise NotImplementedError

    def variance(self) -> float:
        """variance"""
        raise NotImplementedError

    def pdf(self, x: float) -> float:
        """Evaluate the probability density function

        Args:
            x: Quantiles
        """
        return np.exp(self.log_pdf(x))

    def log_pdf(self, x: float) -> float:
        """Evaluate the logarithm of the  probability density function

        Args:
            x: Quantiles
        """
        raise NotImplementedError

    def dlog_pdf(self, x: float) -> float:
        """Partial derivative of the logarithm of the probability density function

        Args:
            x: Quantiles
        """
        raise NotImplementedError

    def random(self, n: int, hpd: float) -> np.ndarray:
        """Draw random samples from the prior distribution

        Args:
            n: Number of draws
            hpd: Highest Prior Density for drawing sample from (true for unimodal distribution)
        """
        raise NotImplementedError

    def _random(self, n: int, hpd: float, f_rvs: Callable, f_ppf: Callable) -> np.ndarray:
        """Draw random samples from a given distribution

        Args:
            n: Number of samples
            hpd: Highest probability density to draw sample from
            f_rvs: Random variable function
            f_ppf: Inverse of cumulative distribution function

        Returns:
            rvs: Random variable samples
        """

        if not isinstance(n, int) or n <= 0:
            raise TypeError('`n` must an integer greater or equal to 1')

        if hpd is not None and (hpd <= 0.0 or hpd > 1.0):
            raise ValueError("`hpd must be between ]0, 1]")

        if hpd is None or hpd == 1.0:
            rvs = f_rvs(n)
        else:
            low = (1.0 - hpd) / 2.0
            rvs = f_ppf(np.random.uniform(low=low, high=1.0 - low, size=n))

        return rvs


class Normal(Prior):
    """Normal distribution

    Args:
        mu: Location parameter
        sigma: Scale parameter > 0
    """

    def __eq__(self, other):
        return self._m == other._m and self._s == other._s

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):

        if not isinstance(mu, (int, float)):
            raise TypeError("The location parameter `mu` must be a float")

        if not isinstance(sigma, (int, float)):
            raise TypeError("The scale parameter `sigma` must be a float")

        if sigma <= 0.0:
            raise ValueError("The scale parameter `sigma` must be > 0")

        self._m = float(mu)
        self._s = float(sigma)
        self._s2 = self._s ** 2
        self._cst = -0.5 * np.log(2.0 * np.pi * self._s2)

    def __repr__(self):
        return "N({:.2g}, {:.2g})".format(self._m, self._s)

    @property
    def mean(self):
        return self._m

    @property
    def mode(self):
        return self._m

    @property
    def variance(self):
        return self._s2

    def log_pdf(self, x):
        return self._cst - 0.5 * (x - self._m) ** 2 / self._s2

    def dlog_pdf(self, x):
        return -(x - self._m) / self._s2

    def random(self, n=1, hpd=None):
        f_rvs = lambda n: stats.norm.rvs(loc=self._m, scale=self._s, size=n)
        f_ppf = lambda u: stats.norm.ppf(u, loc=self._m, scale=self._s)
        return self._random(n, hpd, f_rvs, f_ppf)

    def find_hyperparameters(
        self, lb: Real = 0, ub: Real = 1, lb_prob: Real = 0.01, ub_prob: Real = 0.01
    ):
        """Find hyper-parameter values based on lower and upper bounds
        Args:
            lb: Lower bound
            ub: Upper bound
            lb_prob: Probability at the lower bound
            ub_prob: Probability at the upper bound
        """

        def delta_tail(x, lb, ub, lb_prob, ub_prob):
            mean, sigma = np.exp(x)
            e0 = stats.norm.cdf(lb, loc=mean, scale=sigma) - lb_prob
            e1 = stats.norm.sf(ub, loc=mean, scale=sigma) - ub_prob
            return np.array([e0, e1])

        self._m = (ub - lb) / 2.0

        x, info, *_ = optimize.fsolve(
            func=delta_tail,
            x0=[np.log(self._m), np.log(self._s)],
            args=(lb, ub, lb_prob, ub_prob),
            maxfev=5000,
            full_output=True,
        )

        if np.max(info['fvec']) < 1e-6:
            self._m, self._s = np.exp(x)
            self._s2 = self._s ** 2
            self._cst = -0.5 * np.log(2.0 * np.pi * self._s2)

            print(f'solution found: mu={self._m:.4f}, sigma={self._s:.4f}')
        else:
            print('\nTry different initial values')


class Gamma(Prior):
    """Gamma distribution

    Args:
        a: Shape parameter > 0
        b: Inverse scale parameter > 0
    """

    def __init__(self, a: float = 3.0, b: float = 1.0):

        if not isinstance(a, (int, float)):
            raise TypeError("The shape parameter `a` must be a float")

        if not isinstance(b, (int, float)):
            raise TypeError("The inverse scale parameter `b` must be a float")

        if a <= 0.0:
            raise ValueError("The shape parameter `a` must be > 0")

        if b <= 0.0:
            raise ValueError("The inverse scale parameter `b` must be > 0")

        self._a = float(a)
        self._b = float(b)
        self._cst = self._a * np.log(self._b) - special.gammaln(self._a)

    def __repr__(self):
        return "Ga({:.2g}, {:.2g})".format(self._a, self._b)

    def __eq__(self, other):
        return self._a == other._a and self._b == other._b

    @property
    def mean(self):
        return self._a / self._b

    @property
    def mode(self):
        if self._a < 1:
            raise ValueError("The mode can't be computed for a < 1 !")
        return (self._a - 1.0) / self._b

    @property
    def variance(self):
        return self._a / self._b ** 2

    def log_pdf(self, x):
        return self._cst + (self._a - 1.0) * np.log(x) - self._b * x

    def dlog_pdf(self, x):
        return (self._a - 1.0) / x - self._b

    def random(self, n=1, hpd=None):
        f_rvs = lambda n: stats.gamma.rvs(a=self._a, scale=1.0 / self._b, size=n)
        f_ppf = lambda u: stats.gamma.ppf(u, a=self._a, scale=1.0 / self._b)
        return self._random(n, hpd, f_rvs, f_ppf)


class Beta(Prior):
    """Beta distribution

    Args:
        a, b: Prior sample sizes > 0
    """

    def __init__(self, a: float = 3.0, b: float = 3.0):

        if not isinstance(a, (int, float)):
            raise TypeError("The prior sample sizes parameter `a` must be a float")

        if not isinstance(b, (int, float)):
            raise TypeError("The prior sample sizes parameter `b` must be a float")

        self._a = float(a)
        self._b = float(b)
        self._cst = special.betaln(self._a, self._b)

    def __repr__(self):
        return 'B({:.2g}, {:.2g})'.format(self._a, self._b)

    def __eq__(self, other):
        return self._a == other._a and self._b == other._b

    @property
    def mean(self):
        return self._a / (self._a + self._b)

    @property
    def mode(self):
        return (self._a - 1.0) / (self._a + self._b - 2.0)

    @property
    def variance(self):
        return self._a * self._b / ((self._a + self._b) ** 2 * (self._a + self._b + 1.0))

    def log_pdf(self, x):
        return (
            (self._a - 1.0) * np.log(x)
            + (self._b - 1.0) * np.log(1.0 - x)
            - self._cst
            - (self._a + self._b - 1.0) * np.log(1.0)
        )

    def dlog_pdf(self, x):
        return (self._a - 1.0) / x + (1.0 - self._b) / (1.0 - x)

    def random(self, n=1, hpd=None):
        f_rvs = lambda n: stats.beta.rvs(a=self._a, b=self._b, size=n)
        f_ppf = lambda u: stats.beta.ppf(u, a=self._a, b=self._b)
        return self._random(n, hpd, f_rvs, f_ppf)


class InverseGamma(Prior):
    """Inverse Gamma distribution

    Args:
        a: Shape parameter > 0
        b: Scale parameter > 0
    """

    def __init__(self, a: float = 3.0, b: float = 1.0):

        if not isinstance(a, (int, float)):
            raise TypeError("The shape parameter `a` must be a float")

        if not isinstance(b, (int, float)):
            raise TypeError("The scale parameter `b` must be a float")

        if a <= 0.0:
            raise ValueError("The shape parameter `a` must be > 0")

        if b <= 0.0:
            raise ValueError("The scale parameter `b` must be > 0")

        self._a = float(a)
        self._b = float(b)
        self._cst = self._a * np.log(self._b) - special.gammaln(self._a)

    def __repr__(self):
        return "iGa({:.2g}, {:.2g})".format(self._a, self._b)

    def __eq__(self, other):
        return self._a == other._a and self._b == other._b

    @property
    def mean(self):
        if self._a <= 1:
            raise ValueError("The mean can't be computed for a <= 1 !")
        return self._b / (self._a - 1.0)

    @property
    def mode(self):
        return self._b / (self._a + 1.0)

    @property
    def variance(self):
        if self._a <= 2:
            raise ValueError("The variance can't be computed for a <= 2 !")
        return self._b ** 2 / ((self._a - 1.0) ** 2 * (self._a - 2.0))

    def log_pdf(self, x):
        return self._cst - (self._a + 1.0) * np.log(x) - self._b / x

    def dlog_pdf(self, x):
        return -(self._a + 1.0) / x + self._b / x ** 2

    def random(self, n=1, hpd=None):
        f_rvs = lambda n: stats.invgamma.rvs(a=self._a, scale=self._b, size=n)
        f_ppf = lambda u: stats.invgamma.ppf(u, a=self._a, scale=self._b)
        return self._random(n, hpd, f_rvs, f_ppf)

    def find_hyperparameters(self, lb, ub, lb_prob=0.01, ub_prob=0.01):
        """Find hyperparameters

        Args:
            lb: Lower value
            ub: Upper value
            lb_prob: Expected prior mass below `lb`
            ub_prob: Expected prior mass above `ub`

        Notes:
            If a solution is found, the values of the shape and scale
            hyperparameters are updated internally

        References:
            Michael Betancourt, Robust Gaussian Processes in Stan, Part 3,
            https://betanalpha.github.io/assets/case_studies/gp_part3/part3.html
        """

        def delta_tail(x, lb, ub, lb_prob, ub_prob):
            a = np.exp(x[0])
            b = np.exp(x[1])
            e0 = stats.invgamma.cdf(lb, a=a, scale=b) - lb_prob
            e1 = 1.0 - stats.invgamma.cdf(ub, a=a, scale=b) - ub_prob
            return np.array([e0, e1])

        x, info, *_ = optimize.fsolve(
            func=delta_tail,
            x0=[np.log(self._a), np.log(self._b)],
            args=(lb, ub, lb_prob, ub_prob),
            maxfev=1000,
            full_output=True,
        )

        if np.max(info['fvec']) < 1e-6:
            self._a, self._b = np.exp(x)
            self._cst = self._a * np.log(self._b) - special.gammaln(self._a)

            print(f'solution found: shape={self._a:.4f}, scale={self._b:.4f}')
        else:
            print('\nTry different initial values')


class LogNormal(Prior):
    """LogNormal distribution

    Args:
        mu: Location parameter
        sigma: Scale parameter > 0
    """

    def __eq__(self, other):
        return self._m == other._m and self._s == other._s

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):

        if not isinstance(mu, (int, float)):
            raise TypeError("The location parameter `mu` must be a real number")

        if not isinstance(sigma, (int, float)):
            raise TypeError("The scale parameter `sigma` must be a real number")

        if sigma <= 0.0:
            raise ValueError("The scale parameter `sigma` must be > 0")

        self._m = float(mu)
        self._s = float(sigma)
        self._s2 = self._s ** 2
        self._cst = -0.5 * np.log(2.0 * np.pi * self._s2)

    def __repr__(self):
        return "LN({:.2g}, {:.2g})".format(self._m, self._s)

    @property
    def mean(self):
        return np.exp(self._m + self._s2 / 2.0)

    @property
    def mode(self):
        return np.exp(self._m - self._s2)

    @property
    def variance(self):
        return np.exp(2.0 * self._m + self._s2) * (np.exp(self._s2) - 1.0)

    def log_pdf(self, x):
        logx = np.log(x)
        return self._cst - 0.5 * (logx - self._m) ** 2 / self._s2 - logx

    def dlog_pdf(self, x):
        return -((np.log(x) - self._m) / self._s2 + 1.0) / x

    def random(self, n=1, hpd=None):
        f_rvs = lambda n: stats.lognorm.rvs(loc=self._m, s=self._s, size=n)
        f_ppf = lambda u: stats.lognorm.ppf(u, s=self._s, loc=self._m)
        return self._random(n, hpd, f_rvs, f_ppf)
