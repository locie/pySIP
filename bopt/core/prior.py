"""prior distributions module"""
import numpy as np
from scipy import special, stats


class Prior:
    """Prior class template

    The notation from the following book are used throughout the module:

    Appendix A - Gelman, A., Stern, H.S., Carlin, J.B., Dunson, D.B., Vehtari, A.
    and Rubin, D.B., 2013. Bayesian data analysis. Chapman and Hall/CRC.

    """

    def mean(self):
        """mean"""
        raise NotImplementedError

    def mode(self):
        """mode"""
        raise NotImplementedError

    def variance(self):
        """variance"""
        raise NotImplementedError

    def pdf(self, x):
        """probability density function at `x`"""
        return np.exp(self.log_pdf(x))

    def log_pdf(self, x):
        """log-probability density function at `x`"""
        raise NotImplementedError

    def dlog_pdf(self, x):
        """1st derivative of the log-probability density function at `x`"""
        raise NotImplementedError

    def d2log_pdf(self, x):
        """2nd derivative of the log-probability density function at `x`"""
        raise NotImplementedError

    def random(self, n=1, prob_mass=None):
        """random number generator

        Parameters
        ----------
        n : int
            number of random variables generated
        prob_mass : float
            probability mass around the mode of the distribution

        Return
        ------
        rvs : array_like
            random samples

        """
        raise NotImplementedError


class Normal(Prior):
    """Normal distribution

    Parameters
    ----------
    mu : float
        location parameter
    sigma : float
        scale parameter > 0

    """

    def __eq__(self, other):
        return self._mu == other._mu and self._sigma == other._sigma

    def __init__(self, mu=0, sigma=1):

        if not isinstance(mu, (int, float)):
            raise TypeError("The location parameter `mu` must be a float")

        if not isinstance(sigma, (int, float)):
            raise TypeError("The scale parameter `sigma` must be a float")

        if sigma <= 0.:
            raise ValueError("The scale parameter `sigma` must be > 0")

        self._mu = float(mu)
        self._sigma = float(sigma)
        self._sigma2 = self._sigma**2
        self._constant = -.5 * np.log(2. * np.pi * self._sigma2)

    def __repr__(self):
        return "N({:.2g}, {:.2g})".format(self._mu, self._sigma)

    @property
    def mean(self):
        return self._mu

    @property
    def mode(self):
        return self._mu

    @property
    def variance(self):
        return self._sigma2

    def log_pdf(self, x):
        return self._constant - .5 * (x - self._mu)**2 / self._sigma2

    def dlog_pdf(self, x):
        return -(x - self._mu) / self._sigma2

    def d2log_pdf(self, x):
        return -1. / self._sigma2

    def random(self, n=1, prob_mass=None):
        if prob_mass is not None and (prob_mass <= 0. or prob_mass > 1.):
            raise ValueError("`prob_mass must be between ]0, 1]")

        if prob_mass is None:
            rvs = stats.norm.rvs(loc=self._mu, scale=self._sigma, size=int(n))
        else:
            cut = (1. - prob_mass) / 2
            u_samples = np.random.uniform(low=cut, high=1. - cut, size=int(n))
            rvs = stats.norm.ppf(u_samples, loc=self._mu, scale=self._sigma)

        return rvs


class Gamma(Prior):
    """Gamma distribution

    Parameters
    ----------
    a: float
      shape parameter > 0
    b: float
      inverse scale parameter > 0

    """

    def __init__(self, a=2, b=10):

        if not isinstance(a, (int, float)):
            raise TypeError("The shape parameter `a` must be a float")

        if not isinstance(b, (int, float)):
            raise TypeError("The inverse scale parameter `b` must be a float")

        if a <= 0.:
            raise ValueError("The shape parameter `a` must be > 0")

        if b <= 0.:
            raise ValueError("The inverse scale parameter `b` must be > 0")

        self._a = float(a)
        self._b = float(b)
        self._constant = self._a * np.log(self._b) - special.gammaln(self._a)

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
            print("The mode can be computed only for a >= 1 !")
        else:
            return (self._a - 1.) / self._b

    @property
    def variance(self):
        return self._a / self._b**2

    def log_pdf(self, x):
        return self._constant + (self._a - 1.) * np.log(x) - self._b * x

    def dlog_pdf(self, x):
        return (self._a - 1.) / x - self._b

    def d2log_pdf(self, x):
        return (1. - self._a) / x**2

    def random(self, n=1, prob_mass=None):
        if prob_mass is not None and (prob_mass <= 0. or prob_mass > 1.):
            raise ValueError("`prob_mass must be between ]0, 1]")

        if prob_mass is None:
            rvs = stats.gamma.rvs(a=self._a, scale=1. / self._b, size=int(n))
        else:
            cut = (1. - prob_mass) / 2.
            u_samples = np.random.uniform(low=cut, high=1. - cut, size=int(n))
            rvs = stats.gamma.ppf(u_samples, a=self._a, scale=1. / self._b)

        return rvs


class Beta(Prior):
    """Beta distribution

    Parameters
    ----------
    a, b : float
        prior sample sizes > 0
    lb : float
        lower bound
    ub : float
        upper bound

    """

    def __init__(self, a=3, b=3, lb=0, ub=1):

        if not isinstance(a, (int, float)):
            raise TypeError("The prior sample sizes parameter `a` "
                            "must be a float")

        if not isinstance(b, (int, float)):
            raise TypeError("The prior sample sizes parameter `b` "
                            "must be a float")

        if not isinstance(lb, (int, float)):
            raise TypeError("The lower bound `lb` must be a float")

        if not isinstance(ub, (int, float)):
            raise TypeError("The upper bound `ub` must be a float")

        if lb >= ub:
            raise ValueError("The upper bound must be strictly "
                             "superior to the lower bound")

        self._lb = float(lb)
        self._ub = float(ub)
        self._a = float(a)
        self._b = float(b)
        self._constant = special.betaln(self._a, self._b)

    def __repr__(self):
        return ("B({:.2g}, {:.2g}, {:.2g}, {:.2g})"
                .format(self._a, self._b, self._lb, self._ub))

    def __eq__(self, other):
        return (
            self._a == other._a
            and self._b == other._b
            and self._lb == other._lb
            and self._ub == other._ub
        )

    @property
    def mean(self):
        return self._lb + (self._ub - self._lb) * self._a / (
            self._a + self._b)

    @property
    def mode(self):
        return self._lb + (self._ub - self._lb) * (self._a - 1.) / (
            self._a + self._b - 2.)

    @property
    def variance(self):
        if self._lb != 0 or self._ub != 1.:
            print("The variance expression is only valid for the "
                  " Beta distribution on the interval [0, 1]")
        else:
            return self._a * self._b / (
                (self._a + self._b)**2 * (self._a + self._b + 1.))

    def log_pdf(self, x):
        return ((self._a - 1.) * np.log(x - self._lb)
                + (self._b - 1.) * np.log(self._ub - x) - self._constant
                - (self._a + self._b - 1.) * np.log(self._ub - self._lb))

    def dlog_pdf(self, x):
        return ((1. - self._a) / (self._lb - x)
                + (1. - self._b) / (self._ub - x))

    def d2log_pdf(self, x):
        return ((1. - self._a) / (self._lb - x)**2
                + (1. - self._b) / (self._ub - x)**2)

    def random(self, n=1, prob_mass=None):
        if prob_mass is not None and (prob_mass <= 0. or prob_mass > 1.):
            raise ValueError("`prob_mass must be between ]0, 1]")

        if prob_mass is None:
            rvs = self._lb + (self._ub - self._lb) * (
                stats.beta.rvs(a=self._a, b=self._b, size=int(n)))
        else:
            cut = (1. - prob_mass) / 2.
            u_samples = np.random.uniform(low=cut, high=1. - cut, size=int(n))
            rvs = self._lb + (self._ub - self._lb) * (
                stats.beta.ppf(u_samples, a=self._a, b=self._b))

        return rvs


class InverseGamma(Prior):
    """Inverse Gamma distribution

    Parameters
    ---------
    a: float
      shape parameter > 0
    b: float
      scale parameter > 0

    """

    def __init__(self, a=2, b=0.05):

        if not isinstance(a, (int, float)):
            raise TypeError("The shape parameter `a` must be a float")

        if not isinstance(b, (int, float)):
            raise TypeError("The scale parameter `b` must be a float")

        if a <= 0.:
            raise ValueError("The shape parameter `a` must be > 0")

        if b <= 0.:
            raise ValueError("The scale parameter `b` must be > 0")

        self._a = float(a)
        self._b = float(b)
        self._constant = self._a * np.log(self._b) - special.gammaln(self._a)

    def __repr__(self):
        return "iGa({:.2g}, {:.2g})".format(self._a, self._b)

    def __eq__(self, other):
        return self._a == other._a and self._b == other._b

    @property
    def mean(self):
        if self._a <= 1:
            print("The mean can be computed only for a > 1 !")
        else:
            return self._b / (self._a - 1.)

    @property
    def mode(self):
        return self._b / (self._a + 1.)

    @property
    def variance(self):
        if self._a <= 2:
            print("The variance can be computed only for a > 2 !")
        else:
            return self._b**2 / ((self._a - 1.)**2 * (self._a - 2.))

    def log_pdf(self, x):
        return self._constant - (self._a + 1.) * np.log(x) - self._b / x

    def dlog_pdf(self, x):
        return -(self._a + 1.) / x + self._b / x**2

    def d2log_pdf(self, x):
        return ((self._a + 1.) * x - 2. * self._b) / x**3

    def random(self, n=1, prob_mass=None):
        if prob_mass is not None and (prob_mass <= 0. or prob_mass > 1.):
            raise ValueError("`prob_mass must be between ]0, 1]")

        if prob_mass is None:
            rvs = stats.invgamma.rvs(a=self._a, scale=self._b, size=int(n))
        else:
            cut = (1. - prob_mass) / 2.
            u_samples = np.random.uniform(low=cut, high=1. - cut, size=int(n))
            rvs = stats.invgamma.ppf(u_samples, a=self._a, scale=self._b)

        return rvs
