from inspect import Parameter, Signature
from makefun import with_signature
import numpy as np
import pymc as pm
from scipy import stats

__ALL__ = ["Normal", "Gamma", "Beta", "InverseGamma", "LogNormal"]


class BasePrior:
    """Prior class template

    This contains both the scipy distribution (to compute standard numerical operations)
    as well as the pymc distribution factory (to be used with the bayesian inference
    engine).
    """

    @property
    def scipy_dist(self) -> stats.rv_continuous:
        raise NotImplementedError

    @property
    def pymc_dist(self) -> type:
        raise NotImplementedError

    @property
    def shape_parameters(self) -> tuple:
        raise NotImplementedError

    def __eq__(self, __value: object) -> bool:
        return self.shape_parameters == __value.shape_parameters

    @property
    def mean(self) -> float:
        return self.scipy_dist.mean()

    def pdf(self, x: float) -> float:
        """Evaluate the probability density function

        Args:
            x: Quantiles
        """
        return self.scipy_dist.pdf(x)

    def logpdf(self, x: float) -> float:
        """Evaluate the logarithm of the  probability density function

        Args:
            x: Quantiles
        """
        return self.scipy_dist.logpdf(x)

    def random(self, n=1, hpd=None) -> np.ndarray:
        """Draw random samples from the prior distribution

        Args:
            n: Number of draws
            hpd: Highest Prior Density for drawing sample from (true for unimodal
                distribution)
        """
        if not isinstance(n, int) or n <= 0:
            raise TypeError("`n` must an integer greater or equal to 1")

        if hpd is not None and (hpd <= 0.0 or hpd > 1.0):
            raise ValueError("`hpd must be between ]0, 1]")

        if hpd is None or hpd == 1.0:
            return self.scipy_dist.rvs(n)

        low = (1.0 - hpd) / 2.0
        return self.scipy_dist.ppf(np.random.uniform(low=low, high=1.0 - low, size=n))


class PriorMeta(type):
    def __new__(cls, name, bases, attrs):
        annots = attrs.get("__annotations__")
        scipy_dist = annots.pop("scipy_dist")
        pymc_dist = annots.pop("pymc_dist")
        dist_args = annots
        defaults = {
            k: default
            for k in dist_args.keys()
            if (default := attrs.get(k, False)) is not False
        }

        def make_scipy_dist(self):
            return scipy_dist(*[getattr(self, k) for k in dist_args.keys()])

        def make_pymc_dist_factory(self):
            return lambda name: pymc_dist(
                name, *[getattr(self, k) for k in dist_args.keys()]
            )

        sign = Signature(
            parameters=[
                Parameter("self", Parameter.POSITIONAL_OR_KEYWORD),
                *[
                    Parameter(
                        k,
                        Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=annot,
                        default=defaults.get(k, Parameter.empty),
                    )
                    for k, annot in dist_args.items()
                ],
            ]
        )

        @with_signature(sign)
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def shape_parameters(self):
            return tuple([getattr(self, arg) for arg in dist_args.keys()])

        attrs["__init__"] = __init__
        attrs["shape_parameters"] = property(shape_parameters)
        attrs["scipy_dist"] = property(make_scipy_dist)
        attrs["pymc_dist"] = property(make_pymc_dist_factory)
        return super().__new__(cls, name, bases, attrs)

    def params_repr(self):
        return ", ".join(f"{param}={getattr(self, param):.3e}" for param in self.params)


class Normal(BasePrior, metaclass=PriorMeta):
    """Normal prior distribution

    Parameters
    ----------
    mu: float
        Mean of the normal distribution
    sigma: float
        Standard deviation of the normal distribution
    """

    mu: float = 0.0
    sigma: float = 1.0
    scipy_dist: stats.norm
    pymc_dist: pm.Normal


class Gamma(BasePrior, metaclass=PriorMeta):
    """Gamma prior distribution

    Parameters
    ----------
    alpha: float
        Shape parameter of the gamma distribution
    beta: float
        Rate parameter of the gamma distribution
    """

    alpha: float = 3.0
    beta: float = 1.0
    scipy_dist: lambda a, b: stats.gamma(a=a, scale=1.0 / b)
    pymc_dist: pm.Gamma


class Beta(BasePrior, metaclass=PriorMeta):
    """Beta prior distribution

    Parameters
    ----------
    alpha: float
        Shape parameter of the beta distribution
    beta: float
        Shape parameter of the beta distribution
    """

    alpha: float = 3.0
    beta: float = 3.0
    scipy_dist: stats.beta
    pymc_dist: pm.Beta


class InverseGamma(BasePrior, metaclass=PriorMeta):
    """Inverse Gamma prior distribution

    Parameters
    ----------
    alpha: float
        Shape parameter of the inverse gamma distribution
    beta: float
        Scale parameter of the inverse gamma distribution
    """

    alpha: float = 3.0
    beta: float = 1.0
    scipy_dist: lambda a, b: stats.invgamma(a=a, scale=b)
    pymc_dist: pm.InverseGamma


class LogNormal(BasePrior, metaclass=PriorMeta):
    """Log Normal prior distribution

    Parameters
    ----------
    mu: float
        Mean of the log normal distribution
    sigma: float
        Standard deviation of the log normal distribution
    """

    mu: float = 0.0
    sigma: float = 1.0
    scipy_dist: lambda mu, sigma: stats.lognorm(scale=np.exp(mu), s=sigma)
    pymc_dist: pm.Lognormal
