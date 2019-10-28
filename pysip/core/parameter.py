import numpy as np
from numbers import Real
from typing import Tuple
from .prior import Prior


class Parameter(object):
    """A parameter with a double representation

    Parameter has a value in a constrained space :math:`\\theta` and in unconstrained space
    :math:`\\eta`. These two values are linked by one of the following bijections:

    * `none`: :math:`\\theta = \\eta \\quad \\theta \\in ]-\\infty, \\infty[`
    * `log`: :math:`\\theta = \\exp(\\eta) \\quad \\theta \\in [0, \\infty[`
    * `lower`: :math:`\\theta = \\exp(\\eta) + a \\quad \\theta \\in [a, \\infty[`
    * `upper`: :math:`\\theta = b - \\exp(\\eta) \\quad \\theta \\in ]-\\infty, b]`
    * `logit`: :math:`\\theta = a + \\frac{(b - a)}{1 + \\exp(-\\eta)} \\quad \\theta \\in [a, b]`

    with :math:`a` and :math:`b` the lower and upper bounds given as a tuple such that,
    `bounds` = :math:`(a, b)`. If `transform` = 'fixed' the parameter is not considered as a
    random variable.

    Args:
        name: Parameter name
        value: Parameter value
        scale: Scaling value
        transform: Bijections
        bounds: Parameters bounds
        prior: Prior distribution
    """

    __transforms__ = ['fixed', 'none', 'log', 'lower', 'upper', 'logit']

    def __init__(
        self,
        name: str,
        value: Real = None,
        scale: Real = 1.0,
        transform: str = None,
        bounds: Tuple[Real] = (None, None),
        prior: Prior = None,
        **kwargs,
    ):

        if not isinstance(name, str):
            raise TypeError('`name` should be a string')
        self.name = name

        if not isinstance(scale, Real):
            raise TypeError('`scale` must be a real number')

        if scale <= 0:
            raise ValueError('`scale` must be a real positive number')
        self.scale = scale

        if not isinstance(bounds, tuple):
            raise TypeError('`bounds` should be a tuple')
        lb, ub = bounds

        if lb is not None:
            if not isinstance(lb, Real):
                TypeError('`bounds` values must be real numbers')
            lb = float(lb)

        if ub is not None:
            if not isinstance(ub, Real):
                TypeError('`bounds` values must be real numbers')
            ub = float(ub)

        if lb is not None and ub is not None and lb >= ub:
            raise ValueError('lower bound > upper bound')

        if transform is None:
            if lb is None and ub is None:
                transform = 'none'
            elif lb == 0.0 and ub is None:
                transform = 'log'
            elif ub is None and lb > 0.0:
                transform = 'lower'
            elif lb is None and ub > 0.0:
                transform = 'upper'
            elif lb is not None and ub is not None:
                transform = 'logit'

        if not (isinstance(transform, str) and transform in self.__transforms__):
            raise TypeError(f'Available tranform: {self.__transforms__}')
        self.transform = transform

        self.bounds = (lb, ub)

        if value is not None:
            if not isinstance(value, Real):
                raise TypeError('`value` must be a real number')
            self.theta = float(value) * self.scale
        else:
            self.eta = 0.0

        if transform in ['lower', 'logit'] and self.value <= lb:
            raise ValueError('`value` is outside the bounds')

        if transform in ['upper', 'logit'] and self.value >= ub:
            raise ValueError('`value` is outside the bounds')

        if prior is not None and not isinstance(prior, Prior):
            raise ValueError('`prior` must be an instance of Prior')
        self.prior = prior

    def __repr__(self):
        return (
            f'name={self.name} value={self.value:.3e} transform={self.transform}'
            f' bounds={self.bounds} prior={self.prior}'
        )

    def __eq__(self, other):
        return (
            self.value == other.value
            and self.name == other.name
            and self.transform == other.transform
            and self.bounds == other.bounds
            and self.prior == other.prior
        )

    @property
    def theta(self):
        """Return constrained parameter"""
        return self.value * self.scale

    @theta.setter
    def theta(self, x):
        """Set constrained parameter"""
        self.value = x / self.scale
        self._transform()

    @property
    def eta(self):
        """Return unconstrained parameter"""
        return self._eta

    @eta.setter
    def eta(self, x):
        """Return unconstrained parameter"""
        self._eta = x
        self._inv_transform()

    @property
    def free(self):
        """Return True if the parameter is not fixed"""
        return not self.transform == 'fixed'

    def _transform(self):
        """Get :math:`\\eta`"""

        if self.transform in ['none', 'fixed']:
            self._eta = self.value

        elif self.transform == 'log':
            self._eta = np.log(self.value)

        elif self.transform == 'lower':
            self._eta = np.log(self.value - self.bounds[0])

        elif self.transform == 'upper':
            self._eta = np.log(self.bounds[1] - self.value)

        elif self.transform == 'logit':
            self._eta = np.log((self.value - self.bounds[0]) / (self.bounds[1] - self.value))

    def _transform_jacobian(self):
        """Get the jacobian of :math:`\\eta = f(\\theta)`

        The transformations are independent, therefore, instead of returning a
        diagonal jacobian matrix we return a vector. In this case, the
        determinant of the jacobian adjustment in the log posterior,
        e.g. ln(|J|), can be replaced by ln(J).sum()
        """
        if self.transform in ['none', 'fixed']:
            return 1.0

        elif self.transform == 'log':
            return 1.0 / self.value

        elif self.transform == 'lower':
            return 1.0 / (self.value - self.bounds[0])

        elif self.transform == 'upper':
            return -1.0 / (self.bounds[1] - self.value)

        elif self.transform == 'logit':
            return (self.bounds[1] - self.bounds[0]) / (
                (self.value - self.bounds[0]) * (self.bounds[1] - self.value)
            )

    def _inv_transform(self):
        """Get :math:`\\theta`"""
        if self.transform in ['none', 'fixed']:
            self.value = self._eta

        elif self.transform == 'log':
            self.value = np.exp(self._eta)

        elif self.transform == 'lower':
            self.value = np.exp(self._eta) + self.bounds[0]

        elif self.transform == 'upper':
            self.value = self.bounds[1] - np.exp(self._eta)

        elif self.transform == 'logit':
            self.value = self.bounds[0] + (self.bounds[1] - self.bounds[0]) / (
                1 + np.exp(-self._eta)
            )

    def _inv_transform_jacobian(self):
        """Get the jacobian of :math:`\\theta = f^{-1}(\\eta)`

        The transformations are independent, therefore, instead of returning a
        diagonal jacobian matrix we return a vector. In this case, the
        determinant of the jacobian adjustment in the log posterior,
        e.g. ln(det|J|), can be replaced by ln(J).sum()

        Notes:
            chain rule doesn't use absolute value like the jacobian adjustment
        """
        if self.transform in ['none', 'fixed']:
            return 1.0

        elif self.transform == 'log':
            return self.value

        elif self.transform == 'lower':
            return np.exp(self._eta)

        elif self.transform == 'upper':
            return -np.exp(self._eta)

        elif self.transform == 'logit':
            x = np.exp(-self._eta)
            return (self.bounds[1] - self.bounds[0]) * x / (1.0 + x) ** 2

    def _inv_transform_dlog_jacobian(self):
        """Partial derivative of the logarithm of the inverse transform

         :math:`\\frac{\\partial \\ln(\\theta)}{\\partial \\eta}`
        """
        if self.transform in ['none', 'fixed']:
            return 0.0

        elif self.transform in ['log', 'lower', 'upper']:
            return 1.0

        elif self.transform == 'logit':
            return -np.tanh(self._eta / 2.0)

    def _penalty(self):
        """If :math:`\\theta` is a constrained parameter, the penalty function
        increases the objective function near the bounds"""
        if self.transform in ["none", "fixed"]:
            return 0.0

        elif self.transform == 'log':
            return 1e-12 / (self.value - 1e-12)

        elif self.transform == 'lower':
            return np.abs(self.bounds[0]) / (self.value - self.bounds[0])

        elif self.transform == 'upper':
            return np.abs(self.bounds[1]) / (self.bounds[1] - self.value)

        elif self.transform == 'logit':
            return np.abs(self.bounds[0]) / (self.value - self.bounds[0]) + np.abs(
                self.bounds[1]
            ) / (self.bounds[1] - self.value)

    def _d_penalty(self):
        """If :math:`\\theta` is a constrained parameter, the derivative of the
        penalty function increases the gradient near the bounds"""
        if self.transform in ["none", "fixed"]:
            return 0.0

        elif self.transform == 'log':
            return -1e-12 / (self.value - 1e-12) ** 2

        elif self.transform == 'lower':
            return -np.abs(self.bounds[0]) / (self.value - self.bounds[0]) ** 2

        elif self.transform == 'upper':
            return np.abs(self.bounds[1]) / (self.bounds[1] - self.value) ** 2

        elif self.transform == 'logit':
            return (
                np.abs(self.bounds[1]) / (self.bounds[1] - self.value) ** 2
                - np.abs(self.bounds[0]) / (self.value - self.bounds[0]) ** 2
            )
