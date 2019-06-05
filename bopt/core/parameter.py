import numpy as np
from .prior import Prior


class Parameter(object):
    """A parameter with a double representation..

    A parameter is a floating value that can be represented both as
    :math:`\\theta` and :math:`\\eta` that are linked with:

    .. math::

        \\eta = f(\\theta)


    where :math:`f` can be one of these transforms:

    .. math::

        \\eta = \\theta

    .. math::

        \\eta = \\log(\\theta)

    .. math::

        \\eta = \\log(\\frac{\\theta - l_b}{u_b - \\theta})

    Attributes:
        name (str): name
        value (float): value
        transform (str) : transformation ('none', 'log', 'lowup', 'fixed', 'lower', 'upper')
        bounds (tuple): constrained parameters bounds
        prior (Prior): prior
    """

    __available_transforms__ = ['none', 'log', 'lowup', 'fixed', 'lower', 'upper']
    __eps__ = np.finfo(float).eps

    def __init__(self, name, value=0.0, transform=None, bounds=(None, None), prior=None):

        if not isinstance(name, str):
            raise TypeError('name should be a string')
        self.name = name

        if not isinstance(bounds, tuple):
            raise TypeError('bounds should be a tuple')

        if bounds[0] is not None and bounds[1] is not None and bounds[0] > bounds[1]:
            raise ValueError('lower bound > upper bound')
        self.bounds = bounds

        if not isinstance(value, float):
            raise TypeError('value should be a float')
        self.value = value

        if not transform:
            if bounds[0] is None and bounds[1] is None:
                transform = 'none'
            elif bounds[0] is not None and bounds[1] is None and bounds[0] == 0:
                transform = 'log'
            elif bounds[0] is not None and bounds[1] is None and bounds[0] > 0:
                transform = 'lower'
            elif bounds[0] is None and bounds[1] > 0:
                transform = 'upper'
            elif bounds[0] is not None and bounds[1] is not None:
                transform = 'lowup'

        if not(isinstance(transform, str) and transform in self.__available_transforms__):
            raise ValueError(f'transform should be one of those strings: {self.__available_transforms__}')
        self.transform = transform

        if prior is not None:
            if not isinstance(prior, Prior):
                raise ValueError('prior must be an instance of Prior')
        self.prior = prior

    def __repr__(self):
        return f'Parameter({self.name}, value={self.value:.3e}, transform={self.transform}, bounds={self.bounds}, prior={self.prior})'

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
        return self.value

    @theta.setter
    def theta(self, x):
        self.value = x

    @property
    def eta(self):
        return self._transform()

    @eta.setter
    def eta(self, x):
        return self._inv_transform(x)

    @property
    def free(self):
        return not(self.transform == 'fixed')

    def _transform(self):
        """Get :math:`\\eta`

        The values to close to the bounds are clipped to the square root of
        machine precision for float numbers to avoid devision by 0 in
        _transform and _transform_jacobian.
        """
        if self.transform in ["none", "fixed"]:
            return self.value

        elif self.transform == "log":
            if self.value < self.__eps__:
                self.value = self.__eps__

            return np.log(self.value)

        elif self.transform == "lower":
            if np.abs(self.value - self.bounds[0]) < self.__eps__:
                self.value = self.bounds[0] + self.__eps__
            return np.log(self.value - self.bounds[0])

        elif self.transform == "upper":
            if np.abs(self.bounds[1] - self.value) < self.__eps__:
                self.value = self.bounds[1] - self.__eps__
            return np.log(self.bounds[1] - self.value)

        elif self.transform == "lowup":
            if np.abs(self.bounds[1] - self.value) < self.__eps__:
                self.value = self.bounds[1] - self.__eps__

            if np.abs(self.value - self.bounds[0]) < self.__eps__:
                self.value = self.bounds[0] + self.__eps__

            return np.log((self.value - self.bounds[0]) / (self.bounds[1] - self.value))

    def _transform_jacobian(self):
        """Get the jacobian of :math:`\\eta = f(\\theta)`

        The transformations are independent, therefore, instead of returning a
        diagonal jacobian matrix we return a vector. In this case, the
        determinant of the jacobian adjustment in the log posterior,
        e.g. ln(|J|), can be replaced by ln(J).sum()
        """
        if self.transform in ["none", "fixed"]:
            return 1.0
        elif self.transform == "log":
            return 1.0 / self.value
        elif self.transform == "lower":
            return 1.0 / (self.value - self.bounds[0])
        elif self.transform == "upper":
            return 1.0 / (self.value - self.bounds[1])
        elif self.transform == "lowup":
            return ((self.bounds[1] - self.bounds[0])
                    / ((self.value - self.bounds[0])
                    * (self.bounds[1] - self.value)))

    def _inv_transform(self, value):
        """Set the value :math:`\\theta`"""
        if self.transform in ["none", "fixed"]:
            self.value = value
        elif self.transform == "log":
            self.value = np.exp(value)
        elif self.transform == "lower":
            self.value = np.exp(value) + self.bounds[0]
        elif self.transform == "upper":
            self.value = self.bounds[1] - np.exp(value)
        elif self.transform == "lowup":
            self.value = self.bounds[0] + ((self.bounds[1] - self.bounds[0])
                                           / (1 + np.exp(-value)))

    def _inv_transform_jacobian(self):
        """Get the jacobian of :math:`\\theta = f^{-1}(\\eta)`

        The transformations are independent, therefore, instead of returning a
        diagonal jacobian matrix we return a vector. In this case, the
        determinant of the jacobian adjustment in the log posterior,
        e.g. ln(det|J|), can be replaced by ln(J).sum()
        """
        if self.transform in ["none", "fixed"]:
            return 1.0
        elif self.transform in ["log", "lower", "upper"]:
            return self.value
        elif self.transform == "lowup":
            x = np.exp(-self._transform())
            return (x * (self.bounds[1] - self.bounds[0])) / (1.0 + x) ** 2

    def _inv_transform_dlog_jacobian(self):
        """Partial derivative of the logarithm of the inverse transform

         :math:`\\frac{\\partial \\ln(\\theta)}{\\partial \\eta}`
        """
        if self.transform in ["none", "fixed"]:
            return 0.0
        elif self.transform in ["log", "lower", "upper"]:
            return 1.0
        elif self.transform == "lowup":
            # faster with -tanh(eta/2)
            x = np.exp(self._transform())
            return (1.0 - x) / (1.0 + x)

    def _inv_transform_d2log_jacobian(self):
        """Second partial derivative of the logarithm of the inverse transform

        :math:`\\frac{\\partial^{2} \\ln(\\theta)}{\\partial^{2} \\eta}`
        """
        if self.transform in ["none", "log", "fixed", "lower", "upper"]:
            return 0.0
        elif self.transform == "lowup":
            # faster with -1/(2*cosh(eta/2)**2)
            x = np.exp(self._transform())
            return (-2.0 * x) / (1.0 + x) ** 2

    def _penalty(self):
        """If :math:`\\theta` is a constrained parameter, the penalty function
        increases the objective function near the bounds"""
        if self.transform in ["none", "fixed"]:
            return 0.0
        elif self.transform == "log":
            return 1e-14 / (self.value - 1e-14)
        elif self.transform == "lower":
            return np.abs(self.bounds[0]) / (self.value - self.bounds[0])
        elif self.transform == "upper":
            return np.abs(self.bounds[1]) / (self.bounds[1] - self.value)
        elif self.transform == "lowup":
            return (np.abs(self.bounds[0]) / (self.value - self.bounds[0])
                    + np.abs(self.bounds[1]) / (self.bounds[1] - self.value))

    def _d_penalty(self):
        """If :math:`\\theta` is a constrained parameter, the derivative of the
        penalty function increases the gradient near the bounds"""
        if self.transform in ["none", "fixed"]:
            return 0.0
        elif self.transform == "log":
            return - 1e-14 / (self.value - 1e-14)**2
        elif self.transform == "lower":
            return - np.abs(self.bounds[0]) / (self.value - self.bounds[0]) ** 2
        elif self.transform == "upper":
            return np.abs(self.bounds[1]) / (self.bounds[1] - self.value)**2
        elif self.transform == "lowup":
            return (np.abs(self.bounds[1]) / (self.bounds[1] - self.value)**2
                    - np.abs(self.bounds[0]) / (self.value - self.bounds[0])**2)
