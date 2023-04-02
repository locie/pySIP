""" Parameter transforms

This module contains the different transforms that can be applied to a parameter.
The transforms are used to transform a parameter value θ to the unconstrained space
η and vice versa. The transforms are used to ensure that the parameters are
constrained to a certain range and that the optimization algorithm can find the
optimal parameters.

The transforms are defined in the class ParameterTransform. The class has the
following abstract attributes / methods:

    - name: Name of the transform
    - transform: Transform a parameter value θ to the unconstrained space η
    - untransform: Transform a parameter value η to the constrained space θ
    - grad_transform: Gradient of the transform function
    - grad_untransform: Gradient of the untransform function

The transforms are registered in the Transforms namespace. The namespace is used
to get the transform class from the name of the transform.

The following transforms are available:

    - None: No transform
    - Fixed: Fixed parameter
    - Log: Log transform
    - Lower: Lower bound
    - Upper: Upper bound
    - Logit: Logit transform

An auto transform is also available. The auto transform will select the best
transform based on the bounds of the parameter.
"""

from abc import ABCMeta, abstractmethod
import math
from typing_extensions import Self

from ..utils.misc import Namespace


Transforms = Namespace()

# decorator to register transforms
def register_transform(cls):
    Transforms[cls.name] = cls
    return cls


class ParameterTransform(metaclass=ABCMeta):
    def __init__(self, bounds: tuple):
        self.lb, self.ub = bounds

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def transform(self, θ: float) -> float:
        """ Transform a parameter value θ to the unconstrained space η

        Parameters
        ----------
        θ : float
            Parameter value in the constrained space θ

        Returns
        -------
        η : float
            Parameter value in the unconstrained space η
        """
        pass

    @abstractmethod
    def untransform(self, η: float) -> float:
        """ Transform a parameter value η to the constrained space θ

        Parameters
        ----------
        η : float
            Parameter value in the unconstrained space η

        Returns
        -------
        θ : float
            Parameter value in the constrained space θ
        """
        pass

    @abstractmethod
    def grad_transform(self, θ: float) -> float:
        """ Gradient of the transform function

        Parameters
        ----------
        θ : float
            Parameter value in the constrained space θ

        Returns
        -------
        grad : float
            Gradient of the transform function
        """
        pass

    @abstractmethod
    def grad_untransform(self, η: float) -> float:
        """ Gradient of the untransform function

        Parameters
        ----------
        η : float
            Parameter value in the unconstrained space η

        Returns
        -------
        grad : float
            Gradient of the untransform function
        """
        pass

    @abstractmethod
    def penalty(self, θ: float) -> float:
        """ Penalty for the parameter value θ

        Parameters
        ----------
        θ : float
            Parameter value in the constrained space θ

        Returns
        -------
        penalty : float
            Penalty for the parameter value θ
        """
        pass

    def in_bounds(self, x: float) -> bool:
        """ Check if the parameter value is in the bounds of the transform

        Parameters
        ----------
        x : float
            Parameter value in the constrained space θ

        Returns
        -------
        in_bounds : bool
            True if the parameter value is in the bounds of the transform
        """
        return True

    def __repr__(self):
        return f"{self.name}"

    def __eq__(self, __value: Self) -> bool:
        if isinstance(__value, ParameterTransform):
            return self.name == __value.name
        return False


@register_transform
class NoneTransform(ParameterTransform):
    """ No transform, i.e. θ = η"""
    name = "none"

    def transform(self, θ: float) -> float:
        return θ

    def untransform(self, η: float) -> float:
        return η

    def grad_transform(self, θ: float) -> float:
        return 1.0

    def grad_untransform(self, η: float) -> float:
        return 1.0

    def penalty(self, θ: float) -> float:
        return 0.0


@register_transform
class FixedTransform(NoneTransform):
    """ Fixed transform, i.e. θ = η, but the parameter is not considered as a random
    variable.
    """
    name = "fixed"


@register_transform
class LogTransform(ParameterTransform):
    """ Log transform, i.e. θ = exp(η)"""
    name = "log"

    def transform(self, θ: float) -> float:
        return math.log(θ)

    def untransform(self, η: float) -> float:
        return math.exp(η)

    def grad_transform(self, θ: float) -> float:
        return 1.0 / θ

    def grad_untransform(self, η: float) -> float:
        return math.exp(η)

    def penalty(self, θ: float) -> float:
        return 1e-12 / (θ - 1e-12)


@register_transform
class LowerTransform(ParameterTransform):
    """ Lower bound transform, i.e. θ = exp(η) + a, where a is the lower bound"""
    name = "lower"

    def transform(self, θ: float) -> float:
        return math.log(θ - self.lb)

    def untransform(self, η: float) -> float:
        return math.exp(η) + self.lb

    def grad_transform(self, θ: float) -> float:
        return 1.0 / (θ - self.lb)

    def grad_untransform(self, η: float) -> float:
        return math.exp(η)

    def penalty(self, θ: float) -> float:
        return abs(self.lb) / (θ - self.lb)

    def in_bounds(self, x: float) -> bool:
        return x > self.lb


@register_transform
class UpperTransform(ParameterTransform):
    """ Upper bound transform, i.e. θ = a - exp(η), where a is the upper bound"""
    name = "upper"

    def transform(self, θ: float) -> float:
        return math.log(self.ub - θ)

    def untransform(self, η: float) -> float:
        return self.ub - math.exp(η)

    def grad_transform(self, θ: float) -> float:
        return 1.0 / (self.ub - θ)  # TODO: check

    def grad_untransform(self, η: float) -> float:
        return -math.exp(η)

    def penalty(self, θ: float) -> float:
        return abs(self.ub) / (self.ub - θ)

    def in_bounds(self, x: float) -> bool:
        return x < self.ub


@register_transform
class LogitTransform(ParameterTransform):
    """ Logit transform, i.e. θ = a + (b - a) / (1 + exp(-η)), where a and b are the
    lower and upper bounds, respectively
    ."""
    name = "logit"

    def transform(self, θ: float) -> float:
        return math.log((θ - self.lb) / (self.ub - θ))

    def untransform(self, η: float) -> float:
        return self.lb + (self.ub - self.lb) / (1 + math.exp(-η))

    def grad_transform(self, θ: float) -> float:
        return (self.ub - self.lb) / ((θ - self.lb) * (self.ub - θ))

    def grad_untransform(self, η: float) -> float:
        x = math.exp(-η)
        return (self.ub - self.lb) * x / (1 + x) ** 2

    def penalty(self, θ: float) -> float:
        return UpperTransform.penalty(self, θ) + LowerTransform.penalty(self, θ)

    def in_bounds(self, x: float) -> bool:
        return self.lb < x < self.ub


def auto_transform(bounds):
    """ Automatically select a transform based on the bounds

    Parameters
    ----------
    bounds : tuple
        Lower and upper bounds of the parameter. Both bounds can be None.

    Returns
    -------
    transform : ParameterTransform
        Transform that is automatically selected based on the bounds
    """
    lb, ub = bounds
    if lb is None and ub is None:
        return Transforms["none"](bounds)
    if lb == 0.0 and ub is None:
        return Transforms["log"](bounds)
    if ub is None and lb > 0.0:
        return Transforms["lower"](bounds)
    if lb is None and ub > 0.0:
        return Transforms["upper"](bounds)
    if lb is not None and ub is not None:
        return Transforms["logit"](bounds)
    raise ValueError("No transform found for bounds {}".format(bounds))
