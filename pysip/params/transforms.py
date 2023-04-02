# .. math::
#    :nowrap:

#     \\begin{equation*}
#         \\theta = loc + f(\\eta) \\times scale
#     \\end{equation*}


# where :math:`\\theta_{sd} = f(\\eta)` is one of the following `transform`

# * `none`: :math:`\\theta_{sd} =\\eta \\qquad \\theta_{sd} \\in ]-\\infty, \\infty[`
# * `log`: :math:`\\theta_{sd} =\\exp(\\eta) \\qquad \\theta_{sd} \\in [0, \\infty[`
# * `lower`: :math:`\\theta_{sd} =\\exp(\\eta) + a \\qquad \\theta_{sd} \\in [a,
#   \\infty[`
# * `upper`: :math:`\\theta_{sd} =b - \\exp(\\eta) \\qquad \\theta_{sd} \\in
#   ]-\\infty, b]`
# * `logit`: :math:`\\theta_{sd}
#   =a+\\frac{(b-a)}{1+\\exp(-\\eta)}\\qquad\\theta_{sd}\\in [a, b]`

# where :math:`a` and :math:`b` are the lower and upper bounds given as a tuple, such
# that, `bounds` = :math:`(a, b)`. If `transform` = 'fixed' the parameter is not
# considered as a random variable.

# The `prior` distribution is on the standardized constrained parameter value
# :math:`\\theta_{sd}`. In order to put the prior on the constrained parameter value
# θ, use the default configuration, `loc` = 0 and `scale` = 1.

from abc import ABCMeta, abstractmethod
import math

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
        pass

    @abstractmethod
    def untransform(self, η: float) -> float:
        pass

    @abstractmethod
    def grad_transform(self, θ: float) -> float:
        pass

    @abstractmethod
    def grad_untransform(self, η: float) -> float:
        pass

    @abstractmethod
    def penalty(self, θ: float) -> float:
        pass

    def in_bounds(self, x: float) -> bool:
        return True

    def __repr__(self):
        return f"{self.name}"


@register_transform
class NoneTransform(ParameterTransform):
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
    name = "fixed"


@register_transform
class LogTransform(ParameterTransform):
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
