from typing import Literal, Optional, Tuple

from pydantic import ConfigDict, confloat, validator
from pydantic.dataclasses import dataclass

from .prior import BasePrior
from .transforms import FixedTransform, ParameterTransform, Transforms, auto_transform


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class Parameter:
    """A Parameter has a value in the constrained space θ and in the
    unconstrained space η. See the documentation of the class for more details on the
    different transformations.

    Parameters
    ----------
    name : str
        Parameter name
    value : float
        Parameter value
    loc : float
        Location value
    scale : float
        Scaling value
    transform : str | ParameterTransform
        Bijections transform θ -> η and untransform η -> θ
    bounds : tuple
        Parameters bounds
    prior : Distribution
        Prior distribution

    Attributes
    ----------
    theta : float
        Parameter value in the constrained space θ
    theta_sd : float
        Parameter value in the standardized constrained space θ_sd
    eta : float
        Parameter value in the unconstrained space η
    eta_sd : float
        Parameter value in the standardized unconstrained space η_sd
    free : bool
        False if the parameter is fixed
    """

    name: str
    value: float = None
    loc: float = 0.0
    scale: confloat(gt=0) = 1.0
    bounds: Tuple[Optional[float], Optional[float]] = (None, None)
    transform: ParameterTransform | Literal[
        "auto", "fixed", "none", "log", "lower", "upper", "logit"
    ] = "auto"
    prior: BasePrior = None

    @validator("bounds")
    def _validate_bounds(cls, bounds):
        lb, ub = bounds
        if lb is not None and ub is not None and lb >= ub:
            raise ValueError("`lb` must be lower than `ub`")
        return bounds

    @validator("transform")
    def _validate_transform(cls, transform, values):
        if isinstance(transform, ParameterTransform):
            return transform
        bounds = values.get("bounds")
        if transform == "auto":
            return auto_transform(bounds)
        return Transforms.get(transform)(bounds)

    def __post_init_post_parse__(self):
        if self.value is None:
            self.eta = 0.0
        else:
            self.theta = self.loc + self.scale * self.value
        if not self.transform.in_bounds(self.value):
            raise ValueError(
                f"Initial value {self.value} is out of bounds {self.bounds}"
            )

    def __repr__(self):
        return (
            f"name={self.name} value={self.value:.3e} transform={self.transform}"
            f" bounds={self.bounds} prior={self.prior}"
        )

    @property
    def theta(self) -> float:
        """Returns constrained parameter value θ"""
        return self.loc + self.scale * self.value

    @theta.setter
    def theta(self, x):
        """Set constrained parameter value θ"""
        self.value = (x - self.loc) / self.scale
        self._eta = self.transform.transform(self.value)

    @property
    def theta_sd(self) -> float:
        return self.value

    @theta_sd.setter
    def theta_sd(self, x):
        self.value = x
        self.transform.transform(self.value)

    @property
    def eta(self) -> float:
        return self._eta

    @eta.setter
    def eta(self, x):
        self._eta = x
        self.value = self.transform.untransform(self._eta)

    @property
    def free(self) -> bool:
        return not isinstance(self.transform, FixedTransform)

    def get_transformed(self):
        """Do inverse transformation θsd = f^{-1}(η)"""
        return self.transform.transform(self._eta)

    def get_transform_jacobian(self):
        """Get the jacobian of η = f(θsd)"""
        return self.transform.grad_transform(self.value)

    def get_inv_transformed(self):
        """Do inverse transformation θsd = f^{-1}(η)"""
        return self.transform.untransform(self._eta)

    def get_inv_transform_jacobian(self) -> float:
        """Get the jacobian of θsd = f⁻¹(η)"""
        return self.transform.grad_untransform(self._eta)

    def get_penalty(self) -> float:
        """Penalty function"""
        return self.transform.penalty(self.value)

    def get_grad_penalty(self) -> float:
        """Penalty function"""
        return self.transform.grad_penalty(self.value)