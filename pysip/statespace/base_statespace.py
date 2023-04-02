from dataclasses import field
from typing import NamedTuple, Self, Tuple

import numpy as np
from pydantic import ConfigDict, conint
from pydantic.dataclasses import dataclass

from ..params import Parameters
from ..utils.math import nearest_cholesky
from . import discretization
from .meta import MetaStateSpace
from .nodes import Node


class States(NamedTuple):
    C: np.ndarray
    D: np.ndarray
    R: np.ndarray
    A: np.ndarray
    B0: np.ndarray
    B1: np.ndarray
    Q: np.ndarray


class DiscreteStates(NamedTuple):
    A: np.ndarray
    B0: np.ndarray
    B1: np.ndarray
    Q: np.ndarray


def zeros(m, n):
    return np.zeros((m, n))


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class StateSpace(metaclass=MetaStateSpace):
    """Linear Gaussian Continuous-Time State-Space Model"""

    parameters: Parameters = field(default=None, repr=False)
    hold_order: conint(ge=1, le=1) = 0
    method: str = "mfd"
    name: str = ""

    def _coerce_attributes(self):
        for attr in ["states", "params", "inputs", "outputs"]:
            setattr(self, attr, [Node(*s) for s in getattr(self, attr)])

        if self.parameters:
            if not isinstance(self.parameters, Parameters):
                self.parameters = Parameters(self.parameters)
        else:
            self.parameters = Parameters([p.name for p in self.params])
        self.parameters.name = self.name

    def _init_states(self):
        self.A = np.zeros((self.nx, self.nx))
        self.B = np.zeros((self.nx, self.nu))
        self.C = np.zeros((self.ny, self.nx))
        self.D = np.zeros((self.ny, self.nu))
        self.Q = np.zeros((self.nx, self.nx))
        self.R = np.zeros((self.ny, self.ny))
        self.x0 = np.zeros((self.nx, 1))
        self.P0 = np.zeros((self.nx, self.nx))
        self.set_constant_continuous_ssm()
        self._diag = np.diag_indices_from(self.A)

    def __post_init__(self):
        if self.name == "":
            self.name = self.__class__.__name__
        self._coerce_attributes()
        self._init_states()

    @property
    def nx(self):
        return len(self.states)

    @property
    def ny(self):
        return len(self.outputs)

    @property
    def nu(self):
        return len(self.inputs)

    def set_constant_continuous_ssm(self):
        """Set constant values in state-space model"""
        pass

    def update_continuous_ssm(self):
        """Update the state-space model with the constrained parameters"""
        pass

    def get_discrete_ssm(self, dt: float) -> DiscreteStates:
        """Return the updated discrete state-space model

        Parameters
        ----------
        dt : float
            Sampling time

        Returns
        -------
        DiscreteStates
            Discrete state-space model, a 4-elements namedtuple containing
            - **A**: Discrete state matrix
            - **B0**: Discrete input matrix (zero order hold)
            - **B1**: Discrete input matrix (first order hold)
            - **Q**: Upper Cholesky factor of the process noise covariance matrix
        """
        self.update_continuous_ssm()
        return self.discretization(dt)

    def discretization(
        self, dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Discretization of LTI state-space model. Should be overloaded by subclasses.

        Parameters
        ----------
        dt : float
            Sampling time

        Returns
        -------
        DiscreteStates
            Discrete state-space model, a 4-elements namedtuple containing
            - **A**: Discrete state matrix
            - **B0**: Discrete input matrix (zero order hold)
            - **B1**: Discrete input matrix (first order hold)
            - **Q**: Upper Cholesky factor of the process noise covariance matrix
        """

        if self.nu == 0:
            Ad = discretization.state(self.A, dt)
            B0d = np.zeros((self.nx, self.nu))
            B1d = B0d
        else:
            Ad, B0d, B1d = discretization.state_input(
                self.A, self.B, dt, self.hold_order, "expm"
            )

        # Qd = disc_diffusion_mfd(self.A, self.Q.T @ self.Q, dt)
        Qd = nearest_cholesky(
            discretization.diffusion_mfd(self.A, self.Q.T @ self.Q, dt)
        )

        return Ad, B0d, B1d, Qd


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class RCModel(StateSpace):
    """Dynamic thermal model"""

    latent_forces: str = ""

    def __post_init__(self):
        print(self.name)
        super().__post_init__()
        eig = np.real(np.linalg.eigvals(self.A))
        if np.all(eig < 0) and eig.max() / eig.min():
            self._method = "analytic"
        else:
            self._method = "mfd"

    def __repr__(self):
        return f"\n{self.__class__.__name__}" + "-" * len(self.__class__.__name__)

    def __le__(self, gp):
        """Create a Latent Force Model"""
        from .latent_force_model import LatentForceModel

        return LatentForceModel(self, gp, self.latent_forces)

    def discretization(self, dt: float) -> DiscreteStates:
        """Discretization of RC model

        Parameters
        ----------
        dt : float
            Sampling time

        Returns
        -------
        DiscreteStates
            Discrete state-space model, a 4-elements namedtuple containing
            - **A**: Discrete state matrix
            - **B0**: Discrete input matrix (zero order hold)
            - **B1**: Discrete input matrix (first order hold)
            - **Q**: Upper Cholesky factor of the process noise covariance matrix
        """

        if self.method == "analytic":
            Ad, B0d, B1d = discretization.state_input(
                self.A, self.B, dt, self.hold_order, "analytic"
            )
            Qd = nearest_cholesky(
                discretization.diffusion_lyap(self.A, self.Q.T @ self.Q, Ad)
            )
        else:
            Ad, B0d, B1d = discretization.state_input(
                self.A, self.B, dt, self.hold_order, "expm"
            )
            Qd = nearest_cholesky(
                discretization.diffusion_mfd(self.A, self.Q.T @ self.Q, dt)
            )

        return DiscreteStates(Ad, B0d, B1d, Qd)


class GPModel(StateSpace):
    """Gaussian Process"""

    def __post_init__(self):
        if hasattr(self, "J"):
            self.states = self.states_block * int(self.J + 1)
        super().__post_init__()

    def __repr__(self):
        return f"\n{self.__class__.__name__}" + "-" * len(self.__class__.__name__)

    def __mul__(self, gp: Self):
        if not isinstance(gp, GPModel):
            raise TypeError("`gp` must be an GPModel instance")
        # TODO: refactor to avoid circular import
        from .gaussian_process import GPProduct
        return GPProduct(self, gp)

    def __add__(self, gp: Self):
        if not isinstance(gp, GPModel):
            raise TypeError("`gp` must be an GPModel instance")
        # TODO: refactor to avoid circular import
        from .gaussian_process import GPSum
        return GPSum(self, gp)

    def discretization(
        self, dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Discretization of the temporal Gaussian Process

        Parameters
        ----------
        dt : float
            Sampling time

        Returns
        -------
        DiscreteStates
            Discrete state-space model, a 4-elements namedtuple containing
            - **A**: Discrete state matrix
            - **B0**: Discrete input matrix (zero order hold)
            - **B1**: Discrete input matrix (first order hold)
            - **Q**: Upper Cholesky factor of the process noise covariance matrix
        """
        Ad = discretization.state(self.A, dt)
        B0d = np.zeros((self.nx, self.nu))
        Qd = nearest_cholesky(
            discretization.diffusion_stationary(self.P0.T @ self.P0, Ad)
        )

        return Ad, B0d, B0d, Qd
