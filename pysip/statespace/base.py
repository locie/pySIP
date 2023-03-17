from collections import namedtuple
from dataclasses import dataclass, field
from typing import NamedTuple, Tuple

import numpy as np

from ..core import Parameters
from ..utils.draw import TikzStateSpace
from ..utils.math import nearest_cholesky
from .discretization import (
    disc_diffusion_lyap,
    disc_diffusion_mfd,
    disc_diffusion_stationary,
    disc_state,
    disc_state_input,
)
from .meta import MetaStateSpace
from .nodes import Node

local_ssm = namedtuple("ssm", "A, B0, B1, Q")


def zeros(m, n):
    return np.zeros((m, n))


@dataclass
class StateSpace(TikzStateSpace, metaclass=MetaStateSpace):
    """Linear Gaussian Continuous-Time State-Space Model"""

    parameters: list = field(default=None)
    _names: list = field(init=False)
    nx: int = field(init=False)
    nu: int = field(init=False)
    ny: int = field(init=False)
    hold_order: int = field(default=0)
    method: str = "mfd"
    name: str = field(default="")

    def __post_init__(self):

        if self.hold_order not in [0, 1]:
            raise TypeError("`hold_order` must be either 0 or 1")

        if self.name == "":
            self.name = self.__class__.__name__

        if hasattr(self, "states"):
            self.nx = len(self.states)
            self.states = [Node(*s) for s in self.states]
        if hasattr(self, "params"):
            self.params = [Node(*s) for s in self.params]
            self._names = [p.name for p in self.params]
        if hasattr(self, "inputs"):
            self.nu = len(self.inputs)
            self.inputs = [Node(*s) for s in self.inputs]
        if hasattr(self, "outputs"):
            self.ny = len(self.outputs)
            self.outputs = [Node(*s) for s in self.outputs]

        if self.parameters:
            if not isinstance(self.parameters, Parameters):
                self.parameters = Parameters(self.parameters)
        else:
            self.parameters = Parameters(self._names)
        self.parameters._name = self.name

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

    def set_constant_continuous_ssm(self):
        """Set constant values in state-space model"""
        pass

    def update_continuous_ssm(self):
        """Update the state-space model with the constrained parameters"""
        pass

    def get_discrete_ssm(self, dt: float) -> Tuple[NamedTuple, np.ndarray]:
        """Return the updated discrete state-space model

        Args:
            dt: Sampling time

        Retuns:
            2-elements tuple containing
                - **ssm**: Discrete state-space model
                - **index**: Index of unique sampling time
        """
        self.update_continuous_ssm()
        Ad, B0d, B1d, Qd = self.discretization(dt)
        return local_ssm(Ad, B0d, B1d, Qd)

    def discretization(
        self, dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Discretization of LTI state-space model

        Args:
            dt: Sampling time
            jacobian: Discretize the jacobian if True

        Returns:
            6-elements tuple containing
                - **idx**: Index of unique time intervals
                - **Ad**: Discrete state matrix
                - **B0d**: Discrete input matrix (zero order hold)
                - **B1d**: Discrete input matrix (first order hold)
                - **Qd**: Upper Cholesky factor of the process noise covariance matrix
                - **d**: Tuple of the discrete jacobian matrices if `jacobian` = True
                    - **dAd**: Jacobian discrete state matrix
                    - **dB0d**: Jacobian discrete input matrix (zero order hold)
                    - **dB1d**: Jacobian discrete input matrix (first order hold)
                    - **dQd**: Jacobian of the upper Cholesky factor of the process
                      noise covariance
        """
        # Different sampling time up to the nanosecond

        if self.nu == 0:
            Ad = disc_state(self.A, dt)
            B0d = np.zeros((self.nx, self.nu))
            B1d = B0d
        else:
            Ad, B0d, B1d = disc_state_input(self.A, self.B, dt, self.hold_order, "expm")

        Qd = nearest_cholesky(disc_diffusion_mfd(self.A, self.Q.T @ self.Q, dt))

        return Ad, B0d, B1d, Qd


@dataclass
class RCModel(StateSpace):
    """Dynamic thermal model"""

    latent_forces: str = field(default="")

    def __post_init__(self):
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

    def discretization(
        self, dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Discretization of RC model

        Args:
            dt: sampling time

        Returns:
            4-elements tuple containing
                - **Ad**: Discrete state matrix
                - **B0d**: Discrete input matrix (zero order hold)
                - **B1d**: Discrete input matrix (first order hold)
                - **Qd**: Upper Cholesky factor of the process noise covariance
        """

        if self.method == "analytic":
            Ad, B0d, B1d = disc_state_input(
                self.A, self.B, dt, self.hold_order, "analytic"
            )
            Qd = nearest_cholesky(disc_diffusion_lyap(self.A, self.Q.T @ self.Q, Ad))
        else:
            Ad, B0d, B1d = disc_state_input(self.A, self.B, dt, self.hold_order, "expm")
            Qd = nearest_cholesky(disc_diffusion_mfd(self.A, self.Q.T @ self.Q, dt))

        return Ad, B0d, B1d, Qd


@dataclass
class GPModel(StateSpace):
    """Gaussian Process"""

    def __post_init__(self):
        if hasattr(self, "J"):
            self.states = self.states_block * int(self.J + 1)
        super().__post_init__()

    def __repr__(self):
        return f"\n{self.__class__.__name__}" + "-" * len(self.__class__.__name__)

    def __mul__(self, gp):
        """Product of two Gaussian Process model

        Args:
            gp: GPModel instance

        Returns:
            product of the two GP model
        """
        if not isinstance(gp, GPModel):
            raise TypeError("`gp` must be an GPModel instance")

        from .gaussian_process import GPProduct

        return GPProduct(self, gp)

    def __add__(self, gp):
        """Sum of two Gaussian Process model

        Args:
            gp: GPModel instance

        Returns:
            sum of the two GP model
        """
        if not isinstance(gp, GPModel):
            raise TypeError("`gp` must be an GPModel instance")

        from .gaussian_process import GPSum

        return GPSum(self, gp)

    def discretization(
        self, dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Discretization of temporal Gaussian Process

        Args:
            dt: sampling time

        Returns:
            4-elements tuple containing
                - **Ad**: Discrete state matrix
                - **B0d**: Discrete input matrix (zero order hold)
                - **B1d**: Discrete input matrix (first order hold)
                - **Qd**: Upper Cholesky factor of the process noise covariance
        """
        Ad = disc_state(self.A, dt)
        B0d = np.zeros((self.nx, self.nu))
        Qd = nearest_cholesky(disc_diffusion_stationary(self.P0.T @ self.P0, Ad))

        return Ad, B0d, B0d, Qd
