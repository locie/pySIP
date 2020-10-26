from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from functools import partial
from typing import NamedTuple, Tuple, Union

import numpy as np

from ..core import Parameters
from .discretization import (
    disc_state,
    disc_state_input,
    disc_diffusion_mfd,
    disc_diffusion_stationary,
    disc_diffusion_lyap,
    disc_d_state,
    disc_d_state_input,
    disc_d_diffusion_mfd,
    disc_d_diffusion_stationary,
    disc_d_diffusion_lyap,
)
from ..utils.draw import TikzStateSpace
from ..utils.math import nearest_cholesky, diff_upper_cholesky
from .meta import MetaStateSpace
from .nodes import Node

ssm = namedtuple('ssm', 'A, B0, B1, C, D, Q, R, x0, P0')
dssm = namedtuple('dssm', 'dA, dB0, dB1, dC, dD, dQ, dR, dx0, dP0')


def zeros(m, n):
    return np.zeros((m, n))


@dataclass
class StateSpace(TikzStateSpace, metaclass=MetaStateSpace):
    '''Linear Gaussian Continuous-Time State-Space Model'''

    parameters: list = field(default=None)
    _names: list = field(init=False)
    nx: int = field(init=False)
    nu: int = field(init=False)
    ny: int = field(init=False)
    hold_order: int = field(default=0)
    method: str = 'mfd'
    name: str = field(default='')

    def __post_init__(self):

        if self.hold_order not in [0, 1]:
            raise TypeError("`hold_order` must be either 0 or 1")

        if self.name == '':
            self.name = self.__class__.__name__

        if hasattr(self, 'states'):
            self.nx = len(self.states)
            self.states = [Node(*s) for s in self.states]
        if hasattr(self, 'params'):
            self.params = [Node(*s) for s in self.params]
            self._names = [p.name for p in self.params]
        if hasattr(self, 'inputs'):
            self.nu = len(self.inputs)
            self.inputs = [Node(*s) for s in self.inputs]
        if hasattr(self, 'outputs'):
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
        self._init_continuous_dssm()
        self.set_constant_continuous_dssm()

        self._diag = np.diag_indices_from(self.A)

    def init_continuous_dssm(self):
        """Initialize the jacobians of the continuous state-space model"""
        self._init_continuous_dssm()
        self.set_constant_continuous_dssm()

    def _init_continuous_dssm(self):
        """Initialize the jacobians of the continuous state-space model"""

        self.dA = defaultdict(partial(zeros, self.nx, self.nx))
        self.dB = defaultdict(partial(zeros, self.nx, self.nu))
        self.dC = defaultdict(partial(zeros, self.ny, self.nx))
        self.dD = defaultdict(partial(zeros, self.ny, self.nu))
        self.dQ = defaultdict(partial(zeros, self.nx, self.nx))
        self.dR = defaultdict(partial(zeros, self.ny, self.ny))
        self.dx0 = defaultdict(partial(zeros, self.nx, 1))
        self.dP0 = defaultdict(partial(zeros, self.nx, self.nx))

    def delete_continuous_dssm(self):
        """Delete the jacobians of the continuous state-space model"""
        self._delete_continuous_dssm()

    def _delete_continuous_dssm(self):
        """Delete the jacobians of the continuous state-space model"""

        jacobians = ['dA', 'dB', 'dC', 'dD', 'dQ', 'dR', 'dx0', 'dP0']
        for j in jacobians:
            delattr(self, j)

    def set_constant_continuous_ssm(self):
        """Set constant values in state-space model"""
        pass

    def set_constant_continuous_dssm(self):
        """Set constant values in jacobians"""
        pass

    def update_continuous_ssm(self):
        """Update the state-space model with the constrained parameters"""
        pass

    def update_continuous_dssm(self):
        """Update the jacobians with the constrained parameters"""
        pass

    def get_discrete_ssm(self, dt: np.ndarray) -> Tuple[NamedTuple, np.ndarray]:
        """Return the updated discrete state-space model

        Args:
            dt: Sampling time

        Retuns:
            2-elements tuple containing
                - **ssm**: Discrete state-space model
                - **index**: Index of unique sampling time
        """

        self.update_continuous_ssm()
        index, Ad, B0d, B1d, Qd, *_ = self.discretization(dt, False)
        return ssm(Ad, B0d, B1d, self.C, self.D, Qd, self.R, self.x0, self.P0), index

    def get_discrete_dssm(self, dt: np.ndarray) -> Tuple[NamedTuple, NamedTuple, np.ndarray]:
        """Return the updated discrete state-space model with the discrete jacobians

        Args:
            dt: Sampling time

        Retuns:
            3-elements tuple containing
                - **ssm**: Discrete state-space model
                - **dssm**: Jacobian discrete state-space model
                - **index**: Index of unique sampling time
        """

        self.update_continuous_ssm()
        self.update_continuous_dssm()
        index, Ad, B0d, B1d, Qd, dAd, dB0d, dB1d, dQd = self.discretization(dt, True)

        free = self.parameters.free
        dC = np.asarray([self.dC[n] for n, f in zip(self._names, free) if f])
        dD = np.asarray([self.dD[n] for n, f in zip(self._names, free) if f])
        dR = np.asarray([self.dR[n] for n, f in zip(self._names, free) if f])
        dx0 = np.asarray([self.dx0[n] for n, f in zip(self._names, free) if f])
        dP0 = np.asarray([self.dP0[n] for n, f in zip(self._names, free) if f])

        return (
            ssm(Ad, B0d, B1d, self.C, self.D, Qd, self.R, self.x0, self.P0),
            dssm(dAd, dB0d, dB1d, dC, dD, dQd, dR, dx0, dP0),
            index,
        )

    def _lti_disc(self, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Discretization of LTI state-space model

        Args:
            dt: sampling time

        Returns:
            4-elements tuple containing
                - **Ad**: Discrete state matrix
                - **B0d**: Discrete input matrix (zero order hold)
                - **B1d**: Discrete input matrix (first order hold)
                - **Qd**: Upper Cholesky factor of the process noise covariance
        """

        if self.nu == 0:
            Ad = disc_state(self.A, dt)
            B0d = np.zeros((self.nx, self.nu))
            B1d = B0d
        else:
            Ad, B0d, B1d = disc_state_input(self.A, self.B, dt, self.hold_order, 'expm')

        Qd = nearest_cholesky(disc_diffusion_mfd(self.A, self.Q.T @ self.Q, dt))

        return Ad, B0d, B1d, Qd

    def _lti_jacobian_disc(
        self, dt: float, dA: np.ndarray, dB: np.ndarray, dQ: np.ndarray
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Discretization of augmented LTI state-space model

        Args:
            dt: Sampling time
            dA: Jacobian state matrix
            dB: Jacobian input matrix
            dQ: Jacobian Wiener process scaling matrix

        Returns:
            8-elements tuple containing
                - **Ad**: Discrete state matrix
                - **B0d**: Discrete input matrix (zero order hold)
                - **B1d**: Discrete input matrix (first order hold)
                - **Qd**: Upper Cholesky factor of the process noise covariance matrix
                - **dAd**: Jacobian discrete state matrix
                - **dB0d**: Jacobian discrete input matrix (zero order hold)
                - **dB1d**: Jacobian discrete input matrix (first order hold)
                - **dQd**: Jacobian of the upper Cholesky factor of the process noise covariance
        """
        nj = dA.shape[0]

        if self.nu == 0:
            Ad, dAd = disc_d_state(self.A, dA, dt)

            B0d = np.zeros((self.nx, self.nu))
            dB0d = np.zeros((nj, self.nx, self.nu))

            B1d = B0d
            dB1d = dB0d
        else:
            Ad, B0d, B1d, dAd, dB0d, dB1d = disc_d_state_input(
                self.A, self.B, dA, dB, dt, self.hold_order, 'expm'
            )

        Qc = self.Q.T @ self.Q
        dQc = dQ.swapaxes(1, 2) @ self.Q + self.Q.T @ dQ
        Qcd, dQcd = disc_d_diffusion_mfd(self.A, Qc, dA, dQc, dt)

        Qd = nearest_cholesky(Qcd)
        dQd = np.zeros((nj, self.nx, self.nx))
        for n in range(nj):
            if dQcd[n].any():
                dQd[n] = diff_upper_cholesky(Qd, dQcd[n])

        return Ad, B0d, B1d, Qd, dAd, dB0d, dB1d, dQd

    def discretization(
        self, dt: np.ndarray, jacobian: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple]:
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
                    - **dQd**: Jacobian of the upper Cholesky factor of the process noise covariance
        """
        # Different sampling time up to the nanosecond
        dt, _, idx = np.unique(np.round(dt, 9), True, True)
        N = len(dt)

        Ad = np.empty((N, self.nx, self.nx))
        B0d = np.empty((N, self.nx, self.nu))
        B1d = np.empty((N, self.nx, self.nu))
        Qd = np.empty((N, self.nx, self.nx))

        if jacobian:
            free = self.parameters.free

            dA = np.array([self.dA[n] for n, f in zip(self._names, free) if f])
            dB = np.array([self.dB[n] for n, f in zip(self._names, free) if f])
            if isinstance(self, GPModel):
                dQ = np.array([self.dP0[n] for n, f in zip(self._names, free) if f])
            else:
                dQ = np.array([self.dQ[n] for n, f in zip(self._names, free) if f])

            Np = dA.shape[0]
            dAd = np.empty((N, Np, self.nx, self.nx))
            dB0d = np.empty((N, Np, self.nx, self.nu))
            dB1d = np.empty((N, Np, self.nx, self.nu))
            dQd = np.empty((N, Np, self.nx, self.nx))

            for n in range(N):
                (
                    Ad[n],
                    B0d[n],
                    B1d[n],
                    Qd[n],
                    dAd[n],
                    dB0d[n],
                    dB1d[n],
                    dQd[n],
                ) = self._lti_jacobian_disc(dt[n], dA, dB, dQ)

            d = (dAd, dB0d, dB1d, dQd)

        else:
            for n in range(N):
                Ad[n], B0d[n], B1d[n], Qd[n] = self._lti_disc(dt[n])

            d = ()

        return (idx, Ad, B0d, B1d, Qd, *d)


@dataclass
class RCModel(StateSpace):
    """Dynamic thermal model"""

    latent_forces: str = field(default='')

    def __post_init__(self):
        super().__post_init__()

    def __repr__(self):
        return f"\n{self.__class__.__name__}" + "-" * len(self.__class__.__name__)

    def __le__(self, gp):
        """Create a Latent Force Model"""
        from .latent_force_model import LatentForceModel

        return LatentForceModel(self, gp, self.latent_forces)

    def _lti_disc(self, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        eig = np.real(np.linalg.eigvals(self.A))

        if np.all(eig < 0) and eig.max() / eig.min() > 1e-10:
            Ad, B0d, B1d = disc_state_input(self.A, self.B, dt, self.hold_order, 'analytic')
            Qd = nearest_cholesky(disc_diffusion_lyap(self.A, self.Q.T @ self.Q, Ad))
        else:
            Ad, B0d, B1d = disc_state_input(self.A, self.B, dt, self.hold_order, 'expm')
            Qd = nearest_cholesky(disc_diffusion_mfd(self.A, self.Q.T @ self.Q, dt))

        return Ad, B0d, B1d, Qd

    def _lti_jacobian_disc(
        self, dt: float, dA: np.ndarray, dB: np.ndarray, dQ: np.ndarray
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Discretization of RC model and its derivatives

        Args:
            dt: Sampling time
            dA: Derivative state matrix
            dB: Derivative input matrix
            dQ: Derivative Wiener process scaling matrix

        Returns:
            8-elements tuple containing
                - **Ad**: Discrete state matrix
                - **B0d**: Discrete input matrix (zero order hold)
                - **B1d**: Discrete input matrix (first order hold)
                - **Qd**: Upper Cholesky factor of the process noise covariance matrix
                - **dAd**: Derivative discrete state matrix
                - **dB0d**: Derivative discrete input matrix (zero order hold)
                - **dB1d**: Derivative discrete input matrix (first order hold)
                - **dQd**: Derivative of the upper Cholesky factor of the process noise covariance
        """
        Qc = self.Q.T @ self.Q
        dQc = dQ.swapaxes(1, 2) @ self.Q + self.Q.T @ dQ

        eig = np.real(np.linalg.eigvals(self.A))

        if np.all(eig < 0) and eig.max() / eig.min() > 1e-10:
            Ad, B0d, B1d, dAd, dB0d, dB1d = disc_d_state_input(
                self.A, self.B, dA, dB, dt, self.hold_order, 'analytic'
            )
            Qcd, dQcd = disc_d_diffusion_lyap(self.A, Qc, Ad, dA, dQc, dAd)
        else:
            Ad, B0d, B1d, dAd, dB0d, dB1d = disc_d_state_input(
                self.A, self.B, dA, dB, dt, self.hold_order, 'expm'
            )
            Qcd, dQcd = disc_d_diffusion_mfd(self.A, Qc, dA, dQc, dt)

        nj = dQcd.shape[0]
        Qd = nearest_cholesky(Qcd)
        dQd = np.zeros((nj, self.nx, self.nx))
        for n in range(nj):
            if dQcd[n].any():
                dQd[n] = diff_upper_cholesky(Qd, dQcd[n])

        return Ad, B0d, B1d, Qd, dAd, dB0d, dB1d, dQd


@dataclass
class GPModel(StateSpace):
    """Gaussian Process"""

    def __post_init__(self):
        if hasattr(self, 'J'):
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
            raise TypeError('`gp` must be an GPModel instance')

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
            raise TypeError('`gp` must be an GPModel instance')

        from .gaussian_process import GPSum

        return GPSum(self, gp)

    def _lti_disc(self, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    def _lti_jacobian_disc(
        self, dt: float, dA: np.ndarray, dB: np.ndarray, dPinf_upper: np.ndarray
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Discretization of augmented temportal Gaussian Process

        Args:
            dt: Sampling time
            dA: Jacobian state matrix
            dB: Jacobian input matrix
            dPinf: Jacobian upper Cholesky factor of the stationary covariance matrix

        Returns:
            8-elements tuple containing
                - **Ad**: Discrete state matrix
                - **B0d**: Discrete input matrix (zero order hold)
                - **B1d**: Discrete input matrix (first order hold)
                - **Qd**: Upper Cholesky factor of the process noise covariance matrix
                - **dAd**: Jacobian discrete state matrix
                - **dB0d**: Jacobian discrete input matrix (zero order hold)
                - **dB1d**: Jacobian discrete input matrix (first order hold)
                - **dQd**: Jacobian of the upper Cholesky factor of the process noise covariance
        """
        nj = dA.shape[0]
        Ad, dAd = disc_d_state(self.A, dA, dt)

        B0d = np.zeros((self.nx, self.nu))
        dB0d = np.zeros((nj, self.nx, self.nu))

        dPinf = dPinf_upper.swapaxes(1, 2) @ self.P0 + self.P0.T @ dPinf_upper
        Qcd, dQcd = disc_d_diffusion_stationary(self.P0.T @ self.P0, Ad, dPinf, dAd)

        Qd = nearest_cholesky(Qcd)
        dQd = np.zeros((nj, self.nx, self.nx))
        for n in range(nj):
            if dQcd[n].any():
                dQd[n] = diff_upper_cholesky(Qd, dQcd[n])

        return Ad, B0d, B0d, Qd, dAd, dB0d, dB0d, dQd
