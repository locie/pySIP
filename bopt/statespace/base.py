from dataclasses import dataclass, field

import numpy as np
from numpy.linalg import solve
from scipy.linalg import expm, expm_frechet
from scipy.linalg import solve_continuous_lyapunov as lyap

from ..core import Parameters
from ..utils.matrices import pseudo_cholesky
from ..utils.draw import TikzStateSpace

from .nodes import Node
from .meta import MetaStateSpace


@dataclass
class StateSpace(TikzStateSpace, metaclass=MetaStateSpace):
    """Linear Gaussian Continuous-Time State-Space Model

    Attributes
    ----------
    names : array_like
        Parameter names
    Nx : int
        Number of states
    Nu : int
        Number of inputs
    Ny : int
        Number of outputs
    jacobian : bool
        Flag for computing the jacobian matrices
    A : array_like
        State matrix
    B : array_like
        Input matrix
    C : array_like
        Output matrix
    D : array_like
        Feedthrough matrix
    Q : array_like
        Scaling matrix of the Wiener process
    R : array_like
        Measurement noise standard deviation matrix
    x0 : array_like
        Initial state vector mean
    P0 : array_like
        Initial state vector standard deviation matrix
    dA : dict
        Jacobian of the state matrix w.r.t the parameters
    dB : dict
        Jacobian of the input matrix w.r.t the parameters
    dC : dict
        Jacobian of the output matrix w.r.t the parameters
    dD : dict
        Jacobian of the feedthrough matrix w.r.t the parameters
    dQ : dict
        Jacobian of the scaling matrix w.r.t the parameters
    dR : dict
        Jacobian of the measurement noise std matrix w.r.t the parameters
    dx0 : dict
        Jacobian of the initial state vector mean w.r.t the parameters
    dP0 : dict
        Jacobian of the initial state vector std w.r.t the parameters

    """

    parameters: list = field(default=None)
    _names: list = field(init=False)
    Nx: int = field(init=False)
    Nu: int = field(init=False)
    Ny: int = field(init=False)
    jacobian: bool = field(default=True)
    hold_order: str = field(default='zoh')
    name: str = field(default='')
    # _parameters: Parameters = field(default=None)
    # _epsilons: Parameters = field(default=None)

    # @property
    # def parameters(self):
    #     return self._parameters + self._epsilons

    def __post_init__(self):
        """create a state space model of appropriate dimensions

        Parameters
        ----------
        names : array_like
            Parameter names
        Nx : int
            Number of states
        Nu : int
            Number of inputs
        Ny : int
            Number of outputs
        jacobian : bool
            Flag for computing the jacobian matrices

        """
        if not isinstance(self.jacobian, bool):
            raise TypeError("`jacobian` must be a boolean")

        if self.hold_order not in ['zoh', 'foh']:
            raise TypeError("`hold_order` must be either 'zoh' or 'foh'")

        if self.name == '':
            self.name = self.__class__.__name__       

        if hasattr(self, 'states'):
            self.Nx = len(self.states)
            self.states = [Node(*s) for s in self.states]
        if hasattr(self, 'params'):
            self.params = [Node(*s) for s in self.params]
            self._names = [p.name for p in self.params]
        if hasattr(self, 'inputs'):
            self.Nu = len(self.inputs)
            self.inputs = [Node(*s) for s in self.inputs]
        if hasattr(self, 'outputs'):
            self.Ny = len(self.outputs)
            self.outputs = [Node(*s) for s in self.outputs]

        if self.parameters:
            if not isinstance(self.parameters, Parameters):
                self.parameters = Parameters(self.parameters)
        else:
            self.parameters = Parameters(self._names)
        self.parameters._name = self.name

        self.A = np.zeros((self.Nx, self.Nx))
        self.B = np.zeros((self.Nx, self.Nu))
        self.C = np.zeros((self.Ny, self.Nx))
        self.D = np.zeros((self.Ny, self.Nu))
        self.Q = np.zeros((self.Nx, self.Nx))
        self.R = np.zeros((self.Ny, self.Ny))
        self.x0 = np.zeros((self.Nx, 1))
        self.P0 = np.zeros((self.Nx, self.Nx))

        Np = len(self._names)
        self._Ixx = np.broadcast_to(
            np.eye(2 * self.Nx), (Np, 2 * self.Nx, 2 * self.Nx))
        self._0xu = np.zeros((Np, self.Nx, self.Nu))
        self._0xx = np.zeros((Np, self.Nx, self.Nx))
        self._diag = np.diag_indices_from(self.A)

        self.set_constant()

        if self.jacobian:
            self.dA = {k: np.zeros((self.Nx, self.Nx)) for k in self._names}
            self.dB = {k: np.zeros((self.Nx, self.Nu)) for k in self._names}
            self.dC = {k: np.zeros((self.Ny, self.Nx)) for k in self._names}
            self.dD = {k: np.zeros((self.Ny, self.Nu)) for k in self._names}
            self.dQ = {k: np.zeros((self.Nx, self.Nx)) for k in self._names}
            self.dR = {k: np.zeros((self.Ny, self.Ny)) for k in self._names}
            self.dx0 = {k: np.zeros((self.Nx, 1)) for k in self._names}
            self.dP0 = {k: np.zeros((self.Nx, self.Nx)) for k in self._names}

            # Allocate more memory for initial state mean, initial state
            # standard deviation and measurement noise standard deviation
            self._AA = np.zeros((Np, 2 * self.Nx, 2 * self.Nx))
            self._BB = np.zeros((Np, 2 * self.Nx, self.Nu))
            self._AAd = np.zeros((Np, 2 * self.Nx, 2 * self.Nx))

            self.set_jacobian()

    def update(self):
        """Update state-space model and jacobians"""
        self.update_state_space_model()
        if self.jacobian:
            self.update_jacobian()

    def set_constant(self):
        """Set constant values in state-space model"""
        pass

    def set_jacobian(self):
        """Set constant values in jacobians"""
        pass

    def update_state_space_model(self):
        """Update the state-space model with the constrained parameters"""
        pass

    def update_jacobian(self):
        """Update the jacobians with the constrained parameters"""
        pass

    def _lti_disc(self, dt):
        """Discretization of LTI state-space model

        Args:
            dt: sampling time

        Returns:
            Ad: Discrete state matrix
            B0d, B1d: Discrete input matrix
            Qd: Upper Cholesky factor process noise covariance
        """
        Ad = expm(self.A * dt)

        if not np.all(self.Q == 0):
            Q2 = self.Q.T @ self.Q
            Qd = pseudo_cholesky(lyap(self.A, -Q2 + Ad @ Q2 @ Ad.T))
        else:
            Qd = self._0xx[0, :, :]

        if self.Nu != 0:
            bis = solve(self.A, Ad - self._Ixx[0, :self.Nx, :self.Nx])
            B0d = bis @ self.B
            if self.hold_order == 'foh':
                B1d = solve(self.A, -bis + Ad * dt) @ self.B
            else:
                B1d = self._0xu[0, :, :]
        else:
            B0d = self._0xu[0, :, :]
            B1d = B0d

        return Ad, B0d, B1d, Qd

    def _lti_jacobian_disc(self, dt):
        """Discretization of augmented LTI state-space model

        Args:
            dt: sampling time

        Returns:
            Ad: Discrete state matrix
            B0d, B1d: Discrete input matrix
            Qd: Upper Cholesky factor process noise covariance
            dAd: Derivative discrete state matrix
            dB0d, dB1d: Derivative discrete input matrix
            dQd: Derivative upper Cholesky factor process noise covariance
        """
        names = self.parameters.names_free
        N = len(names)

        # Discrete state matrix and its derivative
        self._AA[:N, :self.Nx, :self.Nx] = self.A
        self._AA[:N, self.Nx:, self.Nx:] = self.A
        self._AA[:N, self.Nx:, :self.Nx] = [self.dA[n] for n in names]
        dA = self._AA[:N, self.Nx:, :self.Nx]

        Ad = expm(self.A * dt)
        for i in range(N):
            self._AAd[i, :self.Nx, :self.Nx] = Ad
            self._AAd[i, self.Nx:, self.Nx:] = Ad
            if not np.all(dA[i, :, :] == 0):
                self._AAd[i, self.Nx:, :self.Nx] = (
                    expm_frechet(self.A * dt, dA[i, :, :] * dt, 'SPS', False)
                )
        dAd = self._AAd[:N, self.Nx:, :self.Nx]

        if not np.all(self.Q == 0):
            dQ = np.asarray([self.dQ[k] for k in names])

            # transform to covariance matrix
            Q2 = self.Q.T @ self.Q
            dQ2 = dQ.swapaxes(1, 2) @ self.Q + self.Q.T @ dQ

            # covariance matrix of the process noise
            Qd2 = lyap(self.A, -Q2 + Ad @ Q2 @ Ad.T)

            eq = (-dA @ Qd2 - Qd2 @ dA.swapaxes(1, 2)
                  - dQ2 + dAd @ Q2 @ Ad.T
                  + Ad @ dQ2 @ Ad.T
                  + Ad @ Q2 @ dAd.swapaxes(1, 2))

            # Derivative process noise covariance
            dQd2 = np.asarray([lyap(self.A, eq[i, :, :]) for i in range(N)])

            # Get Cholesky factor
            Qd = pseudo_cholesky(Qd2)
            tmp = solve(Qd.T, solve(Qd.T, dQd2).swapaxes(1, 2))
            dQd = (np.triu(tmp, 1) + self._Ixx[0, :self.Nx, :self.Nx]
                   * 0.5 * tmp.diagonal(0, 1, 2)[:, np.newaxis, :]) @ Qd
        else:
            Qd = self._0xx[0, :, :]
            dQd = self._0xx[:N, :, :]

        if self.Nu != 0:
            # Discrete input matrices and their derivatives
            self._BB[:N, :self.Nx, :] = self.B
            self._BB[:N, self.Nx:, :] = [self.dB[k] for k in names]

            AA = self._AA[:N, :, :]
            AAd = self._AAd[:N, :, :]
            BB = self._BB[:N, :, :]
            bis = solve(AA, AAd - self._Ixx[:N, :, :])
            BB0d = bis @ BB
            B0d = BB0d[0, :self.Nx, :]
            dB0d = BB0d[:, self.Nx:, :]

            if self.hold_order == 'foh':
                BBd1_free = solve(AA, -bis + AAd * dt) @ BB
                B1d = BBd1_free[0, :self.Nx, :]
                dB1d = BBd1_free[:, self.Nx:, :]
            else:
                B1d = self._0xu[0, :, :]
                dB1d = self._0xu[:N, :, :]
        else:
            B0d = self._0xu[0, :, :]
            dB0d = self._0xu[:N, :, :]
            B1d = B0d
            dB1d = dB0d

        return Ad, B0d, B1d, Qd, dAd, dB0d, dB1d, dQd

    def discretization(self, dt):
        """Discretization of LTI state-space model

        Args:
            dt: sampling time

        Returns:
            idx: Index of unique time intervals
            Ad, B0d, B1d, Qd: Discrete state space matrices
            dAd, dB0d, dB1d, dQd: Derivative of discrete state space matrices
                if `jacobian` is True
        """
        # Different sampling time up to the nanosecond
        dt, _, idx = np.unique(np.round(dt, 9), True, True)
        N = len(dt)

        Ad = np.empty((N, self.Nx, self.Nx))
        B0d = np.empty((N, self.Nx, self.Nu))
        B1d = np.empty((N, self.Nx, self.Nu))
        Qd = np.empty((N, self.Nx, self.Nx))

        if self.jacobian:
            Np = len(self.parameters.names_free)
            dAd = np.empty((N, Np, self.Nx, self.Nx))
            dB0d = np.empty((N, Np, self.Nx, self.Nu))
            dB1d = np.empty((N, Np, self.Nx, self.Nu))
            dQd = np.empty((N, Np, self.Nx, self.Nx))

            for n in range(N):
                Ad[n, :, :], B0d[n, :, :], B1d[n, :, :], Qd[n, :, :], \
                    dAd[n, :, :], dB0d[n, :, :], dB1d[n, :, :], dQd[n, :, :] \
                    = self._lti_jacobian_disc(dt[n])

            d = (dAd, dB0d, dB1d, dQd)

        else:
            for n in range(N):
                Ad[n, :, :], B0d[n, :, :], B1d[n, :, :], Qd[n, :, :] \
                    = self._lti_disc(dt[n])

            d = ()

        return (idx, Ad, B0d, B1d, Qd, *d)


@dataclass
class RCModel(StateSpace):
    """Dynamic thermal model"""

    sC: float = field(default=1e8)
    sX: float = field(default=1e2)

    def __post_init__(self):
        super().__post_init__()

    def __repr__(self):
        return (f"\n{self.__class__.__name__}"
                + "-" * len(self.__class__.__name__)
                + f"scale thermal capacities : {self.sC:.2e}"
                + f"scale initial state mean : {self.sX:.2e}")


@dataclass
class GPModel(StateSpace):
    """Gaussian Process"""

    def __post_init__(self):
        if hasattr(self, 'J'):
            self.states = self.states_block * int(self.J + 1)
        super().__post_init__()
        self.count = 1

    def __repr__(self):
        return (f"\n{self.__class__.__name__}"
                + "-" * len(self.__class__.__name__))

    def __mul__(self, gp):
        """Product of two Gaussian Process covariance

        Args:
            gp: GPModel instance

        Returns:
            gp_prod: product of the two GP covariance
        """
        if not isinstance(gp, GPModel):
            raise TypeError('`gp` must be an GPModel instance')

        if not self.jacobian & gp.jacobian:
            raise ValueError('The jacobian attribute must have '
                             'the same boolean value')

        from .gaussian_process.gp_product import GPProduct
        return GPProduct(self, gp)
