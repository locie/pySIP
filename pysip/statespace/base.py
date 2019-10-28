from dataclasses import dataclass, field
from collections import defaultdict, namedtuple
from functools import partial
import numpy as np
from scipy.linalg import expm, expm_frechet, solve_continuous_lyapunov, LinAlgError, LinAlgWarning
from .nodes import Node
from .meta import MetaStateSpace
from ..core import Parameters
from ..utils.math import nearest_cholesky
from ..utils.draw import TikzStateSpace
import warnings

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

        # self.dA = {k: np.zeros((self.nx, self.nx)) for k in self._names}
        # self.dB = {k: np.zeros((self.nx, self.nu)) for k in self._names}
        # self.dC = {k: np.zeros((self.ny, self.nx)) for k in self._names}
        # self.dD = {k: np.zeros((self.ny, self.nu)) for k in self._names}
        # self.dQ = {k: np.zeros((self.nx, self.nx)) for k in self._names}
        # self.dR = {k: np.zeros((self.ny, self.ny)) for k in self._names}
        # self.dx0 = {k: np.zeros((self.nx, 1)) for k in self._names}
        # self.dP0 = {k: np.zeros((self.nx, self.nx)) for k in self._names}

    def delete_continuous_dssm(self):
        """Delete the jacobians of the continuous state-space model"""
        self._delete_continuous_dssm()

    def _delete_continuous_dssm(self):
        """Delete the jacobians of the continuous state-space model"""

        jacobians = ['dA', 'dB', 'dC', 'dD', 'dQ', 'dR', 'dx0', 'dP0']
        for j in jacobians:
            delattr(self, j)

    def set_constant_continuous_ssm(self):
        '''Set constant values in state-space model'''
        pass

    def set_constant_continuous_dssm(self):
        '''Set constant values in jacobians'''
        pass

    def update_continuous_ssm(self):
        '''Update the state-space model with the constrained parameters'''
        pass

    def update_continuous_dssm(self):
        '''Update the jacobians with the constrained parameters'''
        pass

    def get_discrete_ssm(self, dt):
        '''Return the updated discrete state-space model'''

        self.update_continuous_ssm()
        index, Ad, B0d, B1d, Qd, *_ = self.discretization(dt, False)
        return ssm(Ad, B0d, B1d, self.C, self.D, Qd, self.R, self.x0, self.P0), index

    def get_discrete_dssm(self, dt):
        '''Return the updated discrete state-space model with the discrete jacobians'''

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

    def _lti_disc(self, dt):
        '''Discretization of LTI state-space model

        Args:
            dt: sampling time

        Returns:
            Ad: Discrete state matrix
            B0d, B1d: Discrete input matrices
            Qd: Upper Cholesky factor of the process noise covariance matrix
        '''
        if self.nu == 0:
            Ad = expm(self.A * dt)
            B0d = np.zeros((self.nx, self.nu))
            B1d = B0d
        else:
            if self.hold_order == 0:
                AA = np.zeros((self.nx + self.nu, self.nx + self.nu))
                AA[: self.nx, : self.nx] = self.A
                AA[: self.nx, self.nx :] = self.B
                AAd = expm(AA * dt)
                Ad, B0d = AAd[: self.nx, : self.nx], AAd[: self.nx, self.nx :]

                if self.nu >= self.nx:
                    B1d = AA[-self.nx :, self.nx :]
                else:
                    B1d = np.zeros((self.nx, self.nu))

            elif self.hold_order == 1:
                AA = np.zeros((self.nx + 2 * self.nu, self.nx + 2 * self.nu))
                AA[: self.nx, : self.nx] = self.A
                AA[: self.nx, self.nx : self.nx + self.nu] = self.B
                AA[self.nx : self.nx + self.nu, self.nx + self.nu :] = np.eye(self.nu)
                AAd = expm(AA * dt)
                Ad, B0d = AAd[: self.nx, : self.nx], AAd[: self.nx, self.nx : self.nx + self.nu]
                B1d = AAd[: self.nx, self.nx + self.nu :]

            else:
                pass  # prepare for second order hold

        if np.any(self.Q):

            if self.method == 'mfd':
                try:
                    QQd = self._disc_Q_mfd(dt, self.Q.T @ self.Q)
                except (LinAlgError, LinAlgWarning, RuntimeError, RuntimeWarning):
                    QQd = self._disc_Q_lyapunov(self.Q.T @ self.Q, Ad)
            elif self.method == 'lyapunov':
                try:
                    QQd = self._disc_Q_lyapunov(self.Q.T @ self.Q, Ad)
                except (LinAlgError, LinAlgWarning, RuntimeError, RuntimeWarning):
                    QQd = self._disc_Q_mfd(dt, self.Q.T @ self.Q)
            else:
                raise ValueError('`Invalid discretization method`')

            try:
                Qd = np.linalg.cholesky(QQd).T
            except (LinAlgError, LinAlgWarning, RuntimeError, RuntimeWarning):
                Qd = nearest_cholesky(QQd)
        else:
            Qd = np.zeros((self.nx, self.nx))

        return Ad, B0d, B1d, Qd

    def _lti_jacobian_disc(self, dt, dA, dB, dQ):
        '''Discretization of augmented LTI state-space model

        Args:
            dt: Sampling time
            dA: Jacobian state matrix
            dB: Jacobian input matrix
            dQ: Jacobian Wiener process scaling matrix

        Returns:
            Ad: Discrete state matrix
            B0d, B1d: Discrete input matrix
            Qd: Upper Cholesky factor process noise covariance
            dAd: Derivative discrete state matrix
            dB0d, dB1d: Derivative discrete input matrix
            dQd: Derivative upper Cholesky factor process noise covariance matrix
        '''
        N = dA.shape[0]

        if self.nu == 0:
            Ad = expm(self.A * dt)
            B0d = np.zeros((self.nx, self.nu))
            B1d = B0d

            dAd = np.zeros((N, self.nx, self.nx))
            for n in range(N):
                if np.any(dA[n]):
                    dAd[n] = expm_frechet(self.A * dt, dA[n] * dt, 'SPS', False)

            dB0d = np.zeros((N, self.nx, self.nu))
            dB1d = dB0d

        else:
            if self.hold_order == 0:

                AA = np.zeros((self.nx + self.nu, self.nx + self.nu))
                AA[: self.nx, : self.nx] = self.A
                AA[: self.nx, self.nx :] = self.B

                dAA = np.zeros((N, self.nx + self.nu, self.nx + self.nu))
                dAA[:, : self.nx, : self.nx] = dA
                dAA[:, : self.nx, self.nx :] = dB

                AAd = expm(AA * dt)
                dAAd = np.asarray(
                    [expm_frechet(AA * dt, dAA[n] * dt, 'SPS', False) for n in range(N)]
                )

                Ad, B0d = AAd[: self.nx, : self.nx], AAd[: self.nx, self.nx :]

                dAd, dB0d = dAAd[:, : self.nx, : self.nx], dAAd[:, : self.nx, self.nx :]

                if self.nu >= self.nx:
                    B1d = AA[-self.nx :, self.nx :]
                    dB1d = dAA[:, -self.nx :, self.nx :]
                else:
                    B1d = np.zeros((self.nx, self.nu))
                    dB1d = np.zeros((N, self.nx, self.nu))
            else:
                AA = np.zeros((self.nx + 2 * self.nu, self.nx + 2 * self.nu))
                AA[: self.nx, : self.nx] = self.A
                AA[: self.nx, self.nx : self.nx + self.nu] = self.B
                AA[self.nx : self.nx + self.nu, self.nx + self.nu :] = np.eye(self.nu)

                dAA = np.zeros((N, self.nx + 2 * self.nu, self.nx + 2 * self.nu))
                dAA[:, : self.nx, : self.nx] = dA
                dAA[:, : self.nx, self.nx : self.nx + self.nu] = dB

                AAd = expm(AA * dt)
                dAAd = np.asarray(
                    [expm_frechet(AA * dt, dAA[n] * dt, 'SPS', False) for n in range(N)]
                )

                Ad = AAd[: self.nx, : self.nx]
                B0d = AAd[: self.nx, self.nx : self.nx + self.nu]
                B1d = AAd[: self.nx, self.nx + self.nu :]

                dAd = dAAd[:, : self.nx, : self.nx]
                dB0d = dAAd[:, : self.nx, self.nx : self.nx + self.nu]
                dB1d = dAAd[:, : self.nx, self.nx + self.nu :]

        if np.any(self.Q):

            # transform to covariance matrix
            QQ = self.Q.T @ self.Q
            dQQ = dQ.swapaxes(1, 2) @ self.Q + self.Q.T @ dQ

            if self.method == 'mfd':
                try:
                    QQd, dQQd = self._disc_dQ_mfd(dt, QQ, dA, dQQ)
                except (LinAlgError, LinAlgWarning, RuntimeError, RuntimeWarning):
                    QQd, dQQd = self._disc_dQ_lyapunov(dt, QQ, dA, dQQ, Ad, dAd)
            elif self.method == 'lyapunov':
                try:
                    QQd, dQQd = self._disc_dQ_lyapunov(dt, QQ, dA, dQQ, Ad, dAd)
                except (LinAlgError, LinAlgWarning, RuntimeError, RuntimeWarning):
                    QQd, dQQd = self._disc_dQ_mfd(dt, QQ, dA, dQQ)
            else:
                raise ValueError('`Invalid discretization method`')

            try:
                Qd = np.linalg.cholesky(QQd).T
            except (LinAlgError, LinAlgWarning, RuntimeError, RuntimeWarning):
                Qd = nearest_cholesky(QQd)

            try:
                tmp = np.linalg.solve(Qd.T, np.linalg.solve(Qd.T, dQQd).swapaxes(1, 2))
            except (LinAlgError, LinAlgWarning, RuntimeError, RuntimeWarning):
                inv_Qd = np.linalg.pinv(Qd)
                tmp = inv_Qd.T @ dQQd @ inv_Qd

            dQd = (np.triu(tmp, 1) + np.eye(self.nx) / 2 * tmp.diagonal(0, 1, 2)[:, None, :]) @ Qd
        else:
            Qd = np.zeros((self.nx, self.nx))
            dQd = np.zeros((N, self.nx, self.nx))

        return Ad, B0d, B1d, Qd, dAd, dB0d, dB1d, dQd

    def _disc_Q_mfd(self, dt, QQ):
        """Discretization diffusion matrix by Matrix Fraction Decomposition (MFD)"""

        AA = np.zeros((2 * self.nx, 2 * self.nx))
        AA[: self.nx, : self.nx] = self.A
        AA[: self.nx, self.nx :] = QQ
        AA[self.nx :, self.nx :] = -self.A.T
        AAd = expm(AA * dt)

        return AAd[: self.nx, self.nx :] @ AAd[: self.nx, : self.nx].T

    def _disc_Q_lyapunov(self, QQ, Ad):
        """Discretization diffusion matrix by Lyapunov equation"""

        return solve_continuous_lyapunov(self.A, -QQ + Ad @ QQ @ Ad.T)

    def _disc_dQ_mfd(self, dt, QQ, dA, dQQ):
        """Discretization partial derivative of the diffusion matrix by MFD"""

        N = dA.shape[0]

        AA = np.zeros((2 * self.nx, 2 * self.nx))
        AA[: self.nx, : self.nx] = self.A
        AA[: self.nx, self.nx :] = QQ
        AA[self.nx :, self.nx :] = -self.A.T

        dAA = np.zeros((N, 2 * self.nx, 2 * self.nx))
        dAA[:, : self.nx, : self.nx] = dA
        dAA[:, : self.nx, self.nx :] = dQQ
        dAA[:, self.nx :, self.nx :] = -dA.swapaxes(1, 2)

        AAd = expm(AA * dt)
        dAAd = np.asarray([expm_frechet(AA * dt, dAA[n] * dt, 'SPS', False) for n in range(N)])

        Ad = AAd[: self.nx, : self.nx]
        QQd = AAd[: self.nx, self.nx :] @ Ad.T

        dAd = dAAd[:, : self.nx, : self.nx]
        dQQd = dAAd[:, : self.nx, self.nx :] @ Ad.T + AAd[: self.nx, self.nx :] @ dAd.swapaxes(1, 2)

        return QQd, dQQd

    def _disc_dQ_lyapunov(self, dt, QQ, dA, dQQ, Ad, dAd):
        """Discretization partial derivative of the diffusion matrix by MFD"""

        N = dA.shape[0]

        QQd = self._disc_Q_lyapunov(QQ, Ad)
        eq = (
            -dA @ QQd
            - QQd @ dA.swapaxes(1, 2)
            - dQQ
            + dAd @ QQ @ Ad.T
            + Ad @ dQQ @ Ad.T
            + Ad @ QQ @ dAd.swapaxes(1, 2)
        )
        dQQd = np.asarray([solve_continuous_lyapunov(self.A, eq[i, :, :]) for i in range(N)])

        return QQd, dQQd

    def discretization(self, dt: np.ndarray, jacobian: bool = False):
        '''Discretization of LTI state-space model

        Args:
            dt: sampling time array
            jacobian: If set to True, the jacobian are discretized

        Returns:
            idx: Index of unique time intervals
            Ad, B0d, B1d, Qd: Discrete state space matrices
            dAd, dB0d, dB1d, dQd: Derivative of discrete state space matrices
                if `jacobian` is True
        '''
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
    '''Dynamic thermal model'''

    latent_forces: str = field(default='')

    def __post_init__(self):
        super().__post_init__()

    def __repr__(self):
        return f"\n{self.__class__.__name__}" + "-" * len(self.__class__.__name__)

    def __le__(self, gp):
        '''Create a Latent Force Model'''
        from .latent_force_model import LatentForceModel

        return LatentForceModel(self, gp, self.latent_forces)


@dataclass
class GPModel(StateSpace):
    '''Gaussian Process'''

    def __post_init__(self):
        if hasattr(self, 'J'):
            self.states = self.states_block * int(self.J + 1)
        super().__post_init__()

    def __repr__(self):
        return f"\n{self.__class__.__name__}" + "-" * len(self.__class__.__name__)

    def __mul__(self, gp):
        '''Product of two Gaussian Process model

        Args:
            gp: GPModel instance

        Returns:
            product of the two GP model
        '''
        if not isinstance(gp, GPModel):
            raise TypeError('`gp` must be an GPModel instance')

        from .gaussian_process import GPProduct

        return GPProduct(self, gp)

    def __add__(self, gp):
        '''Sum of two Gaussian Process model

        Args:
            gp: GPModel instance

        Returns:
            sum of the two GP model
        '''
        if not isinstance(gp, GPModel):
            raise TypeError('`gp` must be an GPModel instance')

        from .gaussian_process import GPSum

        return GPSum(self, gp)
