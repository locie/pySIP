from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.linalg import cond

from ..utils.math import nearest_cholesky
from .base_statespace import GPModel, RCModel, StateSpace
from . import discretization
from .nodes import Par


@dataclass
class LatentForceModel(StateSpace):
    """Latent Force Model (LFM)

    Args:
        rc: RCModel() gp: GPModel() latent_force: The name of the input considered as
        the latent force

    Notes:
        The MEASURE_DEVIATION of the GPModel is fixed because it is not used in the
        latent force model. The GPModel is augmented into the RCModel, therefore, only
        the measurement noise matrix `R` of the RCModel is used. To avoid useless
        computation, the MEASURE_DEVIATION of the GPModel must stay fixed.
    """

    def __init__(self, rc: RCModel, gp: GPModel, latent_force: str):
        if not isinstance(rc, RCModel):
            raise TypeError("`rc` must be an RCModel instance")

        if not isinstance(gp, GPModel):
            raise TypeError("`gp` must be an GPModel instance")

        if not isinstance(latent_force, (str, list)):
            raise TypeError("`latent_force` must be a string or a list of strings")

        if isinstance(latent_force, str):
            latent_force = [latent_force]

        if not rc.inputs:
            raise ValueError("The `rc` model has no input")

        index = []
        for i, node in enumerate(rc.inputs):
            if node.name in latent_force:
                index.append(i)
        if not index:
            raise ValueError("`latent_force` is not an input of `rc`")

        self._rc = rc
        self._gp = gp

        # The GP have only one output
        for i, node in enumerate(self._gp.params):
            if node.category == Par.MEASURE_DEVIATION:
                self._gp.parameters.set_parameter(
                    node.name, value=0.0, transform="fixed"
                )
                break

        self.parameters = self._rc.parameters + self._gp.parameters
        self.states = self._rc.states + self._gp.states

        self.params = []
        for node in self._rc.params:
            node.name = self._rc.name + "__" + node.name
            self.params.append(node.unpack())
        for node in self._gp.params:
            node.name = self._gp.name + "__" + node.name
            self.params.append(node.unpack())

        self.inputs = self._rc.inputs
        for i in sorted(index, reverse=True):
            del self.inputs[i]
        self.outputs = self._rc.outputs

        # Unpack list of Node as a list of string to rebuild list of Node
        self.states = [s.unpack() for s in self.states]
        self.inputs = [s.unpack() for s in self.inputs]
        self.outputs = [s.unpack() for s in self.outputs]

        # slicing columns in the input matrix corresponding to latent forces
        self.idx = np.full((self._rc.nu), False)
        self.idx[index] = True

        self.hold_order = self._rc.hold_order

        self.name = (
            self._rc.name + "__" + "latent_forces(" + self._rc.latent_forces + ")"
        )

        super().__post_init__()

    def set_constant_continuous_ssm(self):
        self._rc.set_constant_continuous_ssm()
        self._gp.set_constant_continuous_ssm()

    def update_continuous_ssm(self):
        self._rc.update_continuous_ssm()
        self._gp.update_continuous_ssm()

        self.A[: self._rc.nx, : self._rc.nx] = self._rc.A
        self.A[self._rc.nx :, self._rc.nx :] = self._gp.A
        self.A[: self._rc.nx, self._rc.nx :] = self._rc.B[:, self.idx] @ self._gp.C

        self.B[: self._rc.nx, :] = self._rc.B[:, ~self.idx]

        self.C[:, : self._rc.nx] = self._rc.C

        self.D = self._rc.D[:, ~self.idx]

        self.Q[: self._rc.nx, : self._rc.nx] = self._rc.Q
        self.Q[self._rc.nx :, self._rc.nx :] = self._gp.Q

        self.R = self._rc.R

        self.x0[: self._rc.nx] = self._rc.x0
        self.x0[self._rc.nx :] = self._gp.x0

        self.P0[: self._rc.nx, : self._rc.nx] = self._rc.P0
        self.P0[self._rc.nx :, self._rc.nx :] = self._gp.P0

    def _lti_disc(
        self, dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Discretization of Linear Time Invariant Latent Force Model

        Given the block upper triangular form of the state matrix, the Parlett's method
        is used for computing the discrete state matrix.

        Args:
            dt: sampling time

        Returns:
            4-elements tuple containing
                - **Ad**: Discrete state matrix
                - **B0d**: Discrete input matrix (zero order hold)
                - **B1d**: Discrete input matrix (first order hold)
                - **Qd**: Upper Cholesky factor of the process noise covariance
        """
        if cond(self._rc.A, "fro") < 1e12:
            method = "analytic"
        else:
            method = "expm"

        F11, G0, G1 = discretization.state_input(
            self._rc.A, self.B[: self._rc.nx, :], dt, self.hold_order, method
        )

        F22 = discretization.state(self._gp.A, dt)
        A12 = self.A[: self._rc.nx, self._rc.nx :]
        Ad = discretization.expm_triu(self._rc.A, A12, self._gp.A, dt, F11, F22)

        Oxu = np.zeros((self._gp.nx, self.nu))
        B0d = np.vstack([G0, Oxu])
        B1d = np.vstack([G1, Oxu])

        if np.all(np.real(np.linalg.eigvals(self.A)) < 0):
            Q = discretization.diffusion_lyap(self.A, self.Q.T @ self.Q, Ad)
        else:
            Q = discretization.diffusion_mfd(self.A, self.Q.T @ self.Q, dt)

        return Ad, B0d, B1d, nearest_cholesky(Q)


@dataclass
class R2C2_Qgh_Matern32(RCModel):
    """LFM: R2C2_Qgh + Matern32

    The Matérn kernel with smoothness parameter = 3/2 is used to model the
    the solar radiation as a latent force in the model R2C2.

    This LFM is built by hand and allows to check the creation of LFM in
    bopt.Latent_force_model.
    """

    states = [
        ("TEMPERATURE", "xw", "wall temperature"),
        ("TEMPERATURE", "xi", "indoor space temperature"),
        ("ANY", "f(t)", "stochastic process"),
        ("ANY", "df(t)/dt", "derivative stochastic process"),
    ]

    params = [
        ("THERMAL_RESISTANCE", "R2C2Qgh__Ro", "between the outdoor and the wall node"),
        ("THERMAL_RESISTANCE", "R2C2Qgh__Ri", "between the wall node and the indoor"),
        ("THERMAL_CAPACITY", "R2C2Qgh__Cw", "Wall"),
        (
            "THERMAL_CAPACITY",
            "R2C2Qgh__Ci",
            "indoor air, indoor walls, furnitures, etc. ",
        ),
        ("STATE_DEVIATION", "R2C2Qgh__sigw_w", ""),
        ("STATE_DEVIATION", "R2C2Qgh__sigw_i", ""),
        ("MEASURE_DEVIATION", "R2C2Qgh__sigv", ""),
        ("INITIAL_MEAN", "R2C2Qgh__x0_w", ""),
        ("INITIAL_MEAN", "R2C2Qgh__x0_i", ""),
        ("INITIAL_DEVIATION", "R2C2Qgh__sigx0_w", ""),
        ("INITIAL_DEVIATION", "R2C2Qgh__sigx0_i", ""),
        ("MAGNITUDE_SCALE", "Matern32__mscale", ""),
        ("LENGTH_SCALE", "Matern32__lscale", ""),
    ]

    inputs = [
        ("TEMPERATURE", "To", "outdoor air temperature"),
        ("POWER", "Qh", "HVAC system heat"),
    ]

    outputs = [("TEMPERATURE", "xi", "indoor air temperature")]

    def set_constant_continuous_ssm(self):
        self.C[0, 1] = 1.0

    def update_continuous_ssm(self):
        (
            Ro,
            Ri,
            Cw,
            Ci,
            sigw_w,
            sigw_i,
            sigv,
            x0_w,
            x0_i,
            sigx0_w,
            sigx0_i,
            mscale,
            lscale,
            *_,
        ) = self.parameters.theta

        self.A[:] = [
            [-(Ro + Ri) / (Cw * Ri * Ro), 1.0 / (Cw * Ri), 1.0 / Cw, 0.0],
            [1.0 / (Ci * Ri), -1.0 / (Ci * Ri), 1.0 / Ci, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, -3.0 / lscale**2, -2.0 * 3.0**0.5 / lscale],
        ]
        self.B[:2, :] = [[1.0 / (Cw * Ro), 0.0], [0.0, 1.0 / Ci]]
        self.Q[self._diag] = [
            sigw_w,
            sigw_i,
            0.0,
            2.0 * 3.0**0.75 * mscale / lscale**1.5,
        ]
        self.R[0, 0] = sigv
        self.x0[:2, 0] = [x0_w, x0_i]
        self.P0[self._diag] = [sigx0_w, sigx0_i, mscale, 3.0**0.5 * mscale / lscale]
