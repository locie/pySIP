from dataclasses import dataclass

import numpy as np

from pysip.utils.draw import TikzStateSpace
from pysip.utils.math import diff_upper_cholesky, nearest_cholesky

from ..base import GPModel
from ..discretization import disc_d_diffusion_stationary, disc_diffusion_stationary


@dataclass
class Matern12(GPModel):
    """Matérn covariance function with smoothness parameter = 1/2"""

    states = [("ANY", "f(t)", "stochastic process")]

    params = [
        ("MAGNITUDE_SCALE", "mscale", "control the overall variance of the function"),
        ("LENGTH_SCALE", "lscale", "control the smoothness of the function"),
        ("MEASURE_DEVIATION", "sigv", "measurement standard deviation"),
    ]

    inputs = []

    outputs = [("ANY", "f(t)", "stochastic process")]

    def set_constant_continuous_ssm(self):
        self.C[0, 0] = 1.0

    def set_constant_continuous_dssm(self):
        self.dR["sigv"][0, 0] = 1.0
        self.dP0["mscale"][:] = 1.0

    def update_continuous_ssm(self):
        mscale, lscale, sigv, *_ = self.parameters.theta

        self.A[:] = -1.0 / lscale
        self.Q[:] = 2.0**0.5 * mscale / lscale**0.5
        self.R[:] = sigv
        self.P0[:] = mscale

    def update_continuous_dssm(self):
        mscale, lscale, *_ = self.parameters.theta

        self.dA["lscale"][:] = 1.0 / lscale**2
        self.dQ["mscale"][:] = 2.0**0.5 / lscale**0.5
        self.dQ["lscale"][:] = -(2.0**0.5) * mscale / (2.0 * lscale**1.5)


@dataclass
class Matern32(GPModel):
    """Matérn covariance function with smoothness parameter = 3/2"""

    states = [
        ("ANY", "f(t)", "stochastic process"),
        ("ANY", "df(t)/dt", "derivative stochastic process"),
    ]

    params = [
        ("MAGNITUDE_SCALE", "mscale", "control the overall variance of the function"),
        ("LENGTH_SCALE", "lscale", "control the smoothness of the function"),
        ("MEASURE_DEVIATION", "sigv", "measurement standard deviation"),
    ]

    inputs = []

    outputs = [("ANY", "f(t)", "stochastic process")]

    def set_constant_continuous_ssm(self):
        self.A[0, 1] = 1.0
        self.C[0, 0] = 1.0

    def set_constant_continuous_dssm(self):
        self.dP0["mscale"][0, 0] = 1.0
        self.dR["sigv"][0, 0] = 1.0

    def update_continuous_ssm(self):
        mscale, lscale, sigv, *_ = self.parameters.theta

        self.A[1, :] = [-3.0 / lscale**2, -2.0 * 3.0**0.5 / lscale]
        self.Q[1, 1] = 2.0 * 3.0**0.75 * mscale / lscale**1.5
        self.R[0, 0] = sigv
        self.P0[self._diag] = [mscale, 3.0**0.5 * mscale / lscale]

    def update_continuous_dssm(self):
        mscale, lscale, *_ = self.parameters.theta

        self.dA["lscale"][1, :] = [6.0 / lscale**3, 2.0 * 3.0**0.5 / lscale**2]

        self.dQ["mscale"][1, 1] = 2.0 * 3.0**0.75 / lscale**1.5
        self.dQ["lscale"][1, 1] = -(3.0**1.75) * mscale / lscale**2.5

        self.dP0["mscale"][1, 1] = 3.0**0.5 / lscale
        self.dP0["lscale"][1, 1] = -(3.0**0.5) * mscale / lscale**2

    def _lti_disc(self, dt):
        mscale, lscale, *_ = self.parameters.theta

        lda = 3**0.5 / lscale
        _exp = np.exp(-dt * lda)
        Ad = np.array(
            [
                [(1.0 + dt * lda) * _exp, dt * _exp],
                [-dt * lda**2 * _exp, (1.0 - dt * lda) * _exp],
            ]
        )

        q00 = mscale**2 * (
            1.0 - (dt**2 * lda**2 + (1.0 + dt * lda) ** 2) * _exp**2
        )
        q01 = 2.0 * mscale**2 * dt**2 * lda**3 * _exp**2
        q11 = mscale**2 * (
            lda**2
            - (dt**2 * lda**4 + lda**2 * (1.0 - dt * lda) ** 2) * _exp**2
        )

        B0d = np.zeros((self.nx, self.nu))
        Qd = nearest_cholesky(np.array([[q00, q01], [q01, q11]]))

        return Ad, B0d, B0d, Qd

    def _lti_jacobian_disc(self, dt, dA, dB, dPinf_upper):
        nj = dA.shape[0]

        mscale, lscale, *_ = self.parameters.theta

        lda = 3**0.5 / lscale
        _exp = np.exp(-dt * lda)
        Ad = np.array(
            [
                [(1.0 + dt * lda) * _exp, dt * _exp],
                [-dt * lda**2 * _exp, (1.0 - dt * lda) * _exp],
            ]
        )

        dAd = np.zeros((nj, self.nx, self.nx))
        e_over_l = _exp / lscale
        dAd[1, :, :] = [
            [lda**2 * dt**2 * e_over_l, lda * dt**2 * e_over_l],
            [
                3.0 * dt * e_over_l * (-dt * lda + 2.0) / lscale**2,
                dt * e_over_l * (2.0 * lda - 3.0 * dt / lscale**2),
            ],
        ]

        B0d = np.zeros((self.nx, self.nu))
        dB0d = np.zeros((nj, self.nx, self.nu))

        q00 = mscale**2 * (
            1.0 - (dt**2 * lda**2 + (1.0 + dt * lda) ** 2) * _exp**2
        )
        q01 = 2.0 * mscale**2 * dt**2 * lda**3 * _exp**2
        q11 = mscale**2 * (
            lda**2
            - (dt**2 * lda**4 + lda**2 * (1.0 - dt * lda) ** 2) * _exp**2
        )

        _exp2 = np.exp(-2.0 * 3**0.5 * dt / lscale)
        # mscale
        dq00_dm = (
            2.0
            * mscale
            * (
                -(3.0 * dt**2 / lscale**2 + (3**0.5 * dt / lscale + 1.0) ** 2)
                * _exp2
                + 1.0
            )
        )
        dq01_dm = 12.0 * 3**0.5 * dt**2 * mscale * _exp2 / lscale**3
        dq11_dm = (
            2.0
            * mscale
            * (
                -(
                    9.0 * dt**2 / lscale**4
                    + 3.0 * (-(3**0.5) * dt / lscale + 1.0) ** 2 / lscale**2
                )
                * _exp2
                + 3.0 / lscale**2
            )
        )

        # lscale
        dq00_dl = -12.0 * 3**0.5 * dt**3 * mscale**2 * _exp2 / lscale**4
        dq01_dl = (
            18.0
            * dt**2
            * mscale**2
            * (2.0 * dt - 3**0.5 * lscale)
            * _exp2
            / lscale**5
        )
        dq11_dl = mscale**2 * (
            2.0
            * 3**0.5
            * dt
            * (
                -9.0 * dt**2 / lscale**4
                - 3.0 * (-(3**0.5) * dt / lscale + 1.0) ** 2 / lscale**2
            )
            * _exp2
            / lscale**2
            + (
                36.0 * dt**2 / lscale**5
                - 6.0 * 3**0.5 * dt * (-(3**0.5) * dt / lscale + 1) / lscale**4
                + 6.0 * (-(3**0.5) * dt / lscale + 1.0) ** 2 / lscale**3
            )
            * _exp2
            - 6.0 / lscale**3
        )

        dQcd = np.zeros((nj, self.nx, self.nx))
        dQcd[0, :, :] = [[dq00_dm, dq01_dm], [dq01_dm, dq11_dm]]
        dQcd[1, :, :] = [[dq00_dl, dq01_dl], [dq01_dl, dq11_dl]]

        Qd = nearest_cholesky(np.array([[q00, q01], [q01, q11]]))
        dQd = np.zeros((nj, self.nx, self.nx))
        dQd[0, :, :] = diff_upper_cholesky(Qd, dQcd[0])
        dQd[1, :, :] = diff_upper_cholesky(Qd, dQcd[1])

        return Ad, B0d, B0d, Qd, dAd, dB0d, dB0d, dQd


@dataclass
class Matern52(GPModel):
    """Matérn covariance function with smoothness parameter = 5/2"""

    states = [
        ("ANY", "f(t)", "stochastic process"),
        ("ANY", "df(t)/dt", "derivative stochastic process"),
        ("ANY", "d²f(t)/d²t", "second derivative stochastic process"),
    ]

    params = [
        ("MAGNITUDE_SCALE", "mscale", "control the overall variance of the function"),
        ("LENGTH_SCALE", "lscale", "control the smoothness of the function"),
        ("MEASURE_DEVIATION", "sigv", "measurement standard deviation"),
    ]

    inputs = []

    outputs = [("ANY", "f(t)", "stochastic process")]

    def set_constant_continuous_ssm(self):
        self.A[0, 1] = 1.0
        self.A[1, 2] = 1.0
        self.C[0, 0] = 1.0

    def set_constant_continuous_dssm(self):
        self.dR["sigv"][0, 0] = 1.0

    def update_continuous_ssm(self):
        mscale, lscale, sigv, *_ = self.parameters.theta

        self.A[2, :] = [
            -(5.0**1.5) / lscale**3,
            -15.0 / lscale**2,
            -3.0 * 5.0**0.5 / lscale,
        ]
        self.Q[2, 2] = 20.0 * 5.0**0.25 / 3.0**0.5 * mscale / lscale**2.5
        self.R[0, 0] = sigv
        # Upper triangular Cholesky factor
        self.P0[:] = [
            [mscale, 0.0, -5.0 * mscale / (3.0 * lscale**2)],
            [0, 15**0.5 * mscale / (3.0 * lscale), 0.0],
            [0.0, 0.0, 10 * 2**0.5 * mscale / (3.0 * lscale**2)],
        ]

    def update_continuous_dssm(self):
        mscale, lscale, *_ = self.parameters.theta

        self.dA["lscale"][2, :] = [
            3.0 * 5.0**1.5 / lscale**4,
            30.0 / lscale**3,
            3.0 * 5.0**0.5 / lscale**2,
        ]

        self.dQ["mscale"][2, 2] = self.Q[2, 2] / mscale

        self.dQ["lscale"][2, 2] = -2.5 * self.Q[2, 2] / lscale

        self.dP0["mscale"][:] = [
            [1.0, 0.0, -5.0 / (3.0 * lscale**2)],
            [0.0, 15**0.5 / (3.0 * lscale), 0.0],
            [0.0, 0.0, 10 * 2**0.5 / (3.0 * lscale**2)],
        ]

        self.dP0["lscale"][:] = [
            [0.0, 0.0, 10.0 * mscale / (3.0 * lscale**3)],
            [0.0, -(15**0.5) * mscale / (3.0 * lscale**2), 0.0],
            [0.0, 0.0, -20.0 * 2.0**0.5 * mscale / (3.0 * lscale**3)],
        ]
