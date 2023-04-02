from dataclasses import dataclass

import numpy as np

from pysip.utils.math import diff_upper_cholesky, nearest_cholesky

from ..base_statespace import GPModel


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