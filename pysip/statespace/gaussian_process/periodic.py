from dataclasses import dataclass, field
import numpy as np
from scipy.special import iv
from ..base import GPModel


@dataclass
class Periodic(GPModel):
    """Periodic covariance function

    `iv` is the modified Bessel function of the first kind.
    Useful relation: iv(J+1, x) = iv(J-1, x) - 2*J/x * iv(J, x)

    Args:
        J: Degree of approximation (default=7)

    References:
        Arno Solin and Simo Särkkä (2014). Explicit link between periodic
        covariance functions and state space models. In Proceedings of the
        Seventeenth International Conference on Artifcial Intelligence and
        Statistics (AISTATS 2014). JMLR: W&CP, volume 33.

    """

    J: int = field(default=7)

    # The state-space is composed of J+1 states_block
    states_block = [
        ('ANY', 'f(t)', 'stochastic process'),
        ('ANY', 'df(t)/dt', 'derivative stochastic process'),
    ]

    params = [
        ('PERIOD', 'period', 'period of the function'),
        ('MAGNITUDE_SCALE', 'mscale', 'control the overall variance of the function'),
        ('LENGTH_SCALE', 'lscale', 'control the smoothness of the function'),
        ('MEASURE_DEVIATION', 'sigv', 'measurement standard deviation'),
    ]

    inputs = []

    outputs = [('ANY', 'sum(f(t))', 'sum of stochastic processes')]

    def set_constant_continuous_ssm(self):
        self.C[0, ::2] = 1.0
        self._kron = np.kron(
            np.diag(range(self.J + 1)), np.array([[0.0, -2.0 * np.pi], [2.0 * np.pi, 0.0]])
        )

    def set_constant_continuous_dssm(self):
        self.dR['sigv'][0, 0] = 1.0

    def update_continuous_ssm(self):
        period, mscale, lscale, sigv, *_ = self.parameters.theta

        q2 = 2.0 * mscale ** 2 * np.exp(-(lscale ** (-2))) * iv(range(self.J + 1), lscale ** (-2))
        q2[0] *= 0.5

        if not np.all(np.isfinite(q2)):
            raise ValueError('Spectral variance coefficients are not finite!')

        self.A[:] = self._kron / period
        self.R[0, 0] = sigv
        self.P0[self._diag] = np.repeat(np.sqrt(q2), 2)

    def update_continuous_dssm(self):
        period, mscale, lscale, *_ = self.parameters.theta

        q2 = 2.0 * mscale ** 2 * np.exp(-(lscale ** (-2))) * iv(range(self.J + 1), lscale ** (-2))
        q2[0] *= 0.5
        q = np.sqrt(q2)

        dq2 = np.empty(int(self.J + 1))
        dq2[:] = mscale ** 2 * lscale ** (-3) * np.exp(-(lscale ** (-2)))
        dq2[0] *= 2.0 * (iv(0, lscale ** (-2)) - iv(1, lscale ** (-2)))
        dq2[1:] *= -4.0 * iv(range(self.J), lscale ** (-2)) + 4.0 * (
            1.0 + np.arange(1, self.J + 1) / (lscale ** (-2))
        ) * iv(range(1, self.J + 1), lscale ** (-2))

        if not np.all(np.isfinite(dq2)):
            raise ValueError('Derivative of spectral variance coefficients are not finite!')

        dql = 0.5 / q * dq2
        dqm = 0.5 / q * q2 * 2.0 / mscale

        self.dA['period'] = self._kron / -(period ** 2)
        self.dP0['lscale'][self._diag] = np.repeat(dql, 2)
        self.dP0['mscale'][self._diag] = np.repeat(dqm, 2)
