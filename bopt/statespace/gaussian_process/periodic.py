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
        J: Degree of approximation (default=6)

    References:
        Arno Solin and Simo Särkkä (2014). Explicit link between periodic
        covariance functions and state space models. In Proceedings of the
        Seventeenth International Conference on Artifcial Intelligence and
        Statistics (AISTATS 2014). JMLR: W&CP, volume 33.

    """
    J: int = field(default=6)

    # The state-space is composed of J+1 states_block
    states_block = [
        ('ANY', 'f(t)', 'stochastic process'),
        ('ANY', 'df(t)/dt', 'derivative stochastic process')
    ]

    params = [
        ('PERIOD', 'period', ''),
        ('MAGNITUDE_SCALE', 'mscale', ''),
        ('LENGTH_SCALE', 'lscale', ''),
        ('MEASURE_DEVIATION', 'sigv', '')
    ]

    inputs = []

    outputs = [
        ('ANY', 'sum(f(t))', 'sum of stochastic processes')
    ]

    def set_constant(self):
        self.C[0, ::2] = 1.0
        self._kron = np.kron(np.diag(range(self.J + 1)),
                             np.array([[0.0, -2.0 * np.pi], [2.0 * np.pi, 0.0]]))

    def set_jacobian(self):
        self.dR["sigv"][0, 0] = 1.0

    def update_state_space_model(self):
        period, mscale, lscale, sigv, *_ = self.parameters.theta

        self._tmp = 2.0 * mscale**2 * np.exp(-lscale**(-2))
        self._q2 = self._tmp * iv(range(self.J + 1), lscale**(-2))
        self._q2[0] *= 0.5

        if not np.all(np.isfinite(self._q2)):
            raise ValueError("Spectral variance coefficients are not finite!")
        self._q = np.sqrt(self._q2)

        self.A[:] = self._kron / period
        self.R[0, 0] = sigv
        self.P0[self._diag] = np.repeat(self._q, 2)

    def update_jacobian(self):
        period, mscale, lscale, *_ = self.parameters.theta

        # index 0
        dq20_lscale = self._tmp * lscale**(-3)
        dq20_lscale *= (iv(0, lscale**(-2)) - iv(1, lscale**(-2)))

        # index 1 --> J
        dq2J_lscale = self._tmp * lscale**(-3) * 2.0
        dq2J_lscale *= ((1.0 + np.arange(1, self.J + 1) * lscale**2)
                        * iv(range(1, self.J + 1), lscale**(-2))
                        - iv(range(0, self.J), lscale**(-2)))
        dq2_lscale = np.append(dq20_lscale, dq2J_lscale)

        if not np.all(np.isfinite(dq2_lscale)):
            raise ValueError("Derivative of spectral variance "
                             "coefficients are not finite!")

        dq_lscale = 0.5 / self._q * dq2_lscale

        dq_mscale = 0.5 / self._q * self._q2 * 2.0 / mscale

        self.dA["period"] = self._kron / -period**2

        self.dP0["lscale"][self._diag] = np.repeat(dq_lscale, 2)

        self.dP0["mscale"][self._diag] = np.repeat(dq_mscale, 2)
