from dataclasses import dataclass
from ..base import GPModel


@dataclass
class Matern12(GPModel):
    '''Matérn covariance function with smoothness parameter = 1/2'''

    states = [('ANY', 'f(t)', 'stochastic process')]

    params = [
        ('MAGNITUDE_SCALE', 'mscale', 'control the overall variance of the function'),
        ('LENGTH_SCALE', 'lscale', 'control the smoothness of the function'),
        ('MEASURE_DEVIATION', 'sigv', 'measurement standard deviation'),
    ]

    inputs = []

    outputs = [('ANY', 'f(t)', 'stochastic process')]

    def set_constant_continuous_ssm(self):
        self.C[0, 0] = 1.0

    def set_constant_continuous_dssm(self):
        self.dR['sigv'][0, 0] = 1.0
        self.dP0['mscale'][:] = 1.0

    def update_continuous_ssm(self):
        mscale, lscale, sigv, *_ = self.parameters.theta

        self.A[:] = -1.0 / lscale
        self.Q[:] = 2.0 ** 0.5 * mscale / lscale ** 0.5
        self.R[:] = sigv
        self.P0[:] = mscale

    def update_continuous_dssm(self):
        mscale, lscale, *_ = self.parameters.theta

        self.dA['lscale'][:] = 1.0 / lscale ** 2
        self.dQ['mscale'][:] = 2.0 ** 0.5 / lscale ** 0.5
        self.dQ['lscale'][:] = -(2.0 ** 0.5) * mscale / (2.0 * lscale ** 1.5)


@dataclass
class Matern32(GPModel):
    '''Matérn covariance function with smoothness parameter = 3/2'''

    states = [
        ('ANY', 'f(t)', 'stochastic process'),
        ('ANY', 'df(t)/dt', 'derivative stochastic process'),
    ]

    params = [
        ('MAGNITUDE_SCALE', 'mscale', 'control the overall variance of the function'),
        ('LENGTH_SCALE', 'lscale', 'control the smoothness of the function'),
        ('MEASURE_DEVIATION', 'sigv', 'measurement standard deviation'),
    ]

    inputs = []

    outputs = [('ANY', 'f(t)', 'stochastic process')]

    def set_constant_continuous_ssm(self):
        self.A[0, 1] = 1.0
        self.C[0, 0] = 1.0

    def set_constant_continuous_dssm(self):
        self.dP0['mscale'][0, 0] = 1.0
        self.dR['sigv'][0, 0] = 1.0

    def update_continuous_ssm(self):
        mscale, lscale, sigv, *_ = self.parameters.theta

        self.A[1, :] = [-3.0 / lscale ** 2, -2.0 * 3.0 ** 0.5 / lscale]
        self.Q[1, 1] = 2.0 * 3.0 ** 0.75 * mscale / lscale ** 1.5
        self.R[0, 0] = sigv
        self.P0[self._diag] = [mscale, 3.0 ** 0.5 * mscale / lscale]

    def update_continuous_dssm(self):
        mscale, lscale, *_ = self.parameters.theta

        self.dA['lscale'][1, :] = [6.0 / lscale ** 3, 2.0 * 3.0 ** 0.5 / lscale ** 2]

        self.dQ['mscale'][1, 1] = 2.0 * 3.0 ** 0.75 / lscale ** 1.5
        self.dQ['lscale'][1, 1] = -(3.0 ** 1.75) * mscale / lscale ** 2.5

        self.dP0['mscale'][1, 1] = 3.0 ** 0.5 / lscale
        self.dP0['lscale'][1, 1] = -(3.0 ** 0.5) * mscale / lscale ** 2


@dataclass
class Matern52(GPModel):
    '''Matérn covariance function with smoothness parameter = 5/2'''

    states = [
        ('ANY', 'f(t)', 'stochastic process'),
        ('ANY', 'df(t)/dt', 'derivative stochastic process'),
        ('ANY', 'd²f(t)/d²t', 'second derivative stochastic process'),
    ]

    params = [
        ('MAGNITUDE_SCALE', 'mscale', 'control the overall variance of the function'),
        ('LENGTH_SCALE', 'lscale', 'control the smoothness of the function'),
        ('MEASURE_DEVIATION', 'sigv', 'measurement standard deviation'),
    ]

    inputs = []

    outputs = [('ANY', 'f(t)', 'stochastic process')]

    def set_constant_continuous_ssm(self):
        self.A[0, 1] = 1.0
        self.A[1, 2] = 1.0
        self.C[0, 0] = 1.0

    def set_constant_continuous_dssm(self):
        self.dR['sigv'][0, 0] = 1.0

    def update_continuous_ssm(self):
        mscale, lscale, sigv, *_ = self.parameters.theta

        self.A[2, :] = [
            -(5.0 ** 1.5) / lscale ** 3,
            -15.0 / lscale ** 2,
            -3.0 * 5.0 ** 0.5 / lscale,
        ]
        self.Q[2, 2] = 20.0 * 5.0 ** 0.25 / 3.0 ** 0.5 * mscale / lscale ** 2.5
        self.R[0, 0] = sigv
        tmp = (5.0 / 3.0) ** 0.5 * mscale / lscale
        self.P0[:] = [[mscale, 0.0, -tmp], [0.0, tmp, 0.0], [-tmp, 0.0, 5.0 * mscale / lscale ** 2]]

    def update_continuous_dssm(self):
        mscale, lscale, *_ = self.parameters.theta

        tmp = (5.0 / 3.0) ** 0.5 * mscale / lscale

        self.dA['lscale'][2, :] = [
            3.0 * 5.0 ** 1.5 / lscale ** 4,
            30.0 / lscale ** 3,
            3.0 * 5.0 ** 0.5 / lscale ** 2,
        ]

        self.dQ['mscale'][2, 2] = self.Q[2, 2] / mscale

        self.dQ['lscale'][2, 2] = -2.5 * self.Q[2, 2] / lscale

        self.dP0['mscale'][:] = [
            [1.0, 0.0, -tmp / mscale],
            [0.0, tmp / mscale, 0.0],
            [-tmp / mscale, 0.0, 5.0 / lscale ** 2],
        ]

        self.dP0['lscale'][:] = [
            [0.0, 0.0, tmp / lscale],
            [0.0, -tmp / lscale, 0.0],
            [tmp / lscale, 0.0, -10.0 * mscale / lscale ** 3],
        ]
