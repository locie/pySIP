import numpy as np
from .base import StateSpace


class Resonator(StateSpace):
    """Resonator model for periodic time series"""

    states = [
        ('ANY', 'f(t)', 'stochastic process'),
        ('ANY', 'df(t)/dt', 'derivative stochastic process'),
    ]

    params = [
        ('COEFFICIENT', 'freq', ''),
        ('COEFFICIENT', 'damp', ''),
        ('STATE_DEVIATION', 'sigw', ''),
        ('MEASURE_DEVIATION', 'sigv', ''),
        ('INITIAL_MEAN', 'x0_f', ''),
        ('INITIAL_MEAN', 'x0_df', ''),
        ('INITIAL_DEVIATION', 'sigx0_f', ''),
        ('INITIAL_DEVIATION', 'sigx0_df', ''),
    ]

    inputs = []

    outputs = [('ANY', 'f(t)', 'stochastic process')]

    def __post_init__(self):
        super().__post_init__()

    def set_constant_continuous_ssm(self):
        self.C[0, 0] = 1.0

    def set_constant_continuous_dssm(self):
        self.dA['freq'][0, 1] = 2.0 * np.pi
        self.dA['freq'][1, 0] = -2.0 * np.pi
        self.dA['damp'][1, 1] = -1.0
        self.dQ['sigw'][1, 1] = 1.0
        self.dR['sigv'][0, 0] = 1.0
        self.dx0['x0_f'][0, 0] = 1.0
        self.dx0['x0_df'][1, 0] = 1.0
        self.dP0['sigx0_f'][0, 0] = 1.0
        self.dP0['sigx0_df'][1, 1] = 1.0

    def update_continuous_ssm(self):
        freq, damp, sigw, sigv, x0_f, x0_df, sigx0_f, sigx0_df, *_ = self.parameters.theta

        self.A[:] = np.array([[0.0, 2.0 * np.pi * freq], [-2.0 * np.pi * freq, -damp]])
        self.Q[1, 1] = sigw
        self.R[0, 0] = sigv
        self.x0[:, 0] = [x0_f, x0_df]
        self.P0[self._diag] = [sigx0_f, sigx0_df]
