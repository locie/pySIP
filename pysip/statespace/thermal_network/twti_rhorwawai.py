from dataclasses import dataclass

from ..base import RCModel


@dataclass
class TwTi_rhoRwAwAi(RCModel):
    """Second order RC model"""

    states = [
        ('TEMPERATURE', 'xw', 'wall temperature'),
        ('TEMPERATURE', 'xi', 'indoor space temperature'),
    ]

    params = [
        ('THERMAL_RESISTANCE', 'Rw', 'between the outdoor and the indoor'),
        ('COEFFICIENT', 'rho', 'symmetric coefficient of the wall'),
        ('THERMAL_CAPACITY', 'Cw', 'Wall'),
        ('THERMAL_CAPACITY', 'Ci', 'Indoor'),
        ('SOLAR_APERTURE', 'Aw', 'of the wall (m2)'),
        ('SOLAR_APERTURE', 'Ai', 'of the windows (m2)'),
        ('STATE_DEVIATION', 'sigw_w', ''),
        ('STATE_DEVIATION', 'sigw_i', ''),
        ('MEASURE_DEVIATION', 'sigv', ''),
        ('INITIAL_MEAN', 'x0_w', ''),
        ('INITIAL_MEAN', 'x0_i', ''),
        ('INITIAL_DEVIATION', 'sigx0_w', ''),
        ('INITIAL_DEVIATION', 'sigx0_i', ''),
    ]

    inputs = [
        ('TEMPERATURE', 'To', 'outdoor air temperature'),
        ('POWER', 'Qgh', 'global horizontal solar radiation'),
        ('POWER', 'Qh', 'HVAC system heat'),
    ]

    outputs = [('TEMPERATURE', 'xi', 'indoor air temperature')]

    def __post_init__(self):
        super().__post_init__()

    def set_constant_continuous_ssm(self):
        self.C[0, 1] = 1.0

    def set_constant_continuous_dssm(self):
        self.dQ['sigw_w'][0, 0] = 1.0
        self.dQ['sigw_i'][1, 1] = 1.0
        self.dR['sigv'][0, 0] = 1.0
        self.dx0['x0_w'][0, 0] = 1.0
        self.dx0['x0_i'][1, 0] = 1.0
        self.dP0['sigx0_w'][0, 0] = 1.0
        self.dP0['sigx0_i'][1, 1] = 1.0

    def update_continuous_ssm(self):
        (
            Rw,
            rho,
            Cw,
            Ci,
            Aw,
            Ai,
            sigw_w,
            sigw_i,
            sigv,
            x0_w,
            x0_i,
            sigx0_w,
            sigx0_i,
            *_,
        ) = self.parameters.theta

        self.A[:] = [
            [1.0 / (Cw * Rw * rho * (rho - 1.0)), -1.0 / (Cw * Rw * (rho - 1.0))],
            [-1.0 / (Ci * Rw * (rho - 1.0)), 1.0 / (Ci * Rw * (rho - 1.0))],
        ]
        self.B[:] = [[1.0 / (Cw * Rw * rho), Aw / Cw, 0.0], [0.0, Ai / Ci, 1.0 / Ci]]
        self.Q[self._diag] = [sigw_w, sigw_i]
        self.R[0, 0] = sigv
        self.x0[:, 0] = [x0_w, x0_i]
        self.P0[self._diag] = [sigx0_w, sigx0_i]

    def update_continuous_dssm(self):
        Rw, rho, Cw, Ci, Aw, Ai, *_ = self.parameters.theta

        self.dA['Rw'][:] = [
            [-1.0 / (Cw * Rw ** 2 * rho * (rho - 1.0)), 1.0 / (Cw * Rw ** 2 * (rho - 1.0))],
            [1.0 / (Ci * Rw ** 2 * (rho - 1.0)), -1.0 / (Ci * Rw ** 2 * (rho - 1.0))],
        ]
        self.dA['rho'][:] = [
            [
                (1.0 - 2.0 * rho) / (Cw * Rw * rho ** 2 * (rho - 1.0) ** 2),
                1.0 / (Cw * Rw * (rho - 1.0) ** 2),
            ],
            [1.0 / (Ci * Rw * (rho - 1.0) ** 2), -1.0 / (Ci * Rw * (rho - 1.0) ** 2)],
        ]

        self.dA['Cw'][0, :] = [
            -1.0 / (Cw ** 2 * Rw * rho * (rho - 1.0)),
            1.0 / (Cw ** 2 * Rw * (rho - 1.0)),
        ]
        self.dA['Ci'][1, :] = [
            1.0 / (Ci ** 2 * Rw * (rho - 1.0)),
            -1.0 / (Ci ** 2 * Rw * (rho - 1.0)),
        ]

        self.dB['Rw'][0, 0] = -1.0 / (Cw * Rw ** 2 * rho)
        self.dB['rho'][0, 0] = -1.0 / (Cw * Rw * rho ** 2)
        self.dB['Cw'][0, :2] = [-1.0 / (Cw ** 2 * Rw * rho), -Aw / Cw ** 2]
        self.dB['Ci'][1, 1:] = [-Ai / Ci ** 2, -1.0 / Ci ** 2]
        self.dB['Aw'][0, 1] = 1.0 / Cw
        self.dB['Ai'][1, 1] = 1.0 / Ci
