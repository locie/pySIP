from dataclasses import dataclass

from ..base import RCModel


@dataclass
class TwTiTm_rhoRwRmalphaAwcv(RCModel):
    """Second order RC model"""

    states = [
        ('TEMPERATURE', 'xw', 'wall temperature'),
        ('TEMPERATURE', 'xi', 'indoor space temperature'),
        ('TEMPERATURE', 'xm', 'internal mass temperature'),
    ]

    params = [
        ('THERMAL_RESISTANCE', 'Rw', 'between the indoor and the outdoor'),
        ('COEFFICIENT', 'rho', 'Wall ratio'),
        ('THERMAL_RESISTANCE', 'Rm', 'between the indoor and the internal mass'),
        ('THERMAL_CAPACITY', 'Cw', 'of the wall'),
        ('THERMAL_CAPACITY', 'Ci', 'of the indoor air, indoor walls, furnitures, etc.'),
        ('THERMAL_CAPACITY', 'Cm', 'of the internal mass'),
        ('SOLAR_APERTURE', 'Aw', 'of the outdoor surface'),
        ('COEFFICIENT', 'alpha', 'Effective wall/window ratio'),
        ('COEFFICIENT', 'cv', 'ventilation scaling'),
        ('STATE_DEVIATION', 'sigw_w', ''),
        ('STATE_DEVIATION', 'sigw_i', ''),
        ('STATE_DEVIATION', 'sigw_m', ''),
        ('MEASURE_DEVIATION', 'sigv', ''),
        ('INITIAL_MEAN', 'x0_w', ''),
        ('INITIAL_MEAN', 'x0_i', ''),
        ('INITIAL_MEAN', 'x0_m', ''),
        ('INITIAL_DEVIATION', 'sigx0_w', ''),
        ('INITIAL_DEVIATION', 'sigx0_i', ''),
        ('INITIAL_DEVIATION', 'sigx0_m', ''),
    ]

    inputs = [
        ('TEMPERATURE', 'To', 'outdoor air temperature'),
        ('POWER', 'Qgh', 'global horizontal solar radiation'),
        ('POWER', 'Qh', 'HVAC system heat'),
        ('POWER', 'Qv', 'heat from the ventilation system'),
    ]

    outputs = [('TEMPERATURE', 'xi', 'indoor air temperature')]

    def __post_init__(self):
        super().__post_init__()

    def set_constant_continuous_ssm(self):
        self.C[0, 1] = 1.0

    def set_constant_continuous_dssm(self):
        self.dQ['sigw_w'][0, 0] = 1.0
        self.dQ['sigw_i'][1, 1] = 1.0
        self.dQ['sigw_m'][2, 2] = 1.0
        self.dR['sigv'][0, 0] = 1.0
        self.dx0['x0_w'][0, 0] = 1.0
        self.dx0['x0_i'][1, 0] = 1.0
        self.dx0['x0_m'][2, 0] = 1.0
        self.dP0['sigx0_w'][0, 0] = 1.0
        self.dP0['sigx0_i'][1, 1] = 1.0
        self.dP0['sigx0_m'][2, 2] = 1.0

    def update_continuous_ssm(self):
        (
            Rw,
            rho,
            Rm,
            Cw,
            Ci,
            Cm,
            Aw,
            alpha,
            cv,
            sigw_w,
            sigw_i,
            sigw_m,
            sigv,
            x0_w,
            x0_i,
            x0_m,
            sigx0_w,
            sigx0_i,
            sigx0_m,
            *_,
        ) = self.parameters.theta

        self.A[:] = [
            [1.0 / (Cw * Rw * rho * (rho - 1.0)), -1.0 / (Cw * Rw * (rho - 1.0)), 0.0],
            [
                -1.0 / (Ci * Rw * (rho - 1.0)),
                (Rm - Rw * (rho - 1.0)) / (Ci * Rm * Rw * (rho - 1.0)),
                1.0 / (Ci * Rm),
            ],
            [0.0, 1.0 / (Cm * Rm), -1.0 / (Cm * Rm)],
        ]
        self.B[:2, :] = [
            [1.0 / (Cw * Rw * rho), Aw * alpha / Cw, 0.0, 0.0],
            [0.0, -Aw * (alpha - 1.0) / Ci, 1.0 / Ci, cv / Ci],
        ]
        self.Q[self._diag] = [sigw_w, sigw_i, sigw_m]
        self.R[0, 0] = sigv
        self.x0[:, 0] = [x0_w, x0_i, x0_m]
        self.P0[self._diag] = [sigx0_w, sigx0_i, sigx0_m]

    def update_continuous_dssm(self):
        Rw, rho, Rm, Cw, Ci, Cm, Aw, alpha, cv, *_ = self.parameters.theta

        self.dA['Rw'][:2, :2] = [
            [-1.0 / (Cw * Rw ** 2 * rho * (rho - 1.0)), 1.0 / (Cw * Rw ** 2 * (rho - 1.0))],
            [1.0 / (Ci * Rw ** 2 * (rho - 1.0)), -1.0 / (Ci * Rw ** 2 * (rho - 1.0))],
        ]
        self.dA['rho'][:2, :2] = [
            [
                (1.0 - 2.0 * rho) / (Cw * Rw * rho ** 2 * (rho - 1.0) ** 2),
                1.0 / (Cw * Rw * (rho - 1.0) ** 2),
            ],
            [1.0 / (Ci * Rw * (rho - 1.0) ** 2), -1.0 / (Ci * Rw * (rho - 1.0) ** 2)],
        ]

        self.dA['Rm'][1:, 1:] = [
            [1.0 / (Ci * Rm ** 2), -1.0 / (Ci * Rm ** 2)],
            [-1.0 / (Cm * Rm ** 2), 1.0 / (Cm * Rm ** 2)],
        ]
        self.dA['Cw'][0, :2] = [
            -1.0 / (Cw ** 2 * Rw * rho * (rho - 1.0)),
            1.0 / (Cw ** 2 * Rw * (rho - 1.0)),
        ]
        self.dA['Ci'][1, :] = [
            1.0 / (Ci ** 2 * Rw * (rho - 1.0)),
            (-Rm + Rw * (rho - 1.0)) / (Ci ** 2 * Rm * Rw * (rho - 1.0)),
            -1.0 / (Ci ** 2 * Rm),
        ]
        self.dA['Cm'][2, 1:] = [-1.0 / (Rm * Cm ** 2), 1.0 / (Rm * Cm ** 2)]

        self.dB['Rw'][0, 0] = -1.0 / (Cw * Rw ** 2 * rho)
        self.dB['rho'][0, 0] = -1.0 / (Cw * Rw * rho ** 2)
        self.dB['Cw'][0, :2] = [-1.0 / (Cw ** 2 * Rw * rho), -Aw * alpha / Cw ** 2]
        self.dB['Ci'][1, 1:] = [Aw * (alpha - 1.0) / Ci ** 2, -1.0 / Ci ** 2, -cv / Ci ** 2]
        self.dB['Aw'][:2, 1] = [alpha / Cw, (1.0 - alpha) / Ci]
        self.dB['alpha'][:2, 1] = [Aw / Cw, -Aw / Ci]
        self.dB['cv'][1, 3] = 1.0 / Ci
