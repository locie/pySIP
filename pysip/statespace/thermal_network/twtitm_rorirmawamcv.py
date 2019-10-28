from dataclasses import dataclass

from ..base import RCModel


@dataclass
class TwTiTm_RoRiRmAwAmcv(RCModel):
    """Second order RC model"""

    states = [
        ('TEMPERATURE', 'xw', 'wall temperature'),
        ('TEMPERATURE', 'xi', 'indoor space temperature'),
        ('TEMPERATURE', 'xm', 'internal mass temperature'),
    ]

    params = [
        ('THERMAL_RESISTANCE', 'Ro', 'between the outdoor and the wall node'),
        ('THERMAL_RESISTANCE', 'Ri', 'between the wall node and the indoor'),
        ('THERMAL_RESISTANCE', 'Rm', 'between the indoor and the internal mass'),
        ('THERMAL_CAPACITY', 'Cw', 'of the wall'),
        ('THERMAL_CAPACITY', 'Ci', 'of the indoor air, indoor walls, furnitures, etc.'),
        ('THERMAL_CAPACITY', 'Cm', 'of the internal mass'),
        ('SOLAR_APERTURE', 'Aw', 'of the wall'),
        ('SOLAR_APERTURE', 'Am', 'of the internal mass'),
        ('COEFFICIENT', 'cv', 'ventilation heating contribution scaling'),
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
            Ro,
            Ri,
            Rm,
            Cw,
            Ci,
            Cm,
            Aw,
            Am,
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
            [-(Ro + Ri) / (Cw * Ri * Ro), 1.0 / (Cw * Ri), 0.0],
            [1.0 / (Ci * Ri), -(Ri + Rm) / (Ci * Ri * Rm), 1.0 / (Ci * Rm)],
            [0.0, 1.0 / (Cm * Rm), -1.0 / (Cm * Rm)],
        ]
        self.B[:] = [
            [1.0 / (Cw * Ro), Aw / Cw, 0.0, 0.0],
            [0.0, 0.0, 1.0 / Ci, cv / Ci],
            [0.0, Am / Cm, 0.0, 0.0],
        ]
        self.Q[self._diag] = [sigw_w, sigw_i, sigw_m]
        self.R[0, 0] = sigv
        self.x0[:, 0] = [x0_w, x0_i, x0_m]
        self.P0[self._diag] = [sigx0_w, sigx0_i, sigx0_m]

    def update_continuous_dssm(self):
        (Ro, Ri, Rm, Cw, Ci, Cm, Aw, Am, cv, *_) = self.parameters.theta

        self.dA['Ro'][0, 0] = 1.0 / (Cw * Ro ** 2)
        self.dA['Ri'][:2, :2] = [
            [1.0 / (Cw * Ri ** 2), -1.0 / (Cw * Ri ** 2)],
            [-1.0 / (Ci * Ri ** 2), 1.0 / (Ci * Ri ** 2)],
        ]
        self.dA['Rm'][1:, 1:] = [
            [1.0 / (Ci * Rm ** 2), -1.0 / (Ci * Rm ** 2)],
            [-1.0 / (Cm * Rm ** 2), 1.0 / (Cm * Rm ** 2)],
        ]
        self.dA['Cw'][0, :2] = [(Ro + Ri) / (Ri * Ro * Cw ** 2), -1.0 / (Ri * Cw ** 2)]
        self.dA['Ci'][1, :] = [
            -1.0 / (Ri * Ci ** 2),
            (Ri + Rm) / (Ri * Rm * Ci ** 2),
            -1.0 / (Rm * Ci ** 2),
        ]
        self.dA['Cm'][2, 1:] = [-1.0 / (Rm * Cm ** 2), 1.0 / (Rm * Cm ** 2)]

        self.dB['Ro'][0, 0] = -1.0 / (Cw * Ro ** 2)
        self.dB['Cw'][0, :2] = [-1.0 / (Ro * Cw ** 2), -Aw / Cw ** 2]
        self.dB['Ci'][1, 2:] = [-1.0 / Ci ** 2, -cv / Ci ** 2]
        self.dB['Cm'][2, 1] = -Am / Cm ** 2
        self.dB['Aw'][0, 1] = 1.0 / Cw
        self.dB['Am'][2, 1] = 1.0 / Cm
        self.dB['cv'][1, 3] = 1.0 / Ci
