from dataclasses import dataclass

from ..base import RCModel


@dataclass
class TwTi_RoRiAwAicv(RCModel):
    """Second order RC model"""

    states = [
        ('TEMPERATURE', 'xw', 'envelope temperature'),
        ('TEMPERATURE', 'xi', 'indoor temperature'),
    ]

    params = [
        ('THERMAL_RESISTANCE', 'Ro', 'between the outdoor and the envelope'),
        ('THERMAL_RESISTANCE', 'Ri', 'between the envelope and the indoor'),
        ('THERMAL_CAPACITY', 'Cw', 'of the envelope'),
        ('THERMAL_CAPACITY', 'Ci', 'of the indoor'),
        ('SOLAR_APERTURE', 'Aw', 'of the envelope'),
        ('SOLAR_APERTURE', 'Ai', 'of the windows'),
        ('COEFFICIENT', 'cv', 'scaling of the heat from the ventilation'),
        ('STATE_DEVIATION', 'sigw_w', 'of the envelope dynamic'),
        ('STATE_DEVIATION', 'sigw_i', 'of the indoor dynamic'),
        ('MEASURE_DEVIATION', 'sigv', 'of the indoor temperature measurements'),
        ('INITIAL_MEAN', 'x0_w', 'of the envelope temperature'),
        ('INITIAL_MEAN', 'x0_i', 'of the infoor temperature'),
        ('INITIAL_DEVIATION', 'sigx0_w', 'of the envelope temperature'),
        ('INITIAL_DEVIATION', 'sigx0_i', 'of the infoor temperature'),
    ]

    inputs = [
        ('TEMPERATURE', 'To', 'outdoor temperature'),
        ('POWER', 'Qgh', 'solar irradiance'),
        ('POWER', 'Qh', 'HVAC system heat'),
        ('POWER', 'Qv', 'heat from the ventilation system'),
    ]

    outputs = [('TEMPERATURE', 'xi', 'indoor temperature')]

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
            Ro,
            Ri,
            Cw,
            Ci,
            Aw,
            Ai,
            cv,
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
            [-(Ro + Ri) / (Cw * Ri * Ro), 1.0 / (Cw * Ri)],
            [1.0 / (Ci * Ri), -1.0 / (Ci * Ri)],
        ]
        self.B[:] = [[1.0 / (Cw * Ro), Aw / Cw, 0.0, 0.0], [0.0, Ai / Ci, 1.0 / Ci, cv / Ci]]
        self.Q[self._diag] = [sigw_w, sigw_i]
        self.R[0, 0] = sigv
        self.x0[:, 0] = [x0_w, x0_i]
        self.P0[self._diag] = [sigx0_w, sigx0_i]

    def update_continuous_dssm(self):
        (Ro, Ri, Cw, Ci, Aw, Ai, cv, *_) = self.parameters.theta

        self.dA['Ro'][0, 0] = 1.0 / (Cw * Ro ** 2)
        self.dA['Ri'][:] = [
            [1.0 / (Cw * Ri ** 2), -1.0 / (Cw * Ri ** 2)],
            [-1.0 / (Ci * Ri ** 2), 1.0 / (Ci * Ri ** 2)],
        ]
        self.dA['Cw'][0, :] = [(Ro + Ri) / (Cw ** 2 * Ri * Ro), -1.0 / (Cw ** 2 * Ri)]
        self.dA['Ci'][1, :] = [-1.0 / (Ci ** 2 * Ri), 1.0 / (Ci ** 2 * Ri)]

        self.dB['Ro'][0, 0] = -1.0 / (Cw * Ro ** 2)
        self.dB['Cw'][0, :2] = [-1.0 / (Cw ** 2 * Ro), -Aw / Cw ** 2]
        self.dB['Ci'][1, 1:] = [-Ai / Ci ** 2, -1.0 / Ci ** 2, -cv / Ci ** 2]
        self.dB['Aw'][0, 1] = 1.0 / Cw
        self.dB['Ai'][1, 1] = 1.0 / Ci
        self.dB['cv'][1, 3] = 1.0 / Ci
