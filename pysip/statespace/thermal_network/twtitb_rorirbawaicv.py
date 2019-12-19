from dataclasses import dataclass
from ..base import RCModel


@dataclass
class TwTiTb_RoRiRbAwAicv(RCModel):
    """Third order RC model"""

    states = [
        ('TEMPERATURE', 'xw', 'envelope temperature'),
        ('TEMPERATURE', 'xi', 'indoor temperature'),
        ('TEMPERATURE', 'xb', 'boundary wall temperature'),
    ]

    params = [
        ('THERMAL_RESISTANCE', 'Ro', 'between the outdoor and the envelope'),
        ('THERMAL_RESISTANCE', 'Ri', 'between the envelope and the indoor'),
        ('THERMAL_RESISTANCE', 'Rb', 'between the indoor and the boundary'),
        ('THERMAL_CAPACITY', 'Cw', 'of the envelope'),
        ('THERMAL_CAPACITY', 'Ci', 'of the indoor'),
        ('THERMAL_CAPACITY', 'Cb', 'of the wall between the indoor and the boundary'),
        ('SOLAR_APERTURE', 'Aw', 'of the envelope'),
        ('SOLAR_APERTURE', 'Ai', 'of the windows'),
        ('COEFFICIENT', 'cv', 'scaling of the heat from the ventilation'),
        ('STATE_DEVIATION', 'sigw_w', 'of the envelope dynamic'),
        ('STATE_DEVIATION', 'sigw_i', 'of the indoor dynamic'),
        ('STATE_DEVIATION', 'sigw_b', 'of the boundary wall dynamic'),
        ('MEASURE_DEVIATION', 'sigv', 'of the indoor temperature measurements'),
        ('INITIAL_MEAN', 'x0_w', 'of the envelope temperature'),
        ('INITIAL_MEAN', 'x0_i', 'of the infoor temperature'),
        ('INITIAL_MEAN', 'x0_b', 'of the boundary wall temperature'),
        ('INITIAL_DEVIATION', 'sigx0_w', 'of the envelope temperature'),
        ('INITIAL_DEVIATION', 'sigx0_i', 'of the infoor temperature'),
        ('INITIAL_DEVIATION', 'sigx0_b', 'of the boundary wall temperature'),
    ]

    inputs = [
        ('TEMPERATURE', 'To', 'outdoor temperature'),
        ('TEMPERATURE', 'Tb', 'boundary temperature'),
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
        self.dQ['sigw_b'][2, 2] = 1.0
        self.dR['sigv'][0, 0] = 1.0
        self.dx0['x0_w'][0, 0] = 1.0
        self.dx0['x0_i'][1, 0] = 1.0
        self.dx0['x0_b'][2, 0] = 1.0
        self.dP0['sigx0_w'][0, 0] = 1.0
        self.dP0['sigx0_i'][1, 1] = 1.0
        self.dP0['sigx0_b'][2, 2] = 1.0

    def update_continuous_ssm(self):
        (
            Ro,
            Ri,
            Rb,
            Cw,
            Ci,
            Cb,
            Aw,
            Ai,
            cv,
            sigw_w,
            sigw_i,
            sigw_b,
            sigv,
            x0_w,
            x0_i,
            x0_b,
            sigx0_w,
            sigx0_i,
            sigx0_b,
            *_,
        ) = self.parameters.theta

        self.A[:] = [
            [-(Ro + Ri) / (Cw * Ri * Ro), 1.0 / (Cw * Ri), 0.0],
            [1.0 / (Ci * Ri), -(Rb + 2.0 * Ri) / (Ci * Rb * Ri), 2.0 / (Ci * Rb)],
            [0.0, 2.0 / (Cb * Rb), -4.0 / (Cb * Rb)],
        ]
        self.B[:] = [
            [1.0 / (Cw * Ro), 0.0, Aw / Cw, 0.0, 0.0],
            [0.0, 0.0, Ai / Ci, 1.0 / Ci, cv / Ci],
            [0.0, 2.0 / (Cb * Rb), 0.0, 0.0, 0.0],
        ]
        self.Q[self._diag] = [sigw_w, sigw_i, sigw_b]
        self.R[0, 0] = sigv
        self.x0[:, 0] = [x0_w, x0_i, x0_b]
        self.P0[self._diag] = [sigx0_w, sigx0_i, sigx0_b]

    def update_continuous_dssm(self):
        Ro, Ri, Rb, Cw, Ci, Cb, Aw, Ai, cv, *_ = self.parameters.theta

        self.dA['Ro'][0, 0] = 1.0 / (Cw * Ro ** 2)
        self.dA['Ri'][:2, :2] = [
            [1.0 / (Cw * Ri ** 2), -1.0 / (Cw * Ri ** 2)],
            [-1.0 / (Ci * Ri ** 2), 1.0 / (Ci * Ri ** 2)],
        ]
        self.dA['Rb'][1:, 1:] = [
            [2.0 / (Ci * Rb ** 2), -2.0 / (Ci * Rb ** 2)],
            [-2.0 / (Cb * Rb ** 2), 4.0 / (Cb * Rb ** 2)],
        ]
        self.dA['Cw'][0, :2] = [(Ro + Ri) / (Ri * Ro * Cw ** 2), -1.0 / (Ri * Cw ** 2)]
        self.dA['Ci'][1, :] = [
            -1.0 / (Ri * Ci ** 2),
            (Rb + 2.0 * Ri) / (Rb * Ri * Ci ** 2),
            -2.0 / (Rb * Ci ** 2),
        ]
        self.dA['Cb'][2, 1:] = [-2.0 / (Rb * Cb ** 2), 4.0 / (Rb * Cb ** 2)]

        self.dB['Ro'][0, 0] = -1.0 / (Cw * Ro ** 2)
        self.dB['Rb'][2, 1] = -2.0 / (Cb * Rb ** 2)
        self.dB['Cw'][0, :3] = [-1.0 / (Ro * Cw ** 2), 0.0, -Aw / Cw ** 2]
        self.dB['Ci'][1, 2:] = [-Ai / Ci ** 2, -1.0 / Ci ** 2, -cv / Ci ** 2]
        self.dB['Cb'][2, 1] = -2.0 / (Rb * Cb ** 2)
        self.dB['Aw'][0, 2] = 1.0 / Cw
        self.dB['Ai'][1, 2] = 1.0 / Ci
        self.dB['cv'][1, 4] = 1.0 / Ci
