from dataclasses import dataclass

from ..base import RCModel


@dataclass
class TwTiTh_RoRiRhAwAi(RCModel):
    """3rd order RC model"""

    states = [
        ('TEMPERATURE', 'xw', 'wall temperature'),
        ('TEMPERATURE', 'xi', 'indoor temperature'),
        ('TEMPERATURE', 'xh', 'heaters temperature'),
    ]

    params = [
        ('THERMAL_RESISTANCE', 'Ro', 'between the outdoor and the wall node'),
        ('THERMAL_RESISTANCE', 'Ri', 'between the wall node and the indoor'),
        ('THERMAL_RESISTANCE', 'Rh', 'between the heaters and the indoor'),
        ('THERMAL_CAPACITY', 'Cw', 'of the wall'),
        ('THERMAL_CAPACITY', 'Ci', 'of the indoor air, indoor walls, furnitures, etc.'),
        ('THERMAL_CAPACITY', 'Ch', 'of the heaters'),
        ('SOLAR_APERTURE', 'Aw', 'of the wall'),
        ('SOLAR_APERTURE', 'Ai', 'of the windows'),
        ('STATE_DEVIATION', 'sigw_w', 'of the wall temperature'),
        ('STATE_DEVIATION', 'sigw_i', 'of the indoor temperature'),
        ('STATE_DEVIATION', 'sigw_h', 'of the heaters temperature'),
        ('MEASURE_DEVIATION', 'sigv', 'of  the indoor temperature measure'),
        ('INITIAL_MEAN', 'x0_w', 'of the wall temperature'),
        ('INITIAL_MEAN', 'x0_i', 'of the indoor temperature'),
        ('INITIAL_MEAN', 'x0_h', 'of the heaters temperature'),
        ('INITIAL_DEVIATION', 'sigx0_w', 'of the wall temperature'),
        ('INITIAL_DEVIATION', 'sigx0_i', 'of the indoor temperature'),
        ('INITIAL_DEVIATION', 'sigx0_h', 'of the heaters temperature'),
    ]

    inputs = [
        ('TEMPERATURE', 'To', 'outdoor air temperature'),
        ('POWER', 'Qgh', 'global horizontal solar radiation'),
        ('POWER', 'Qh', 'HVAC system power'),
    ]

    outputs = [('TEMPERATURE', 'xi', 'indoor air temperature')]

    def __post_init__(self):
        super().__post_init__()

    def set_constant_continuous_ssm(self):
        self.C[0, 1] = 1.0

    def set_constant_continuous_dssm(self):
        self.dQ['sigw_w'][0, 0] = 1.0
        self.dQ['sigw_i'][1, 1] = 1.0
        self.dQ['sigw_h'][2, 2] = 1.0
        self.dR['sigv'][0, 0] = 1.0
        self.dx0['x0_w'][0, 0] = 1.0
        self.dx0['x0_i'][1, 0] = 1.0
        self.dx0['x0_h'][2, 0] = 1.0
        self.dP0['sigx0_w'][0, 0] = 1.0
        self.dP0['sigx0_i'][1, 1] = 1.0
        self.dP0['sigx0_h'][2, 2] = 1.0

    def update_continuous_ssm(self):
        (
            Ro,
            Ri,
            Rh,
            Cw,
            Ci,
            Ch,
            Aw,
            Ai,
            sigw_w,
            sigw_i,
            sigw_h,
            sigv,
            x0_w,
            x0_i,
            x0_h,
            sigx0_w,
            sigx0_i,
            sigx0_h,
            *_,
        ) = self.parameters.theta

        self.A[:] = [
            [-(Ro + Ri) / (Cw * Ri * Ro), 1.0 / (Cw * Ri), 0.0],
            [1.0 / (Ci * Ri), -(Ri + Rh) / (Ci * Ri * Rh), 1.0 / (Ci * Rh)],
            [0.0, 1.0 / (Ch * Rh), -1.0 / (Ch * Rh)],
        ]
        self.B[:] = [[1.0 / (Cw * Ro), Aw / Cw, 0.0], [0.0, Ai / Ci, 0.0], [0.0, 0.0, 1.0 / Ch]]
        self.Q[self._diag] = [sigw_w, sigw_i, sigw_h]
        self.R[0, 0] = sigv
        self.x0[:, 0] = [x0_w, x0_i, x0_h]
        self.P0[self._diag] = [sigx0_w, sigx0_i, sigx0_h]

    def update_continuous_dssm(self):
        Ro, Ri, Rh, Cw, Ci, Ch, Aw, Ai, *_ = self.parameters.theta

        self.dA['Ro'][0, 0] = 1.0 / (Cw * Ro ** 2)
        self.dA['Ri'][:2, :2] = [
            [1.0 / (Cw * Ri ** 2), -1.0 / (Cw * Ri ** 2)],
            [-1.0 / (Ci * Ri ** 2), 1.0 / (Ci * Ri ** 2)],
        ]
        self.dA['Rh'][1:, 1:] = [
            [1.0 / (Ci * Rh ** 2), -1.0 / (Ci * Rh ** 2)],
            [-1.0 / (Ch * Rh ** 2), 1.0 / (Ch * Rh ** 2)],
        ]
        self.dA['Cw'][0, :2] = [(Ro + Ri) / (Ri * Ro * Cw ** 2), -1.0 / (Ri * Cw ** 2)]
        self.dA['Ci'][1, :] = [
            -1.0 / (Ri * Ci ** 2),
            (Ri + Rh) / (Ri * Rh * Ci ** 2),
            -1.0 / (Rh * Ci ** 2),
        ]
        self.dA['Ch'][2, 1:] = [-1.0 / (Rh * Ch ** 2), 1.0 / (Rh * Ch ** 2)]

        self.dB['Ro'][0, 0] = -1.0 / (Cw * Ro ** 2)
        self.dB['Cw'][0, :2] = [-1.0 / (Ro * Cw ** 2), -Aw / Cw ** 2]
        self.dB['Ci'][1, 1] = -Ai / Ci ** 2
        self.dB['Ch'][2, 2] = -1.0 / Ch ** 2
        self.dB['Aw'][0, 1] = 1.0 / Cw
        self.dB['Ai'][1, 1] = 1.0 / Ci
