from dataclasses import dataclass

from ..base import RCModel


@dataclass
class TwTi_RoRiRbAwAicv(RCModel):
    """Second order RC model"""

    states = [
        ('TEMPERATURE', 'xw', 'wall temperature'),
        ('TEMPERATURE', 'xi', 'indoor space temperature'),
    ]

    params = [
        ('THERMAL_RESISTANCE', 'Ro', 'between the outdoor and the wall node'),
        ('THERMAL_RESISTANCE', 'Ri', 'between the wall node and the indoor'),
        ('THERMAL_RESISTANCE', 'Rb', 'between the indoor and the boundary space'),
        ('THERMAL_CAPACITY', 'Cw', 'Wall'),
        ('THERMAL_CAPACITY', 'Ci', 'indoor air, indoor walls, furnitures, etc. '),
        ('SOLAR_APERTURE', 'Aw', 'of the wall (m2)'),
        ('SOLAR_APERTURE', 'Ai', 'of the windows (m2)'),
        ('COEFFICIENT', 'cv', 'scaling of the heating contribution of the ventilation'),
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
        ('TEMPERATURE', 'Tb', 'boundary air temperature'),
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
        self.dR['sigv'][0, 0] = 1.0
        self.dx0['x0_w'][0, 0] = 1.0
        self.dx0['x0_i'][1, 0] = 1.0
        self.dP0['sigx0_w'][0, 0] = 1.0
        self.dP0['sigx0_i'][1, 1] = 1.0

    def update_continuous_ssm(self):
        (
            Ro,
            Ri,
            Rb,
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
            [1.0 / (Ci * Ri), -(Rb + Ri) / (Ci * Ri * Rb)],
        ]

        self.B[1, 1:] = [1.0 / (Ci * Rb), Ai / (Ci), 1.0 / (Ci), cv / (Ci)]

        self.B[:] = [
            [1.0 / (Cw * Ro), 0.0, Aw / Cw, 0.0, 0.0],
            [0.0, 1.0 / (Ci * Rb), Ai / Ci, 1.0 / Ci, cv / Ci],
        ]
        self.Q[self._diag] = [sigw_w, sigw_i]
        self.R[0, 0] = sigv
        self.x0[:, 0] = [x0_w, x0_i]
        self.P0[self._diag] = [sigx0_w, sigx0_i]

    def update_continuous_dssm(self):
        Ro, Ri, Rb, Cw, Ci, Aw, Ai, cv, *_ = self.parameters.theta

        self.dA['Ro'][0, 0] = 1.0 / (Cw * Ro ** 2)
        self.dA['Ri'][:] = [
            [1.0 / (Cw * Ri ** 2), -1.0 / (Cw * Ri ** 2)],
            [-1.0 / (Ci * Ri ** 2), 1.0 / (Ci * Ri ** 2)],
        ]
        self.dA['Rb'][1, 1] = 1.0 / (Ci * Rb ** 2)
        self.dA['Cw'][0, :] = [(Ro + Ri) / (Cw ** 2 * Ri * Ro), -1.0 / (Cw ** 2 * Ri)]
        self.dA['Ci'][1, :] = [-1.0 / (Ci ** 2 * Ri), (Rb + Ri) / (Ci ** 2 * Ri * Rb)]

        self.dB['Ro'][0, 0] = -1.0 / (Cw * Ro ** 2)
        self.dB['Rb'][1, 1] = -1.0 / (Ci * Rb ** 2)
        self.dB['Cw'][0, :3] = [-1.0 / (Cw ** 2 * Ro), 0, -Aw / Cw ** 2]
        self.dB['Ci'][1, 1:] = [-1.0 / (Ci ** 2 * Rb), -Ai / Ci ** 2, -1.0 / Ci ** 2, -cv / Ci ** 2]
        self.dB['Aw'][0, 2] = 1.0 / Cw
        self.dB['Ai'][1, 2] = 1.0 / Ci
        self.dB['cv'][1, 4] = 1.0 / Ci
