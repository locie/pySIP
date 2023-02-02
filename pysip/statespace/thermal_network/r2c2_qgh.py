from dataclasses import dataclass

from ..base import RCModel


@dataclass
class R2C2Qgh(RCModel):

    states = [
        ('TEMPERATURE', 'xw', 'wall temperature'),
        ('TEMPERATURE', 'xi', 'indoor space temperature'),
    ]

    params = [
        ('THERMAL_RESISTANCE', 'Ro', 'between the outdoor and the wall node'),
        ('THERMAL_RESISTANCE', 'Ri', 'between the wall node and the indoor'),
        ('THERMAL_CAPACITY', 'Cw', 'Wall'),
        ('THERMAL_CAPACITY', 'Ci', 'indoor air, indoor walls, furnitures, etc. '),
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
        self.dQ["sigw_w"][0, 0] = 1.0
        self.dQ["sigw_i"][1, 1] = 1.0
        self.dR["sigv"][0, 0] = 1.0
        self.dx0["x0_w"][0, 0] = 1.0
        self.dx0["x0_i"][1, 0] = 1.0
        self.dP0["sigx0_w"][0, 0] = 1.0
        self.dP0["sigx0_i"][1, 1] = 1.0

    def update_continuous_ssm(self):
        (
            Ro,
            Ri,
            Cw,
            Ci,
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
        self.B[:] = [[1.0 / (Cw * Ro), 1.0 / Cw, 0.0], [0.0, 1.0 / Ci, 1.0 / Ci]]
        self.Q[self._diag] = [sigw_w, sigw_i]
        self.R[0, 0] = sigv
        self.x0[:, 0] = [x0_w, x0_i]
        self.P0[self._diag] = [sigx0_w, sigx0_i]

    def update_continuous_dssm(self):
        Ro, Ri, Cw, Ci, *_ = self.parameters.theta

        self.dA["Ro"][0, 0] = 1.0 / (Cw * Ro ** 2)
        self.dA["Ri"][:] = [
            [1.0 / (Cw * Ri ** 2), -1.0 / (Cw * Ri ** 2)],
            [-1.0 / (Ci * Ri ** 2), 1.0 / (Ci * Ri ** 2)],
        ]
        self.dA["Cw"][0, :] = [(Ro + Ri) / (Cw ** 2 * Ri * Ro), -1.0 / (Cw ** 2 * Ri)]
        self.dA["Ci"][1, :] = [-1.0 / (Ci ** 2 * Ri), 1.0 / (Ci ** 2 * Ri)]

        self.dB["Ro"][0, 0] = -1.0 / (Cw * Ro ** 2)
        self.dB["Cw"][0, :2] = [-1.0 / (Cw ** 2 * Ro), -1.0 / (Cw ** 2)]
        self.dB["Ci"][1, 1:] = [-1.0 / (Ci ** 2), -1.0 / (Ci ** 2)]
