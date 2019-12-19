from dataclasses import dataclass

from ..base import RCModel


@dataclass
class TiTh_RwRhAi(RCModel):
    """Second order RC model"""

    states = [
        ('TEMPERATURE', 'xi', 'indoor temperature'),
        ('TEMPERATURE', 'xh', 'heater temperature'),
    ]

    params = [
        ('THERMAL_RESISTANCE', 'Rw', 'between the outdoor and the indoor'),
        ('THERMAL_RESISTANCE', 'Rh', 'between the indoor and the heater'),
        ('THERMAL_CAPACITY', 'Ci', 'of the indoor'),
        ('THERMAL_CAPACITY', 'Ch', 'of the heater'),
        ('SOLAR_APERTURE', 'Ai', 'effective solar aperture'),
        ('STATE_DEVIATION', 'sigw_i', 'of the indoor dynamic'),
        ('STATE_DEVIATION', 'sigw_h', 'of the heater dynamic'),
        ('MEASURE_DEVIATION', 'sigv', 'of the indoor temperature measurements'),
        ('INITIAL_MEAN', 'x0_i', 'of the infoor temperature'),
        ('INITIAL_MEAN', 'x0_h', 'of the heater temperature'),
        ('INITIAL_DEVIATION', 'sigx0_i', 'of the infoor temperature'),
        ('INITIAL_DEVIATION', 'sigx0_h', 'of the heater temperature'),
    ]

    inputs = [
        ('TEMPERATURE', 'To', 'outdoor temperature'),
        ('POWER', 'Qgh', 'solar irradiance'),
        ('POWER', 'Qh', 'HVAC system heat'),
    ]

    outputs = [('TEMPERATURE', 'xi', 'indoor temperature')]

    def __post_init__(self):
        super().__post_init__()

    def set_constant_continuous_ssm(self):
        self.C[0, 0] = 1.0

    def set_constant_continuous_dssm(self):
        self.dQ['sigw_i'][0, 0] = 1.0
        self.dQ['sigw_h'][1, 1] = 1.0
        self.dR['sigv'][0, 0] = 1.0
        self.dx0['x0_i'][0, 0] = 1.0
        self.dx0['x0_h'][1, 0] = 1.0
        self.dP0['sigx0_i'][0, 0] = 1.0
        self.dP0['sigx0_h'][1, 1] = 1.0

    def update_continuous_ssm(self):
        (
            Rw,
            Rh,
            Ci,
            Ch,
            Ai,
            sigw_i,
            sigw_h,
            sigv,
            x0_i,
            x0_h,
            sigx0_i,
            sigx0_h,
            *_,
        ) = self.parameters.theta

        self.A[:] = [
            [-(Rh + Rw) / (Ci * Rh * Rw), 1.0 / (Ci * Rh)],
            [1.0 / (Ch * Rh), -1.0 / (Ch * Rh)],
        ]
        self.B[:] = [[1.0 / (Ci * Rw), Ai / Ci, 0.0], [0.0, 0.0, 1.0 / Ch]]
        self.Q[self._diag] = [sigw_i, sigw_h]
        self.R[0, 0] = sigv
        self.x0[:, 0] = [x0_i, x0_h]
        self.P0[self._diag] = [sigx0_i, sigx0_h]

    def update_continuous_dssm(self):
        Rw, Rh, Ci, Ch, Ai, *_ = self.parameters.theta

        self.dA['Rw'][0, 0] = 1.0 / (Ci * Rw ** 2)
        self.dA['Rh'][:] = [
            [1.0 / (Ci * Rh ** 2), -1.0 / (Ci * Rh ** 2)],
            [-1.0 / (Ch * Rh ** 2), 1.0 / (Ch * Rh ** 2)],
        ]
        self.dA['Ci'][0, :] = [(Rh + Rw) / (Ci ** 2 * Rh * Rw), -1.0 / (Ci ** 2 * Rh)]
        self.dA['Ch'][1, :] = [-1.0 / (Ch ** 2 * Rh), 1.0 / (Ch ** 2 * Rh)]

        self.dB['Rw'][0, 0] = -1.0 / (Ci * Rw ** 2)
        self.dB['Ci'][0, :2] = [-1.0 / (Ci ** 2 * Rw), -Ai / Ci ** 2]
        self.dB['Ch'][1, 2] = -1.0 / Ch ** 2
        self.dB['Ai'][0, 1] = 1.0 / Ci
