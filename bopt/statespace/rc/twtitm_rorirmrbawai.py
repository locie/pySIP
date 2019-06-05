from dataclasses import dataclass, field

from ..base import RCModel


@dataclass
class TwTiTm_RoRiRmRbAwAi(RCModel):
    """Second order RC model"""

    states = [
        ('TEMPERATURE', 'xw', 'wall temperature'),
        ('TEMPERATURE', 'xi', 'indoor space temperature'),
        ('TEMPERATURE', 'xm', 'internal mass temperature')
    ]

    params = [
        ('THERMAL_RESISTANCE', 'Ro', 'between the outdoor and the wall node'),
        ('THERMAL_RESISTANCE', 'Ri', 'between the wall node and the indoor'),
        ('THERMAL_RESISTANCE', 'Rm', 'between the indoor and the internal mass'),
        ('THERMAL_RESISTANCE', 'Rb', 'between the indoor and the internal mass'),
        ('THERMAL_CAPACITY', 'Cw', 'of the wall'),
        ('THERMAL_CAPACITY', 'Ci', 'of the indoor air, indoor walls, furnitures, etc.'),
        ('THERMAL_CAPACITY', 'Cm', 'of the internal mass'),
        ('SOLAR_APERTURE', 'Aw', 'of the wall'),
        ('SOLAR_APERTURE', 'Ai', 'of the windows'),
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
        ('TEMPERATURE', 'Tb', 'boundary air temperature'),
        ('POWER', 'Qgh', 'global horizontal solar radiation'),
        ('POWER', 'Qh', 'HVAC system heat')
    ]

    outputs = [
        ('TEMPERATURE', 'xi', 'indoor air temperature')
    ]

    def set_constant(self):
        self.C[0, 1] = 1.0

    def set_jacobian(self):
        self.dQ['sigw_w'][0, 0] = 1.0
        self.dQ['sigw_i'][1, 1] = 1.0
        self.dQ['sigw_m'][2, 2] = 1.0
        self.dR['sigv'][0, 0] = 1.0
        self.dx0['x0_w'][0, 0] = self.sX
        self.dx0['x0_i'][1, 0] = self.sX
        self.dx0['x0_m'][2, 0] = self.sX
        self.dP0['sigx0_w'][0, 0] = 1.0
        self.dP0['sigx0_i'][1, 1] = 1.0
        self.dP0['sigx0_m'][2, 2] = 1.0

    def update_state_space_model(self):
        (Ro, Ri, Rm, Rb, Cw, Ci, Cm, Aw, Ai, sigw_w, sigw_i, sigw_m, sigv, x0_w, x0_i, x0_m, sigx0_w, sigx0_i, sigx0_m, *_) = self.parameters.theta

        self.A[:] = (
            (-(Ro + Ri) / (self.sC * Cw * Ri * Ro), 1.0 / (self.sC * Cw * Ri), 0.0),
            (1.0 / (self.sC * Ci * Ri), -(Rb * Ri + Rb * Rm + Ri * Rm) / (self.sC * Ci * Rb * Ri * Rm), 1.0 / (self.sC * Ci * Rm)),
            (0.0, 1.0 / (self.sC * Cm * Rm), -1.0 / (self.sC * Cm * Rm))
        )

        self.B[:2, :] = (
            (1.0 / (self.sC * Cw * Ro), 0.0, Aw / (self.sC * Cw), 0.0),
            (0.0, 1.0 / (self.sC * Ci * Rb), Ai / (self.sC * Ci), 1.0 / (self.sC * Ci))
        )

        self.Q[self._diag] = (sigw_w, sigw_i, sigw_m)

        self.R[0, 0] = sigv

        self.x0[:, 0] = (self.sX * x0_w, self.sX * x0_i, self.sX * x0_m)

        self.P0[self._diag] = (sigx0_w, sigx0_i, sigx0_m)

    def update_jacobian(self):
        (Ro, Ri, Rm, Rb, Cw, Ci, Cm, Aw, Ai, *_) = self.parameters.theta

        self.dA['Ro'][0, 0] = 1.0 / (self.sC * Cw * Ro**2)

        self.dA['Ri'][:2, :2] = [
            [1.0 / (self.sC * Cw * Ri**2), -1.0 / (self.sC * Cw * Ri**2)],
            [-1.0 / (self.sC * Ci * Ri**2), 1.0 / (self.sC * Ci * Ri**2)]
        ]

        self.dA['Rm'][1:, 1:] = [
            [1.0 / (self.sC * Ci * Rm**2), -1.0 / (self.sC * Ci * Rm**2)],
            [-1.0 / (self.sC * Cm * Rm**2), 1.0 / (self.sC * Cm * Rm**2)]
        ]

        self.dA['Rb'][1, 1] = 1.0 / (self.sC * Ci * Rb**2)

        self.dA['Cw'][0, :2] = [
            (Ro + Ri) / (self.sC * Ri * Ro * Cw**2), -1.0 / (self.sC * Ri * Cw**2)
        ]

        self.dA['Ci'][1, :] = [
            -1.0 / (self.sC * Ri * Ci**2),
            (Rb * Ri + Rb * Rm + Ri * Rm) / (self.sC * Rb * Ri * Rm * Ci**2),
            -1.0 / (self.sC * Rm * Ci**2)
        ]

        self.dA['Cm'][2, 1:] = [-1.0 / (self.sC * Rm * Cm**2), 1.0 / (self.sC * Rm * Cm**2)]

        self.dB['Ro'][0, 0] = -1.0 / (self.sC * Cw * Ro**2)

        self.dB['Rb'][1, 1] = -1.0 / (self.sC * Ci * Rb**2)

        self.dB['Cw'][0, :3] = [-1.0 / (self.sC * Ro * Cw**2), 0.0, -Aw / (self.sC * Cw**2)]

        self.dB['Ci'][1, 1:] = [-1.0 / (self.sC * Rb * Ci**2), -Ai / (self.sC * Ci**2), -1.0 / (self.sC * Ci**2)]

        self.dB['Aw'][0, 2] = 1.0 / (self.sC * Cw)

        self.dB['Ai'][1, 2] = 1.0 / (self.sC * Ci)
