from dataclasses import dataclass, field

from ..base import RCModel


@dataclass
class TwTi_RwAwAicv(RCModel):
    """Second order RC model"""

    states = [
        ('TEMPERATURE', 'xw', 'wall temperature'),
        ('TEMPERATURE', 'xi', 'indoor space temperature')
    ]

    params = [
        ('THERMAL_RESISTANCE', 'Rw', 'between the outdoor and the indoor'),
        ('THERMAL_CAPACITY', 'Cw', 'of the wall'),
        ('THERMAL_CAPACITY', 'Ci', 'of the indoor air, indoor walls, furnitures, etc.'),
        ('SOLAR_APERTURE', 'Aw', 'of the wall'),
        ('SOLAR_APERTURE', 'Ai', 'of the windows'),
        ('COEFFICIENT', 'cv', 'ventilation heating contribution scaling'),
        ('STATE_DEVIATION', 'sigw_w', ''),
        ('STATE_DEVIATION', 'sigw_i', ''),
        ('MEASURE_DEVIATION', 'sigv', ''),
        ('INITIAL_MEAN', 'x0_w', ''),
        ('INITIAL_MEAN', 'x0_i', ''),
        ('INITIAL_DEVIATION', 'sigx0_w', ''),
        ('INITIAL_DEVIATION', 'sigx0_i', '')
    ]

    inputs = [
        ('TEMPERATURE', 'To', 'outdoor air temperature'),
        ('POWER', 'Qh', 'HVAC system heat'),
        ('POWER', 'Qgh', 'global horizontal solar radiation'),
        ('POWER', 'Qv', 'heat from the ventilation system')
    ]

    outputs = [
        ('TEMPERATURE', 'xi', 'indoor air temperature')
    ]

    def set_constant(self):
        self.C[0, 1] = 1.0

    def set_jacobian(self):
        self.dQ['sigw_w'][0, 0] = 1.0
        self.dQ['sigw_i'][1, 1] = 1.0
        self.dR['sigv'][0, 0] = 1.0
        self.dx0['x0_w'][0, 0] = self.sX
        self.dx0['x0_i'][1, 0] = self.sX
        self.dP0['sigx0_w'][0, 0] = 1.0
        self.dP0['sigx0_i'][1, 1] = 1.0

    def update_state_space_model(self):
        (Rw, Cw, Ci, Aw, Ai, cv, sigw_w, sigw_i, sigv, x0_w, x0_i, sigx0_w, sigx0_i, *_) = self.parameters.theta

        self.A[:] = [
            [-4.0 / (self.sC * Cw * Rw), 2.0 / (self.sC * Cw * Rw)],
            [2.0 / (self.sC * Ci * Rw), -2.0 / (self.sC * Ci * Rw)]
        ]

        self.B[:] = [
            [2.0 / (self.sC * Cw * Rw), Aw / (self.sC * Cw), 0.0, 0.0],
            [0.0, Ai / (self.sC * Ci), 1.0 / (self.sC * Ci), cv / (self.sC * Ci)]
        ]

        self.Q[self._diag] = [sigw_w, sigw_i]

        self.R[0, 0] = sigv

        self.x0[:, 0] = [self.sX * x0_w, self.sX * x0_i]

        self.P0[self._diag] = [sigx0_w, sigx0_i]

    def update_jacobian(self):
        (Rw, Cw, Ci, Aw, Ai, cv, *_) = self.parameters.theta

        self.dA['Rw'][:] = [
            [4.0 / (self.sC * Cw * Rw**2), -2.0 / (self.sC * Cw * Rw**2)],
            [-2.0 / (self.sC * Ci * Rw**2), 2.0 / (self.sC * Ci * Rw**2)]
        ]

        self.dA['Cw'][0, :] = [4.0 / (self.sC * Rw * Cw**2), -2.0 / (self.sC * Rw * Cw**2)]

        self.dA['Ci'][1, :] = [-2.0 / (self.sC * Rw * Ci**2), 2.0 / (self.sC * Rw * Ci**2)]

        self.dB['Rw'][0, 0] = -2.0 / (self.sC * Cw * Rw**2)

        self.dB['Cw'][0, :2] = [-2.0 / (self.sC * Rw * Cw**2), -Aw / (self.sC * Cw**2)]

        self.dB['Ci'][1, 1:] = [-Ai / (self.sC * Ci**2), -1.0 / (self.sC * Ci**2), -cv / (self.sC * Ci**2)]

        self.dB['Aw'][0, 1] = 1. / (Cw * self.sC)

        self.dB['Ai'][1, 1] = 1. / (Ci * self.sC)

        self.dB['cv'][1, 3] = 1. / (Ci * self.sC)
