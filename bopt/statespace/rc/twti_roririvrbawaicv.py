from dataclasses import dataclass

from ..base import RCModel


@dataclass
class TwTi_RoRiRivRbAwAicv(RCModel):
    """Second order RC model"""

    states = [
        ('TEMPERATURE', 'xw', 'wall temperature'),
        ('TEMPERATURE', 'xi', 'indoor space temperature')
    ]

    params = [
        ('THERMAL_RESISTANCE', 'Ro', 'between the outdoor and the wall node'),
        ('THERMAL_RESISTANCE', 'Ri', 'between the wall node and the indoor'),
        ('THERMAL_RESISTANCE', 'Riv', 'between the outdoor and the indoor'),
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
        ('INITIAL_DEVIATION', 'sigx0_i', '')
    ]

    inputs = [
        ('TEMPERATURE', 'To', 'outdoor air temperature'),
        ('TEMPERATURE', 'Tb', 'boundary air temperature'),
        ('POWER', 'Qgh', 'global horizontal solar radiation'),
        ('POWER', 'Qh', 'HVAC system heat'),
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
        (Ro, Ri, Riv, Rb, Cw, Ci, Aw, Ai, cv, sigw_w, sigw_i, sigv, x0_w, x0_i, sigx0_w, sigx0_i, *_) = self.parameters.theta

        self.A[:] = [
            [-(Ro + Ri) / (self.sC * Cw * Ri * Ro), 1.0 / (self.sC * Cw * Ri)],
            [1.0 / (self.sC * Ci * Ri), -(Rb * Ri + Rb * Riv + Ri * Riv) / (self.sC * Ci * Rb * Ri * Riv)]
        ]

        self.B[0, :3] = [1.0 / (self.sC * Cw * Ro), 0.0, Aw / (self.sC * Cw)]
        self.B[1, :] = [
            1.0 / (self.sC * Ci * Riv),
            1.0 / (self.sC * Ci * Rb),
            Ai / (self.sC * Ci),
            1.0 / (self.sC * Ci),
            cv / (self.sC * Ci)
        ]

        self.Q[self._diag] = [sigw_w, sigw_i]

        self.R[0, 0] = sigv

        self.x0[:, 0] = (self.sX * x0_w, self.sX * x0_i)

        self.P0[self._diag] = (sigx0_w, sigx0_i)

    def update_jacobian(self):
        (Ro, Ri, Riv, Rb, Cw, Ci, Aw, Ai, cv, *_) = self.parameters.theta

        self.dA['Ro'][0, 0] = 1.0 / (self.sC * Cw * Ro**2)

        self.dA['Ri'][:] = [
            [1.0 / (self.sC * Cw * Ri**2), -1.0 / (self.sC * Cw * Ri**2)],
            [-1.0 / (self.sC * Ci * Ri**2), 1.0 / (self.sC * Ci * Ri**2)]
        ]

        self.dA['Riv'][1, 1] = 1.0 / (self.sC * Ci * Riv**2)

        self.dA['Rb'][1, 1] = 1.0 / (self.sC * Ci * Rb**2)

        self.dA['Cw'][0, :] = [(Ro + Ri) / (self.sC * Ri * Ro * Cw**2), -1.0 / (self.sC * Ri * Cw**2)]

        self.dA['Ci'][1, :] = [-1.0 / (self.sC * Ri * Ci**2), (Rb * Ri + Rb * Riv + Ri * Riv) / (self.sC * Rb * Ri * Riv * Ci**2)]

        self.dB['Ro'][0, 0] = -1.0 / (self.sC * Cw * Ro**2)

        self.dB['Riv'][1, 0] = -1.0 / (self.sC * Ci * Riv**2)

        self.dB['Rb'][1, 1] = -1.0 / (self.sC * Ci * Rb**2)

        self.dB['Cw'][0, :3] = [-1.0 / (self.sC * Ro * Cw**2), 0.0, -Aw / (self.sC * Cw**2)]

        self.dB['Ci'][1, :] = [
            -1.0 / (self.sC * Riv * Ci**2),
            -1.0 / (self.sC * Rb * Ci**2),
            -Ai / (self.sC * Ci**2),
            -1.0 / (self.sC * Ci**2),
            -cv / (self.sC * Ci**2)
        ]

        self.dB['Aw'][0, 2] = 1.0 / (self.sC * Cw)

        self.dB['Ai'][1, 2] = 1.0 / (self.sC * Ci)

        self.dB['cv'][1, 4] = 1.0 / (self.sC * Ci)
