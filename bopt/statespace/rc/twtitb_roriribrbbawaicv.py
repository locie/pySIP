from dataclasses import dataclass, field

from ..base import RCModel


@dataclass
class TwTiTb_RoRiRibRbbAwAicv(RCModel):
    """Second order RC model"""

    states = [
        ('TEMPERATURE', 'xw', 'wall temperature'),
        ('TEMPERATURE', 'xi', 'indoor space temperature'),
        ('TEMPERATURE', 'xb', 'wall temperature between the indoor and the boundary space'),
    ]

    params = [
        ('THERMAL_RESISTANCE', 'Ro', 'between the outdoor and the wall node'),
        ('THERMAL_RESISTANCE', 'Ri', 'between the wall node and the indoor'),
        ('THERMAL_RESISTANCE', 'Rib', 'between the indoor and the wall of the boundary space'),
        ('THERMAL_RESISTANCE', 'Rbb', 'between the wall of the boundary space and the boundary space'),
        ('THERMAL_CAPACITY', 'Cw', 'of the wall'),
        ('THERMAL_CAPACITY', 'Ci', 'of the indoor air, indoor walls, furnitures, etc.'),
        ('THERMAL_CAPACITY', 'Cb', 'of the wall between the indoor and the boundary space'),
        ('SOLAR_APERTURE', 'Aw', 'of the wall'),
        ('SOLAR_APERTURE', 'Ai', 'of the windows'),
        ('COEFFICIENT', 'cv', 'ventilation heating contribution scaling'),
        ('STATE_DEVIATION', 'sigw_w', ''),
        ('STATE_DEVIATION', 'sigw_i', ''),
        ('STATE_DEVIATION', 'sigw_b', ''),
        ('MEASURE_DEVIATION', 'sigv', ''),
        ('INITIAL_MEAN', 'x0_w', ''),
        ('INITIAL_MEAN', 'x0_i', ''),
        ('INITIAL_MEAN', 'x0_b', ''),
        ('INITIAL_DEVIATION', 'sigx0_w', ''),
        ('INITIAL_DEVIATION', 'sigx0_i', ''),
        ('INITIAL_DEVIATION', 'sigx0_b', ''),
    ]

    inputs = [
        ('TEMPERATURE', 'To', 'outdoor air temperature'),
        ('TEMPERATURE', 'Tb', 'boundary air temperature'),
        ('POWER', 'Qgh', 'global horizontal solar radiation'),
        ('POWER', 'Qh', 'HVAC system heat'),
        ('POWER', 'Qv', 'heat from the ventilation system'),
    ]

    outputs = [
        ('TEMPERATURE', 'xi', 'indoor air temperature')
    ]

    def set_constant(self):
        self.C[0, 1] = 1.0

    def set_jacobian(self):
        self.dQ['sigw_w'][0, 0] = 1.0
        self.dQ['sigw_i'][1, 1] = 1.0
        self.dQ['sigw_b'][2, 2] = 1.0
        self.dR['sigv'][0, 0] = 1.0
        self.dx0['x0_w'][0, 0] = self.sX
        self.dx0['x0_i'][1, 0] = self.sX
        self.dx0['x0_b'][2, 0] = self.sX
        self.dP0['sigx0_w'][0, 0] = 1.0
        self.dP0['sigx0_i'][1, 1] = 1.0
        self.dP0['sigx0_b'][2, 2] = 1.0

    def update_state_space_model(self):
        (Ro, Ri, Rib, Rbb, Cw, Ci, Cb, Aw, Ai, cv, sigw_w, sigw_i, sigw_b, sigv, x0_w, x0_i, x0_b, sigx0_w, sigx0_i, sigx0_b, *_) = self.parameters.theta

        self.A[:] = [
            [-(Ro + Ri) / (self.sC * Cw * Ri * Ro), 1.0 / (self.sC * Cw * Ri), 0.0],
            [1.0 / (self.sC * Ci * Ri), -(Ri + Rib) / (self.sC * Ci * Ri * Rib), 1.0 / (self.sC * Ci * Rib)],
            [0.0, 1.0 / (self.sC * Cb * Rib), -(Rib + Rbb) / (self.sC * Cb * Rib * Rbb)]
        ]

        self.B[:] = [
            [1.0 / (self.sC * Cw * Ro), 0.0, Aw / (self.sC * Cw), 0.0, 0.0],
            [0.0, 0.0, Ai / (self.sC * Ci), 1.0 / (self.sC * Ci), cv / (self.sC * Ci)],
            [0.0, 1.0 / (self.sC * Cb * Rbb), 0.0, 0.0, 0.0]
        ]

        self.Q[self._diag] = [sigw_w, sigw_i, sigw_b]

        self.R[0, 0] = sigv

        self.x0[:, 0] = [self.sX * x0_w, self.sX * x0_i, self.sX * x0_b]

        self.P0[self._diag] = [sigx0_w, sigx0_i, sigx0_b]

    def update_jacobian(self):
        (Ro, Ri, Rib, Rbb, Cw, Ci, Cb, Aw, Ai, cv, *_) = self.parameters.theta

        self.dA['Ro'][0, 0] = 1.0 / (self.sC * Cw * Ro**2)

        self.dA['Ri'][:2, :2] = [
            [1.0 / (self.sC * Cw * Ri**2), -1.0 / (self.sC * Cw * Ri**2)],
            [-1.0 / (self.sC * Ci * Ri**2), 1.0 / (self.sC * Ci * Ri**2)]
        ]

        self.dA['Rib'][1:, 1:] = [
            [1.0 / (self.sC * Ci * Rib**2), -1.0 / (self.sC * Ci * Rib**2)],
            [-1.0 / (self.sC * Cb * Rib**2), 1.0 / (self.sC * Cb * Rib**2)]
        ]

        self.dA['Rbb'][2, 2] = 1.0 / (self.sC * Cb * Rbb**2)

        self.dA['Cw'][0, :2] = [(Ro + Ri) / (self.sC * Ri * Ro * Cw**2), -1.0 / (self.sC * Ri * Cw**2)]

        self.dA['Ci'][1, :] = (
            -1.0 / (self.sC * Ri * Ci**2),
            (Ri + Rib) / (self.sC * Ri * Rib * Ci**2),
            -1.0 / (self.sC * Rib * Ci**2)
        )

        self.dA['Cb'][2, 1:] = [
            -1.0 / (self.sC * Rib * Cb**2), (Rib + Rbb) / (self.sC * Rib * Rbb * Cb**2)
        ]

        self.dB['Ro'][0, 0] = -1.0 / (self.sC * Cw * Ro**2)

        self.dB['Rbb'][2, 1] = -1.0 / (self.sC * Cb * Rbb**2)

        self.dB['Cw'][0, :3] = [-1.0 / (self.sC * Ro * Cw**2), 0.0, -Aw / (self.sC * Cw**2)]

        self.dB['Ci'][1, 2:] = [-Ai / (self.sC * Ci**2), -1.0 / (self.sC * Ci**2), -cv / (self.sC * Ci**2)]

        self.dB['Cb'][2, 1] = -1.0 / (self.sC * Rbb * Cb**2)

        self.dB['Aw'][0, 2] = 1.0 / (self.sC * Cw)

        self.dB['Ai'][1, 2] = 1.0 / (self.sC * Ci)

        self.dB['cv'][1, 4] = 1.0 / (self.sC * Ci)
