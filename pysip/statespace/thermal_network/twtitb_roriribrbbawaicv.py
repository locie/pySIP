from dataclasses import dataclass

from ..base import RCModel


@dataclass
class TwTiTb_RoRiRibRbbAwAicv(RCModel):
    """Third order RC model"""

    states = [
        ("TEMPERATURE", "xw", "envelope temperature"),
        ("TEMPERATURE", "xi", "indoor temperature"),
        ("TEMPERATURE", "xb", "boundary wall temperature"),
    ]

    params = [
        ("THERMAL_RESISTANCE", "Ro", "between the outdoor and the envelope"),
        ("THERMAL_RESISTANCE", "Ri", "between the envelope and the indoor"),
        ("THERMAL_RESISTANCE", "Rib", "between the indoor and the boundary wall"),
        (
            "THERMAL_RESISTANCE",
            "Rbb",
            "between the boundary wall and the boundary space",
        ),
        ("THERMAL_CAPACITY", "Cw", "of the envelope"),
        ("THERMAL_CAPACITY", "Ci", "of the indoor"),
        ("THERMAL_CAPACITY", "Cb", "of the wall between the indoor and the boundary"),
        ("SOLAR_APERTURE", "Aw", "of the envelope"),
        ("SOLAR_APERTURE", "Ai", "of the windows"),
        ("COEFFICIENT", "cv", "scaling of the heat from the ventilation"),
        ("STATE_DEVIATION", "sigw_w", "of the envelope dynamic"),
        ("STATE_DEVIATION", "sigw_i", "of the indoor dynamic"),
        ("STATE_DEVIATION", "sigw_b", "of the boundary wall dynamic"),
        ("MEASURE_DEVIATION", "sigv", "of the indoor temperature measurements"),
        ("INITIAL_MEAN", "x0_w", "of the envelope temperature"),
        ("INITIAL_MEAN", "x0_i", "of the infoor temperature"),
        ("INITIAL_MEAN", "x0_b", "of the boundary wall temperature"),
        ("INITIAL_DEVIATION", "sigx0_w", "of the envelope temperature"),
        ("INITIAL_DEVIATION", "sigx0_i", "of the infoor temperature"),
        ("INITIAL_DEVIATION", "sigx0_b", "of the boundary wall temperature"),
    ]

    inputs = [
        ("TEMPERATURE", "To", "outdoor temperature"),
        ("TEMPERATURE", "Tb", "boundary temperature"),
        ("POWER", "Qgh", "solar irradiance"),
        ("POWER", "Qh", "HVAC system heat"),
        ("POWER", "Qv", "heat from the ventilation system"),
    ]

    outputs = [("TEMPERATURE", "xi", "indoor temperature")]

    def __post_init__(self):
        super().__post_init__()

    def set_constant_continuous_ssm(self):
        self.C[0, 1] = 1.0

    def update_continuous_ssm(self):
        (
            Ro,
            Ri,
            Rib,
            Rbb,
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
            [1.0 / (Ci * Ri), -(Ri + Rib) / (Ci * Ri * Rib), 1.0 / (Ci * Rib)],
            [0.0, 1.0 / (Cb * Rib), -(Rib + Rbb) / (Cb * Rib * Rbb)],
        ]
        self.B[:] = [
            [1.0 / (Cw * Ro), 0.0, Aw / Cw, 0.0, 0.0],
            [0.0, 0.0, Ai / Ci, 1.0 / Ci, cv / Ci],
            [0.0, 1.0 / (Cb * Rbb), 0.0, 0.0, 0.0],
        ]
        self.Q[self._diag] = [sigw_w, sigw_i, sigw_b]
        self.R[0, 0] = sigv
        self.x0[:, 0] = [x0_w, x0_i, x0_b]
        self.P0[self._diag] = [sigx0_w, sigx0_i, sigx0_b]