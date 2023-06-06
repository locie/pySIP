from dataclasses import dataclass

from ..base import RCModel


@dataclass
class TiTm_RwRmAi(RCModel):
    """Second order RC model"""

    states = [
        ("TEMPERATURE", "xi", "indoor temperature"),
        ("TEMPERATURE", "xm", "internal mass temperature"),
    ]

    params = [
        ("THERMAL_RESISTANCE", "Rw", "between the outdoor and the indoor"),
        ("THERMAL_RESISTANCE", "Rm", "between the indoor and the internal mass"),
        ("THERMAL_CAPACITY", "Ci", "of the indoor"),
        ("THERMAL_CAPACITY", "Cm", "of the internal mass"),
        ("SOLAR_APERTURE", "Ai", "effective solar aperture"),
        ("STATE_DEVIATION", "sigw_i", "of the indoor dynamic"),
        ("STATE_DEVIATION", "sigw_m", "of the internal mass dynamic"),
        ("MEASURE_DEVIATION", "sigv", "of the indoor temperature measurements"),
        ("INITIAL_MEAN", "x0_i", "of the infoor temperature"),
        ("INITIAL_MEAN", "x0_m", "of the internal mass temperature"),
        ("INITIAL_DEVIATION", "sigx0_i", "of the infoor temperature"),
        ("INITIAL_DEVIATION", "sigx0_m", "of the internal mass temperature"),
    ]

    inputs = [
        ("TEMPERATURE", "To", "outdoor temperature"),
        ("POWER", "Qgh", "solar irradiance"),
        ("POWER", "Qh", "HVAC system heat"),
    ]

    outputs = [("TEMPERATURE", "xi", "indoor temperature")]

    def __post_init__(self):
        super().__post_init__()

    def set_constant_continuous_ssm(self):
        self.C[0, 0] = 1.0

    def update_continuous_ssm(self):
        (
            Rw,
            Rm,
            Ci,
            Cm,
            Ai,
            sigw_i,
            sigw_m,
            sigv,
            x0_i,
            x0_m,
            sigx0_i,
            sigx0_m,
            *_,
        ) = self.parameters.theta

        self.A[:] = [
            [-(Rm + Rw) / (Ci * Rm * Rw), 1.0 / (Ci * Rm)],
            [1.0 / (Cm * Rm), -1.0 / (Cm * Rm)],
        ]
        self.B[0, :] = [1.0 / (Ci * Rw), Ai / Ci, 1.0 / Ci]
        self.Q[self._diag] = [sigw_i, sigw_m]
        self.R[0, 0] = sigv
        self.x0[:, 0] = [x0_i, x0_m]
        self.P0[self._diag] = [sigx0_i, sigx0_m]