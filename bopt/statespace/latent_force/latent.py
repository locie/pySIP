from dataclasses import dataclass, field
import numpy as np
from bopt.statespace.base import StateSpace, RCModel, GPModel
from bopt.statespace.nodes import Par


@dataclass
class LatentForceModel(StateSpace):
    """Latent Force Model (LFM)

    The LFM is a combination of a physical model and a Gaussian process in
    stochastic differential equation (SDE) representation.
    """

    def __init__(self, rc, gp, latent_force):
        """Create a state-space model of appropriate dimensions

        Args:
            rc: RCModel instance
            gp: GPModel instance
            latent_force: String or list of strings of the RCModel inputs which
                are considered as latent forces

        Notes:
            The MEASURE_DEVIATION of the GPModel is fixed because it is not
            used in the latent force model. The GPModel is augmented into the
            RCModel, therefore, only the measurement noise matrix `R` of the
            RCModel is used. To avoid useless computation, the
            MEASURE_DEVIATION of the GPModel must stay fixed.

        TODO
            Be careful with overrided names in jacobian dict
        """
        if not isinstance(rc, RCModel):
            raise TypeError('`rc` must be an RCModel instance')

        if not isinstance(gp, GPModel):
            raise TypeError('`gp` must be an GPModel instance')

        if not isinstance(latent_force, (str, list)):
            raise TypeError(
                '`latent_force` must be a string or a list of strings'
            )

        if isinstance(latent_force, str):
            latent_force = [latent_force]

        if not rc.jacobian & gp.jacobian:
            raise ValueError('The jacobian attribute of `rc` and `gp` must '
                             'have the same boolean value')

        if not rc.inputs:
            raise ValueError('The `rc` model has no input')

        index = []
        for i, node in enumerate(rc.inputs):
            if node.name in latent_force:
                index.append(i)
        if not index:
            raise ValueError('`latent_force` is not an input of `rc`')

        self.rc = rc
        self.gp = gp

        # The GP have only one output
        tmp = self.gp.params.copy()
        for i, node in enumerate(self.gp.params):
            if node.category == Par.MEASURE_DEVIATION:
                self.gp.parameters.set_parameter(
                    node.name, value=0.0, transform='fixed'
                )
                del tmp[i]
                break

        self.rc.parameters._name = self.rc.__class__.__name__
        self.gp.parameters._name = self.gp.__class__.__name__
        self.parameters = self.rc.parameters + self.gp.parameters
        self.states = self.rc.states + self.gp.states
        self.params = self.rc.params + tmp
        self.inputs = self.rc.inputs.copy()
        for i in sorted(index, reverse=True):
            del self.inputs[i]
        self.outputs = self.rc.outputs.copy()

        # Unpack list of Node as a list of string to rebuild list of Node
        self.states = [s.unpack() for s in self.states]
        self.params = [s.unpack() for s in self.params]
        self.inputs = [s.unpack() for s in self.inputs]
        self.outputs = [s.unpack() for s in self.outputs]

        # slicing columns in the input matrix corresponding to latent forces
        self.idx = np.full((self.rc.Nu), False)
        self.idx[index] = True

        super().__post_init__()

    def set_constant(self):
        self.rc.set_constant()
        self.gp.set_constant()

    def set_jacobian(self):
        self.rc.set_jacobian()
        self.gp.set_jacobian()

    def update_state_space_model(self):
        self.rc.update_state_space_model()
        self.gp.update_state_space_model()

        self.A[:self.rc.Nx, :self.rc.Nx] = self.rc.A
        self.A[self.rc.Nx:, self.rc.Nx:] = self.gp.A
        self.A[:self.rc.Nx, self.rc.Nx:] = self.rc.B[:, self.idx] @ self.gp.C

        self.B[:self.rc.Nx, :] = self.rc.B[:, ~self.idx]

        self.C[:, :self.rc.Nx] = self.rc.C

        self.D = self.rc.D[:, ~self.idx]

        self.Q[:self.rc.Nx, :self.rc.Nx] = self.rc.Q
        self.Q[self.rc.Nx:, self.rc.Nx:] = self.gp.Q

        self.R = self.rc.R

        self.x0[:self.rc.Nx] = self.rc.x0
        self.x0[self.rc.Nx:] = self.gp.x0

        self.P0[:self.rc.Nx, :self.rc.Nx] = self.rc.P0
        self.P0[self.rc.Nx:, self.rc.Nx:] = self.gp.P0

    def update_jacobian(self):
        self.rc.update_jacobian()
        self.gp.update_jacobian()

        for k in self.rc._names:
            self.dA[k][:self.rc.Nx, :self.rc.Nx] = self.rc.dA[k]
            self.dA[k][:self.rc.Nx, self.rc.Nx:] = (
                self.rc.dB[k][:, self.idx] @ self.gp.C)

            self.dB[k][:self.rc.Nx, :] = self.rc.dB[k][:, ~self.idx]

            self.dC[k][:, :self.rc.Nx] = self.rc.dC[k]

            self.dD[k] = self.rc.dD[k][:, ~self.idx]

            self.dQ[k][:self.rc.Nx, :self.rc.Nx] = self.rc.dQ[k]

            self.dR[k] = self.rc.dR[k]

            self.dx0[k][:self.rc.Nx] = self.rc.dx0[k]

            self.dP0[k][:self.rc.Nx, :self.rc.Nx] = self.rc.dP0[k]

        for k in self.gp._names:
            self.dA[k][self.rc.Nx:, self.rc.Nx:] = self.gp.dA[k]

            self.dQ[k][self.rc.Nx:, self.rc.Nx:] = self.gp.dQ[k]

            self.dx0[k][self.rc.Nx:] = self.gp.dx0[k]

            self.dP0[k][self.rc.Nx:, self.rc.Nx:] = self.gp.dP0[k]


@dataclass
class R2C2_Qgh_Matern32(RCModel):
    """LFM: R2C2_Qgh + Matern32

      The Matérn kernel with smoothness parameter = 3/2 is used to model the
      the solar radiation as a latent force in the model R2C2.

      This LFM is built by hand and allows to check the creation of LFM in
      bopt.Latent_force_model.

      """

    states = [
        ('TEMPERATURE', 'xw', 'wall temperature'),
        ('TEMPERATURE', 'xi', 'indoor space temperature'),
        ('ANY', 'f(t)', 'stochastic process'),
        ('ANY', 'df(t)/dt', 'derivative stochastic process')
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
        ('MAGNITUDE_SCALE', 'mscale', ''),
        ('LENGTH_SCALE', 'lscale', '')
    ]

    inputs = [
        ('TEMPERATURE', 'To', 'outdoor air temperature'),
        ('POWER', 'Qh', 'HVAC system heat')
    ]

    outputs = [
        ('TEMPERATURE', 'xi', 'indoor air temperature')
    ]

    def set_constant(self):
        self.A[2, 3] = 1.0
        self.C[0, 1] = 1.0

    def set_jacobian(self):
        self.dQ["sigw_w"][0, 0] = 1.0
        self.dQ["sigw_i"][1, 1] = 1.0
        self.dR["sigv"][0, 0] = 1.0
        self.dx0["x0_w"][0, 0] = self.sX
        self.dx0["x0_i"][1, 0] = self.sX
        self.dP0["sigx0_w"][0, 0] = 1.0
        self.dP0["sigx0_i"][1, 1] = 1.0
        self.dP0["mscale"][2, 2] = 1.0

    def update_state_space_model(self):
        (Ro, Ri, Cw, Ci, sigw_w, sigw_i, sigv, x0_w, x0_i,
         sigx0_w, sigx0_i, mscale, lscale, *_) = self.parameters.theta

        self.A[0, :3] = [-(Ro+Ri) / (Cw*Ri*Ro*self.sC),
                         1.0 / (Cw*Ri*self.sC),
                         1.0 / (Cw*self.sC)]

        self.A[1, :3] = [1.0 / (Ci*Ri*self.sC),
                         -1.0 / (Ci*Ri*self.sC),
                         1.0 / (Ci*self.sC)]

        self.A[3, 2:] = [-3.0 / lscale**2, -2.0*3.0**0.5 / lscale]

        self.B[0, 0] = 1.0 / (Cw*Ro*self.sC)
        self.B[1, 1] = 1.0 / (Ci*self.sC)

        self.Q[self._diag] = [
            sigw_w, sigw_i, 0.0, 2.0*3.0**0.75*mscale / lscale**1.5
        ]

        self.R[0, 0] = sigv

        self.x0[:2, 0] = [x0_w*self.sX, x0_i*self.sX]

        self.P0[self._diag] = [
            sigx0_w, sigx0_i, mscale, 3.0**0.5*mscale / lscale
        ]

    def update_jacobian(self):
        (Ro, Ri, Cw, Ci, _, _, _, _, _, _, _,
         mscale, lscale, *_) = self.parameters.theta

        self.dA["Ro"][0, 0] = 1.0 / (Cw*Ro**2*self.sC)

        self.dA["Ri"][:2, :2] = [
            [1.0 / (Cw*Ri**2*self.sC), -1.0 / (Cw*Ri**2*self.sC)],
            [-1.0 / (Ci*Ri**2*self.sC), 1.0 / (Ci*Ri**2*self.sC)]
        ]

        self.dA["Cw"][0, :3] = [(Ro+Ri) / (Cw**2*Ri*Ro*self.sC),
                                -1.0 / (Cw**2*Ri*self.sC),
                                -1.0 / (Cw**2*self.sC)]

        self.dA["Ci"][1, :3] = [-1.0 / (Ci**2*Ri*self.sC),
                                1.0 / (Ci**2*Ri*self.sC),
                                -1.0 / (Ci**2*self.sC)]

        self.dA["lscale"][3, 2:] = [6.0 / lscale**3, 2.0*3.0**0.5 / lscale**2]

        self.dB["Ro"][0, 0] = -1.0 / (Cw*Ro**2*self.sC)

        self.dB["Cw"][0, 0] = -1.0 / (Cw**2*Ro*self.sC)

        self.dB["Ci"][1, 1] = -1.0 / (Ci**2*self.sC)

        self.dQ["mscale"][3, 3] = 2.0*3.0**0.75 / lscale**1.5

        self.dQ["lscale"][3, 3] = -3.0**1.75*mscale / lscale**2.5

        self.dP0["mscale"][3, 3] = 3.0**0.5 / lscale

        self.dP0["lscale"][3, 3] = -3.0**0.5*mscale / lscale**2
