from dataclasses import dataclass, field
from itertools import product

import numpy as np

from bopt.statespace.base import GPModel, StateSpace
from bopt.statespace.gaussian_process.periodic import Periodic
from bopt.statespace.nodes import Par


@dataclass
class GPProduct(GPModel):
    '''Product of two Gaussian Process Covariance'''

    def __init__(self, gp1, gp2):
        """Create a state-space model of appropriate dimensions

        Args:
            gp1, gp2: GPModel instance

        Notes:
            The MEASURE_DEVIATION and MAGNITUDE_SCALE of the `gp2` are fixed
            because they are already defined in `gp1`.

        """
        if not isinstance(gp1, GPModel):
            raise TypeError('`gp1` must be an GPModel instance')

        if not isinstance(gp2, GPModel):
            raise TypeError('`gp2` must be an GPModel instance')

        if not gp1.jacobian & gp2.jacobian:
            raise ValueError('`gp1.jacobian` and `gp2.jacobian` must have '
                             'the same boolean value')

        self._gp1 = gp1
        self._gp2 = gp2

        tmp = self._gp2.params.copy()
        for i, node in enumerate(self._gp2.params):
            if node.category == Par.MAGNITUDE_SCALE:
                self._gp2.parameters.set_parameter(
                    node.name, value=1.0, transform='fixed'
                )
                del tmp[i]
                break

        for i, node in enumerate(tmp):
            if node.category == Par.MEASURE_DEVIATION:
                self._gp2.parameters.set_parameter(
                    node.name, value=0.0, transform='fixed')
                del tmp[i]
                break

        self.parameters = self._gp1.parameters + self._gp2.parameters


        self.states = []
        for s1, s2 in product(self._gp1.states, self._gp2.states):
            state = (s1.category.name, s1.name + '__' + s2.name, s1.description + ' ' + s2.description)
            self.states.append(state)


        self.params = []
        for node in self._gp1.params:
            node.name = self._gp1.name + '__' + node.name
            self.params.append(node.unpack())
        for node in self._gp2.params:
            node.name = self._gp2.name + '__' + node.name
            self.params.append(node.unpack())

        self.inputs = []

        if np.sum(self._gp1.C.squeeze()) + np.sum(self._gp2.C.squeeze()) > 2:
            self.outputs = [
                ('ANY', 'sum(f(t))', 'sum of stochastic processes')
            ]
        else:
            self.outputs = [
                ('ANY', 'f(t)', 'stochastic processes')
            ]

        self._I1 = np.eye(self._gp1.Nx)
        self._I2 = np.eye(self._gp2.Nx)

        self.is_periodic1 = isinstance(self._gp1, Periodic)
        self.is_periodic_2 = isinstance(self._gp2, Periodic)

        super().__post_init__()

    def set_constant(self):
        self._gp1.set_constant()
        self._gp2.set_constant()

    def set_jacobian(self):
        self._gp1.set_jacobian()
        self._gp2.set_jacobian()

    def update_state_space_model(self):
        self._gp1.update_state_space_model()
        self._gp2.update_state_space_model()

        self.A = np.kron(self._gp1.A, self._I2) + \
            np.kron(self._I1, self._gp2.A)

        self.C = np.kron(self._gp1.C, self._gp2.C)

        # special case for Periodic covariance
        if self.is_periodic1 and not self.is_periodic_2:
            self.Q = np.kron(self._gp1.P0, self._gp2.Q)

        elif not self.is_periodic1 and self.is_periodic_2:
            self.Q = np.kron(self._gp1.Q, self._gp2.P0)

        elif self.is_periodic1 and self.is_periodic_2:
            self.Q = np.kron(self._gp1.P0, self._gp2.P0)

        else:
            self.Q = np.kron(self._gp1.Q, self._gp2.Q)

        self.R = self._gp1.R

        self.P0 = np.kron(self._gp1.P0, self._gp2.P0)

    def update_jacobian(self):
        self._gp1.update_jacobian()
        self._gp2.update_jacobian()

        s1 = self._gp1.name + '__'
        s2 = self._gp2.name + '__'

        # loop over gp1 keys
        for n in self._gp1._names:
            self.dA[s1 + n] = np.kron(self._gp1.dA[n], self._I2)

            # special case for Periodic covariance
            if self.is_periodic1 and not self.is_periodic_2:
                self.dQ[s1 + n] = np.kron(self._gp1.dP0[n], self._gp2.Q)

            elif not self.is_periodic1 and self.is_periodic_2:
                self.dQ[s1 + n] = np.kron(self._gp1.dQ[n], self._gp2.P0)

            elif self.is_periodic1 and self.is_periodic_2:
                self.dQ[s1 + n] = np.kron(self._gp1.dP0[n], self._gp2.P0)

            else:
                self.dQ[s1 + n] = np.kron(self._gp1.dQ[n], self._gp2.Q)

            self.dP0[s1 + n] = np.kron(self._gp1.dP0[n], self._gp2.P0)

        # loop over gp2 keys
        for n in self._gp2._names:
            self.dA[s2 + n] = np.kron(self._I1, self._gp2.dA[n])

            # special case for Periodic covariance
            if self.is_periodic1 and not self.is_periodic_2:
                self.dQ[s2 + n] = np.kron(self._gp1.P0, self._gp2.dQ[n])

            elif not self.is_periodic1 and self.is_periodic_2:
                self.dQ[s2 + n] = np.kron(self._gp1.Q, self._gp2.dP0[n])

            elif self.is_periodic1 and self.is_periodic_2:
                self.dQ[s2 + n] = np.kron(self._gp1.P0, self._gp2.dP0[n])

            else:
                self.dQ[s2 + n] = np.kron(self._gp1.Q, self._gp2.dQ[n])

            self.dP0[s2 + n] = np.kron(self._gp1.P0, self._gp2.dP0[n])

if __name__ == "__main__":
    from bopt.statespace.gaussian_process.gp_product import GPProduct
    from bopt.statespace.gaussian_process.periodic import Periodic
    from bopt.statespace.gaussian_process.matern import Matern32

    out = GPProduct(Periodic(), Matern32())
    out.parameters.theta = np.random.random(7)
    out.update()

    # the jacobian must have as many keys as out.params
    # check out.dB for instance
