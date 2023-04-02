from dataclasses import dataclass
from itertools import product

import numpy as np

from ..base_statespace import GPModel
from ..nodes import Par
from .periodic import Periodic


@dataclass
class GPProduct(GPModel):
    """Product of two Gaussian Process Covariance

    Args:
        gp1: GPModel instance
        gp2: GPModel instance

    Notes:
        The MEASURE_DEVIATION and MAGNITUDE_SCALE of the `gp2` are fixed
        because they are already defined in `gp1`.
    """

    def __init__(self, gp1: GPModel, gp2: GPModel):
        if not isinstance(gp1, GPModel):
            raise TypeError("`gp1` must be an GPModel instance")

        if not isinstance(gp2, GPModel):
            raise TypeError("`gp2` must be an GPModel instance")

        self._gp1 = gp1
        self._gp2 = gp2

        for node in self._gp2.params:
            if node.category == Par.MAGNITUDE_SCALE:
                self._gp2.parameters.set_parameter(
                    node.name, value=1.0, transform="fixed"
                )
            if node.category == Par.MEASURE_DEVIATION:
                self._gp2.parameters.set_parameter(
                    node.name, value=0.0, transform="fixed"
                )

        self.parameters = self._gp1.parameters + self._gp2.parameters

        self.states = []
        for s1, s2 in product(self._gp1.states, self._gp2.states):
            state = (
                s1.category.name,
                s1.name + "*" + s2.name,
                s1.description + "*" + s2.description,
            )
            self.states.append(state)

        self.name = self._gp1.name + "__x__" + self._gp2.name

        self.params = []
        for node in self._gp1.params:
            node.name = self._gp1.name + "__" + node.name
            self.params.append(node.unpack())
        for node in self._gp2.params:
            node.name = self._gp2.name + "__" + node.name
            self.params.append(node.unpack())

        self.inputs = []

        if np.sum(self._gp1.C.squeeze()) + np.sum(self._gp2.C.squeeze()) > 2:
            self.outputs = [("ANY", "sum(f(t))", "sum of stochastic processes")]
        else:
            self.outputs = [("ANY", "f(t)", "stochastic processes")]

        self._I1 = np.eye(self._gp1.nx)
        self._I2 = np.eye(self._gp2.nx)

        # special case for Periodic covariance
        if isinstance(self._gp1, Periodic):
            self._Q1 = self._gp1.P0
        else:
            self._Q1 = self._gp1.Q

        if isinstance(self._gp2, Periodic):
            self._Q2 = self._gp2.P0
        else:
            self._Q2 = self._gp2.Q

        super().__post_init__()
    def set_constant_continuous_ssm(self):
        self._gp1.set_constant_continuous_ssm()
        self._gp2.set_constant_continuous_ssm()

    def update_continuous_ssm(self):
        self._gp1.update_continuous_ssm()
        self._gp2.update_continuous_ssm()

        self.A = np.kron(self._gp1.A, self._I2) + np.kron(self._I1, self._gp2.A)
        self.C = np.kron(self._gp1.C, self._gp2.C)
        self.Q = np.kron(self._Q1, self._Q2)
        self.R = self._gp1.R
        self.P0 = np.kron(self._gp1.P0, self._gp2.P0)
