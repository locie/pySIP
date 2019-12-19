from dataclasses import dataclass, field
from ..nodes import Par
from ..base import GPModel


@dataclass
class GPSum(GPModel):
    '''Sum of two Gaussian Process model

    Args:
        gp1: GPModel instance
        gp2: GPModel instance

    Notes:
        The MEASURE_DEVIATION of `gp2` is fixed because it is already defined in `gp1`.
    '''

    def __init__(self, gp1: GPModel, gp2: GPModel):

        if not isinstance(gp1, GPModel):
            raise TypeError('`gp1` must be an GPModel instance')

        if not isinstance(gp2, GPModel):
            raise TypeError('`gp2` must be an GPModel instance')

        self._gp1 = gp1
        self._gp2 = gp2

        for node in self._gp2.params:
            if node.category == Par.MEASURE_DEVIATION:
                self._gp2.parameters.set_parameter(node.name, value=0.0, transform='fixed')
                break

        self.parameters = self._gp1.parameters + self._gp2.parameters

        _states = self._gp1.states + self._gp2.states
        self.states = []
        for s in _states:
            self.states.append(s.unpack())

        self.params = []
        for node in self._gp1.params:
            node.name = self._gp1.name + '__' + node.name
            self.params.append(node.unpack())
        for node in self._gp2.params:
            node.name = self._gp2.name + '__' + node.name
            self.params.append(node.unpack())

        self.inputs = []

        self.outputs = [('ANY', 'sum(f(t))', 'sum of stochastic processes')]

        self.name = self._gp1.name + '__+__' + self._gp2.name

        super().__post_init__()

    def init_continuous_dssm(self):
        self._init_continuous_dssm()
        self._gp1._init_continuous_dssm()
        self._gp2._init_continuous_dssm()
        self.set_constant_continuous_dssm()

    def delete_continuous_dssm(self):
        self._gp1.delete_continuous_dssm()
        self._gp2.delete_continuous_dssm()
        self._delete_continuous_dssm()

    def set_constant_continuous_ssm(self):
        self._gp1.set_constant_continuous_ssm()
        self._gp2.set_constant_continuous_ssm()

    def set_constant_continuous_dssm(self):
        self._gp1.set_constant_continuous_dssm()
        self._gp2.set_constant_continuous_dssm()

    def update_continuous_ssm(self):
        self._gp1.update_continuous_ssm()
        self._gp2.update_continuous_ssm()

        self.A[: self._gp1.nx, : self._gp1.nx] = self._gp1.A
        self.A[self._gp1.nx :, self._gp1.nx :] = self._gp2.A

        self.C[:, : self._gp1.nx] = self._gp1.C
        self.C[:, self._gp1.nx :] = self._gp2.C

        self.Q[: self._gp1.nx, : self._gp1.nx] = self._gp1.Q
        self.Q[self._gp1.nx :, self._gp1.nx :] = self._gp2.Q

        self.R = self._gp1.R

        self.P0[: self._gp1.nx, : self._gp1.nx] = self._gp1.P0
        self.P0[self._gp1.nx :, self._gp1.nx :] = self._gp2.P0

    def update_continuous_dssm(self):
        self._gp1.update_continuous_dssm()
        self._gp2.update_continuous_dssm()

        s1 = self._gp1.name + '__'
        s2 = self._gp2.name + '__'

        for n in self._gp1._names:
            self.dA[s1 + n][: self._gp1.nx, : self._gp1.nx] = self._gp1.dA[n]
            self.dQ[s1 + n][: self._gp1.nx, : self._gp1.nx] = self._gp1.dQ[n]
            self.dR[s1 + n] = self._gp1.dR[n]
            self.dP0[s1 + n][: self._gp1.nx, : self._gp1.nx] = self._gp1.dP0[n]

        for n in self._gp2._names:
            self.dA[s2 + n][self._gp1.nx :, self._gp1.nx :] = self._gp2.dA[n]
            self.dQ[s2 + n][self._gp1.nx :, self._gp1.nx :] = self._gp2.dQ[n]
            self.dR[s2 + n] = self._gp2.dR[n]
            self.dP0[s2 + n][self._gp1.nx :, self._gp1.nx :] = self._gp2.dP0[n]
