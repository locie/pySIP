import numpy as np
from .prior import Prior
from .parameter import Parameter


class Parameters:
    __transforms__ = ["none", "log", "lowup", "fixed", "lower", "upper"]

    def __init__(self, parameters, name=''):

        self._name = name

        if isinstance(parameters, dict):
            self._parameters = parameters
        elif isinstance(parameters[0], str):
            self._parameters = {name: Parameter(name) for name in parameters}
        else:
            self._parameters = {k['name']: Parameter(**k) for k in parameters}

    def __repr__(self):
        def _repr(d, level=0):
            s = ''
            if level == 0:
                s += 'Parameters ' + self._name + '\n'
            for k, v in d.items():
                s += '    ' * level
                if isinstance(v, Parameter):
                    s += v.__repr__() + '\n'
                else:
                    s += f'* {k}\n'
                    s += _repr(v, level + 1)
            return s
        return _repr(self._parameters)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n == len(self._parameters):
            raise StopIteration
        result = list(self._parameters.values())[self.n]
        self.n += 1
        return result

    def __add__(self, other):
        left_name = self._name if self._name else 'left'
        right_name = other._name if other._name else 'right'
        return Parameters(
            parameters={left_name: self._parameters,
                        right_name: other._parameters},
            name='__'.join((left_name, right_name))
        )

    def __eq__(self, other):
        for k in self._parameters:
            if k not in other._parameters:
                return False
            if self._parameters[k] != other._parameters[k]:
                return False
        return True

    @property
    def parameters(self):
        def _leafs(node):
            if isinstance(node, Parameter):
                return [node]
            return sum([_leafs(node[k]) for k in node], [])
        return _leafs(self._parameters)

    def set_parameter(self, *args, **kwargs):
        def _get(d, *args):
            if not d:
                return None
            if not args:
                return d
            return _get(d.get(args[0], {}), *args[1:])

        def _set(d, v, *args):
            if len(args) == 1:
                d[args[0]] = v
            else:
                _set(d[args[0]], v, *args[1:])

        parameter = _get(self._parameters, *args)
        if parameter:
            parameter = Parameter(**{**parameter.__dict__, **kwargs})
            _set(self._parameters, parameter, *args)

    @property
    def theta(self):
        """values of :math:`\\theta`"""
        return [h.theta for h in self.parameters]

    @theta.setter
    def theta(self, x):
        """set values to :math:`\\theta` and do the change in :math:`\\eta`"""
        for p, value in zip(self.parameters, x):
            p.theta = value

    def theta_by_name(self, names):
        """return ordered theta value corresponding to names"""
        return [h.value for n in names for h in self.parameters if h.name == n]

    @property
    def theta_free(self):
        """values of :math:`\\theta`"""
        return [p.value for p in self.parameters if p.free]

    @property
    def theta_jacobian(self):
        """Jacobian of :math:`\\theta`

        return only a vector because the jacobian matrix,
        :math:`J = \\frac{\\partial \\theta}{\\partial \\eta}` is diagonal.

        Chain rules and Jacobian adjustment are used to transform the gradient
        of the constrained space in the unconstrained space e.g.
        ..math:: \\frac{\\partial \\ln p(\\eta|y)}{\\partial \\eta} =
                 \\frac{\\partial \\ln p(\\theta|y)}{\\partial \\theta} J
                 + \\frac{\\partial \\ln |\\det(J)|}{\\partial \\eta}
        """
        return [p._inv_transform_jacobian() for p in self.parameters if p.free]

    @property
    def theta_log_jacobian(self):
        """logarithm of the determinant of the jacobian matrix:
            :math:`\\ln |\\det(J)|`
        """
        return np.sum([np.log(p._inv_transform_jacobian()) for p in self.parameters])

    @property
    def theta_dlog_jacobian(self):
        """partial derivative of the logarithm of jacobian matrix

        :math:`\\frac{\\partial \\ln |\\det(J)|}{\\partial \\eta}`
        """
        return [p._inv_transform_dlog_jacobian() for p in self.parameters if p.free]

    @property
    def theta_d2log_jacobian(self):
        """second partial derivative of the logarithm of jacobian matrix

        ..math::`\\frac{\\partial^{2} \\ln |\\det(J)|}{\\partial^{2} \\eta}`
        """
        return [p._inv_transform_d2log_jacobian() for p in self.parameters if p.free]

    @property
    def eta(self):
        """values of :math:`\\eta`"""
        return np.array([p.eta for p in self.parameters])

    @eta.setter
    def eta(self, x):
        """set values to :math:`\\eta` and do the change in :math:`\\theta`
        values can be set to :math:`\\eta` only if the corresponding parameters
        are not fixed"""
        for p, value in zip([_p for _p in self.parameters if _p.free], x):
            p.eta = value

    @property
    def eta_free(self):
        """values of free unconstrained parameters :math:`\\eta`"""
        return self.eta[self.free]

    @property
    def eta_jacobian(self):
        """Jacobian of :math:`\\eta`

        return only a vector because the jacobian matrix,
        :math:`J = \\frac{\\partial \\eta}{\\partial \\theta}` is diagonal.

        Chain rules and Jacobian adjustment are used to transform the gradient
        of the unconstrained space in the constrained space e.g.

        ..math:: \\frac{\\partial \\ln p(\\theta|y)}{\\partial \\theta} =
                 \\frac{\\partial \\ln p(\\eta|y)}{\\partial \\eta} J
                 + \\frac{\\partial \\ln |\\det(J)|}{\\partial \\theta}
        """
        return [h._transform_jacobian() for h in self.parameters
                if h.transform != "fixed"]

    @property
    def prior(self):
        """Evaluate the logarithm of the prior probability density function

        :math:`\\ln p(\\theta)`
        """
        return np.sum([h.prior.log_pdf(h.value) if h.prior is not None else 0.
                       for h in self.parameters if h.transform != "fixed"])

    @property
    def d_prior(self):
        """Evaluate the partial derivative of the logarithm of the prior
        probability density function

        :math:`\\frac{\\partial \\ln p(\\theta)}{\\partial \\theta}`
        """
        return ([h.prior.dlog_pdf(h.value) if h.prior is not None else 0.
                 for h in self.parameters if h.transform != "fixed"])

    @property
    def d2_prior(self):
        """Evaluate the second partial derivative of the logarithm of the prior
        probability density function

        :math:`\\frac{\\partial^{2} \\ln p(\\theta)}{\\partial^{2} \\theta}`
        """
        return ([h.prior.d2log_pdf(h.value) if h.prior is not None else 0.
                 for h in self.parameters if h.transform != "fixed"])

    @property
    def free(self):
        """return a boolean list of free parameters"""
        return [p.free for p in self.parameters]

    @property
    def names(self):
        """return parameter names"""
        return [p.name for p in self.parameters]

    @property
    def names_free(self):
        """return parameter names"""
        return [p.name for p in self.parameters if p.free]

    @property
    def penalty(self, scaling=1e-4):
        """penalty function"""
        return scaling * np.sum([p._penalty() for p in self.parameters
                                 if p.transform not in ["fixed", "none"]])

    @property
    def d_penalty(self, scaling=1e-4):
        """derivative of the penalty function"""
        return ([scaling * p._d_penalty() for p in self.parameters
                 if p.transform not in ["fixed", "none"]])

    def prior_init(self, prob_mass=None):
        """Generate random samples from the prior distributions"""

        return ([p.prior.random(1, prob_mass)[0] if p.prior is not None else
                 p.value for p in self.parameters])
