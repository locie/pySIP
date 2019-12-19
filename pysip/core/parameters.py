import numpy as np
from typing import Union, Tuple, List
from .prior import Prior
from .parameter import Parameter


class Parameters:
    """Factory of Parameter instances

    Args:
        parameters: There is two options for instantiating Parameters: `parameters` is a list of
            strings corresponding to the parameters names. In this case all the parameters have the
            default settings; `parameters` is a list of dictionaries, where the arguments of
            Parameter can be modified as key-value pairs
        name: Name of this specific instance

    Notes:
        Multiple instances of Parameters can be added together
        ::

            >>> p_alpha = Parameters(['a', 'b', 'c'], name='alpha')
            >>> p_beta = Parameters(['c', 'd', 'e'], name='beta')
            >>> params = [
                {'name': 'a', 'value': 1.0, 'transform': 'log'},
                {'name': 'b', 'value': 2.0, 'bounds': (1.0, 3.0)},
            ]
            >>> p_gamma = Parameters(parameters=params, name='gamma')
            >>> print(p_alpha + p_beta + p_gamma)

            Parameters alpha__beta__gamma
            * alpha__beta
                * alpha
                    name=a value=0.000e+00 transform=none bounds=(None, None) prior=None
                    name=b value=0.000e+00 transform=none bounds=(None, None) prior=None
                    name=c value=0.000e+00 transform=none bounds=(None, None) prior=None
                * beta
                    name=c value=0.000e+00 transform=none bounds=(None, None) prior=None
                    name=d value=0.000e+00 transform=none bounds=(None, None) prior=None
                    name=e value=0.000e+00 transform=none bounds=(None, None) prior=None
            * gamma
                name=a value=1.000e+00 transform=log bounds=(None, None) prior=None
                name=b value=2.000e+00 transform=logit bounds=(1.0, 3.0) prior=None
    """

    def __init__(self, parameters: list, name: str = ''):

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
            parameters={left_name: self._parameters, right_name: other._parameters},
            name='__'.join((left_name, right_name)),
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
        """Returns the list of Parameter instance"""

        def _leafs(node):
            if isinstance(node, Parameter):
                return [node]
            return sum([_leafs(node[k]) for k in node], [])

        return _leafs(self._parameters)

    def set_parameter(self, *args, **kwargs):
        """Change settings of Parameters after instantiation

        ::

                p_alpha = Parameters(['a', 'b', 'c'], name='alpha')
                p_alpha.set_parameter('a', value=1, transform='log')
        """

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
    def theta(self) -> List:
        """Get the constrained parameter values :math:`\\mathbf{\\theta}`"""
        return [h.theta for h in self.parameters]

    @theta.setter
    def theta(self, x: Union[Tuple, List, np.ndarray]):
        """Set the constrained parameter values :math:`\\mathbf{\\theta}`

        Args:
            x: New constrained parameter values
        """

        if len(x) != len(self.parameters):
            raise ValueError(f'theta has {len(self.parameters)} parameters but {len(x)} are given')

        for p, value in zip(self.parameters, x):
            p.theta = value

    @property
    def theta_free(self) -> List:
        """Get constrained parameter values :math:`\\mathbf{\\theta}` which are not fixed"""
        return [p.theta for p in self.parameters_free]

    @theta_free.setter
    def theta_free(self, x: Union[Tuple, List, np.ndarray]):
        """Set constrained parameter values :math:`\\mathbf{\\theta}` which are not fixed

        Args:
            x: New free constrained parameter values
        """

        if len(x) != self.n_par:
            raise ValueError(f'theta has {self.n_par} free parameters but {len(x)} are given')

        for p, value in zip(self.parameters_free, x):
            p.theta = value

    @property
    def theta_sd(self) -> List:
        """Get standardized parameter values :math:`\\mathbf{\\theta_{sd}}` which are not fixed"""
        return [p.theta_sd for p in self.parameters_free]

    @theta_sd.setter
    def theta_sd(self, x: Union[Tuple, List, np.ndarray]):
        """Set standardized parameter values :math:`\\mathbf{\\theta_{sd}}` which are not fixed

        Args:
            x: New free standardized parameter values
        """

        if len(x) != self.n_par:
            raise ValueError(f'theta_sd has {self.n_par} free parameters but {len(x)} are given')

        for p, value in zip(self.parameters_free, x):
            p.theta_sd = value

    @property
    def theta_jacobian(self) -> List:
        """Inverse transform jacobian :math:`\\partial \\theta \\,/\\, \\partial \\eta`

        .. math::

            \\frac{\\partial \\pi(\\eta \\mid y)}{\\partial \\eta} =
            \\frac{\\partial \\pi(\\theta \\mid y)}{\\partial \\theta}
            \\frac{\\partial \\theta}{\\partial \\eta}


        where :math:`\\pi(\\theta \\mid y) = \\log p(\\theta \\mid y)` is the logarithm
        of the posterior distribution in the constrained parameter space and,
        :math:`\\theta = f^{-1}(\\eta)` is the inverse bijective transformation.

        Notes:
            Only univariate change of variables are supported.
        """
        return [p._inv_transform_jacobian() for p in self.parameters_free]

    @property
    def theta_log_jacobian(self) -> List:
        """Logarithm of the jacobian adjustment
        :math:`\\log \\left| \\partial \\theta \\,/\\, \\partial \\eta \\right|`

        .. math::

            \\pi(\\eta \\mid y) = \\pi(\\theta \\mid y) +
            \\log \\left| \\frac{\\partial \\theta}{\\partial \\eta} \\right|


        where :math:`\\pi(\\theta \\mid y) = \\log p(\\theta \\mid y)` is the logarithm
        of the posterior distribution in the constrained parameter space and,
        :math:`\\theta = f^{-1}(\\eta)` is the inverse bijective transformation.

        Notes:
            The jacobian adjustment is required if a prior distribution is used. This formula is
            works for univariate change of variables only. For multivariate case, the absolute
            value of the determinant of the jacobian matrix must be used.
        """
        return np.sum(np.log(np.abs(self.theta_jacobian)))

    @property
    def theta_dlog_jacobian(self) -> List:
        """Partial derivative of the logarithm of the jacobian adjustment
        :math:`\\partial \\log \\left| \\partial \\theta \\,/\\, \\partial \\eta \\right| \\,/\\, \\partial \\eta`


        .. math::

            \\frac{\\partial \\pi(\\eta \\mid y)}{\\partial \\eta} =
            \\frac{\\partial \\pi(\\theta \\mid y)}{\\partial \\theta}
            \\frac{\\partial \\theta}{\\partial \\eta} +
            \\frac{\\partial}{\\partial \\eta}
            \\log \\left|\\frac{\\partial \\theta}{\\partial \\eta}\\right|


        where :math:`\\pi(\\theta \\mid y) = \\log p(\\theta \\mid y)` is the logarithm
        of the posterior distribution in the constrained parameter space and,
        :math:`\\theta = f^{-1}(\\eta)` is the inverse bijective transformation.

        Notes:
            The jacobian adjustment is required if a prior distribution is used. This formula is
            works for univariate change of variables only. For multivariate case, the absolute
            value of the determinant of the jacobian matrix must be used.
        """
        return [p._inv_transform_dlog_jacobian() for p in self.parameters_free]

    @property
    def eta(self) -> np.ndarray:
        """Get the unconstrained parameter values :math:`\\mathbf{\\eta}`"""
        return np.array([p.eta for p in self.parameters])

    @eta.setter
    def eta(self, x: np.ndarray):
        """Set the unconstrained parameter values :math:`\\mathbf{\\eta}` which are not fixed

        Args:
            x: New free unconstrained parameter values
        """

        if len(x) != self.n_par:
            raise ValueError(f'eta has {self.n_par} free parameters but {len(x)} are given')

        for p, value in zip(self.parameters_free, x):
            p.eta = value

    @property
    def eta_free(self) -> np.ndarray:
        """Get the unconstrained parameter values :math:`\\mathbf{\\eta}` which are not fixed"""
        return self.eta[self.free]

    @property
    def eta_jacobian(self) -> List:
        """Transform jacobian :math:`\\partial \\eta \\,/\\, \\partial \\theta`

        .. math::

            \\frac{\\partial \\pi(\\theta \\mid y)}{\\partial \\theta} =
            \\frac{\\partial \\pi(\\eta \\mid y)}{\\partial \\eta}
            \\frac{\\partial \\eta}{\\partial \\theta}


        where :math:`\\pi(\\theta \\mid y) = \\log p(\\theta \\mid y)` is the logarithm
        of the posterior distribution in the constrained parameter space and,
        :math:`\\eta = f(\\theta)` is the bijective transformation.

        Notes:
            Only univariate change of variables are supported.
        """
        return [p._transform_jacobian() for p in self.parameters_free]

    @property
    def prior(self) -> float:
        """Get the logarithm of the prior distribution :math:`\\log p(\\theta)`"""
        return np.sum(
            [p.prior.log_pdf(p.value) for p in self.parameters_free if p.prior is not None]
        )

    @property
    def d_prior(self) -> List:
        """Get the partial derivative of logarithm of the prior distribution :math:`\\partial \\log p(\\theta) \\,/\\, \\partial \\theta`"""
        return [
            p.prior.dlog_pdf(p.value) if p.prior is not None else 0.0 for p in self.parameters_free
        ]

    @property
    def free(self) -> List[bool]:
        """Return the list of free parameters"""
        return [p.free for p in self.parameters]

    @property
    def names(self) -> List[str]:
        """Return the list of parameter names"""
        return [p.name for p in self.parameters]

    @property
    def names_free(self) -> List[str]:
        """Return the list of free parameter names"""
        return [p.name for p in self.parameters_free]

    @property
    def penalty(self, scaling: float = 1e-4) -> float:
        """penalty function

        Args:
            scaling: Scaling coefficient of the penalty function
        """
        return scaling * np.sum([p._penalty() for p in self.parameters_free])

    @property
    def d_penalty(self, scaling: float = 1e-4) -> List:
        """Partial derivative of the penalty function

        Args:
            scaling: Scaling coefficient of the penalty function
        """
        return [scaling * p._d_penalty() for p in self.parameters_free]

    def prior_init(self, hpd=None):
        """Draw a random sample from the prior distribution, :math:`\\theta \sim p(\\theta)`

        Args:
            hpd: Highest Prior Density to draw sample from (True for unimodal distribution)
        """

        for p in self.parameters:
            if p.prior is not None:
                p.theta_sd = p.prior.random(hpd=hpd)[0]

    @property
    def scale(self) -> List:
        """Scales of the constrained parameters :math:`\\theta`"""
        return [p.scale for p in self.parameters_free]

    @property
    def n_par(self) -> int:
        """Number of free parameters"""
        return np.sum(self.free)

    @property
    def parameters_free(self) -> List:
        """Returns the Parameter instances which are not fixed"""
        return [p for p in self.parameters if p.free]
