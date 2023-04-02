from typing import List, Sequence, Union
from typing_extensions import Self

import numpy as np

from .parameter import Parameter

def _coerce_params(parameters: Union[List[str], List[dict]]) -> List[dict]:
    if isinstance(parameters, dict):
        return  parameters
    if isinstance(parameters[0], str):
        return {name: Parameter(name) for name in parameters}
    return {k["name"]: Parameter(**k) for k in parameters}

class Parameters:
    """Factory of Parameter instances

    Parameters
    ----------
    parameters : list
        There is two options for instantiating Parameters
            - `parameters` is a list of strings corresponding to the parameters names.
              In this case all the parameters have the default settings
            - `parameters` is a list of dictionaries, where the arguments of Parameter
              can be modified as key-value pairs
    name : str, optional
        Name of this specific instance, by default ""

    Attributes
    ----------
    free: List[bool]
        False if the parameter is fixed
    names: List[str]
        List of the parameters names
    names_free: List[str]
        List of the parameters names, only for free parameters
    scales: List[float]
        List of the parameters scales
    n_par: int
        Number of free parameters
    parameters_free: List[Parameter]
        List of the free parameters
    theta: List[float]
        Parameter value in the constrained space θ
    theta_free: List[float]
        Parameter value in the constrained space θ, only for free parameters
    theta_sd: List[float]
        Parameter value in the standardized constrained space θ_sd, only for free
        parameters
    theta_jacobian: List[float]
        Gradient of the transformation from θ to η, only for free parameters
    theta_log_jacobian: float
        Sum of the log of the absolute values of the gradent of the transformation from
        θ to η for each parameter
    eta: List[float]
        Parameter value in the unconstrained space η
    eta_free: List[float]
        Parameter values in the unconstrained space η, only for free parameters
    eta_jacobian: List[float]
        Jacobian of the transformation from θ to η, only for free parameters
    penalty: float
        Sum of the penalty term for the parameters
    prior: float
        Sum of the logarithm of the prior distribution, only for free parameters with a
        defined prior.
    """

    def __init__(self, parameters: Union[List[str], List[dict]], name: str = ""):
        self.name = name
        self._parameters = _coerce_params(parameters)


    def __repr__(self):
        # TODO: make that easier to maintain
        def _repr(d, level=0):
            s = ""
            if level == 0:
                s += "Parameters " + self._name + "\n"
            for k, v in d.items():
                s += "    " * level
                if isinstance(v, Parameter):
                    s += v.__repr__() + "\n"
                else:
                    s += f"* {k}\n"
                    s += _repr(v, level + 1)
            return s

        return _repr(self._parameters)

    def __iter__(self):
        return iter(self._parameters.values())

    def __eq__(self, other: Self) -> bool:
        return self._parameters == other._parameters

    def __add__(self, other: Self) -> Self:
        left_name = self._name if self._name else "left"
        right_name = other._name if other._name else "right"
        return Parameters(
            parameters={left_name: self._parameters, right_name: other._parameters},
            name="__".join((left_name, right_name)),
        )

    @property
    def parameters(self):
        """Returns the list of Parameter instance"""
        # TODO: use a flatten dict to hold the parameters

        def _leafs(node):
            if isinstance(node, Parameter):
                return [node]
            return sum([_leafs(node[k]) for k in node], [])

        return _leafs(self._parameters)

    def __len__(self) -> int:
        return len(self.parameters)

    def set_parameter(self, *args, **kwargs):
        """Change settings of Parameters after instantiation

        p_alpha = Parameters(['a', 'b', 'c'], name='alpha')
        p_alpha.set_parameter('a', value=1, transform='log')
        """

        # TODO: properly deal with such nested dicts, maybe using Box
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
    def theta(self) -> List[float]:
        """Get the constrained parameter values θ"""
        return [h.theta for h in self.parameters]

    @theta.setter
    def theta(self, x: Sequence[float]):
        if len(x) != len(self):
            raise ValueError(f"{len(x)} values are given but {len(self)} are expected")

        for p, value in zip(self.parameters, x):
            p.theta = value

    @property
    def theta_free(self) -> List[float]:
        return [p.theta for p in self.parameters_free]

    @theta_free.setter
    def theta_free(self, x: Sequence[float]):
        if len(x) != self.n_par:
            raise ValueError(f"{len(x)} values are given but {len(self)} are expected")

        for p, value in zip(self.parameters_free, x):
            p.theta = value

    @property
    def theta_sd(self) -> List[float]:
        return [p.theta_sd for p in self.parameters_free]

    @theta_sd.setter
    def theta_sd(self, x: Sequence[float]):
        if len(x) != self.n_par:
            raise ValueError(f"{len(x)} values are given but {len(self)} are expected")

        for p, value in zip(self.parameters_free, x):
            p.theta_sd = value

    @property
    def theta_jacobian(self) -> List:
        return [p.get_inv_transform_jacobian() for p in self.parameters_free]

    @property
    def theta_log_jacobian(self) -> List:
        return np.sum(np.log(np.abs(self.theta_jacobian)))

    @property
    def eta(self) -> np.ndarray:
        return np.array([p.eta for p in self.parameters])

    @eta.setter
    def eta(self, x: np.ndarray):
        if len(x) != self.n_par:
            raise ValueError(f"{len(x)} values are given but {len(self)} are expected")

        for p, value in zip(self.parameters_free, x):
            p.eta = value

    @property
    def eta_free(self) -> np.ndarray:
        return self.eta[self.free]

    @property
    def eta_jacobian(self) -> List:
        return [p.get_transform_jacobian() for p in self.parameters_free]

    @property
    def penalty(self) -> float:
        SCALING = 1e-4
        return SCALING * np.sum([p.get_penalty() for p in self.parameters_free])

    @property
    def prior(self) -> float:
        return np.sum(
            [
                p.prior.log_pdf(p.value)
                for p in self.parameters_free
                if p.prior is not None
            ]
        )

    @property
    def free(self) -> List[bool]:
        return [p.free for p in self.parameters]

    @property
    def names(self) -> List[str]:
        return [p.name for p in self.parameters]

    @property
    def names_free(self) -> List[str]:
        return [p.name for p in self.parameters_free]

    def prior_init(self, hpd=None):
        """Draw a random sample from the prior distribution for each parameter
        that have a defined prior.

        Arguments
        ---------
        hpd: float
            Highest posterior density interval.
        """
        for p in self.parameters:
            if p.prior is not None:
                p.theta_sd = p.prior.random(hpd=hpd)[0]

    @property
    def scale(self) -> List[float]:
        return [p.scale for p in self.parameters_free]

    @property
    def n_par(self) -> int:
        return np.sum(self.free)

    @property
    def parameters_free(self) -> List[Parameter]:
        return [p for p in self.parameters if p.free]
