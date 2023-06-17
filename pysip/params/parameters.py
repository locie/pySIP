from typing import List, Literal, Sequence, Union
from typing_extensions import Self
from flatten_dict import flatten
from flatten_dict.reducers import make_reducer

import numpy as np

from .parameter import Parameter


def _coerce_params(parameters: Union[List[str], List[dict]]) -> List[dict]:
    if isinstance(parameters, dict):
        return parameters
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
                s += "Parameters " + self.name + "\n"
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
        left_name = self.name if self.name else "left"
        right_name = other.name if other.name else "right"
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
    def theta(self) -> np.ndarray:
        """Get the constrained parameter values θ"""
        return np.array([h.theta for h in self.parameters])

    @theta.setter
    def theta(self, x: Sequence[float]):
        if len(x) != len(self):
            raise ValueError(f"{len(x)} values are given but {len(self)} are expected")

        for p, value in zip(self.parameters, x):
            p.theta = value

    @property
    def theta_free(self) -> np.ndarray:
        return np.array([p.theta for p in self.parameters_free])

    @theta_free.setter
    def theta_free(self, x: Sequence[float]):
        if len(x) != self.n_par:
            raise ValueError(f"{len(x)} values are given but {len(self)} are expected")

        for p, value in zip(self.parameters_free, x):
            p.theta = value

    @property
    def theta_sd(self) -> np.ndarray:
        return np.array([p.theta_sd for p in self.parameters_free])

    @theta_sd.setter
    def theta_sd(self, x: Sequence[float]):
        if len(x) != self.n_par:
            raise ValueError(f"{len(x)} values are given but {len(self)} are expected")

        for p, value in zip(self.parameters_free, x):
            p.theta_sd = value

    @property
    def theta_jacobian(self) -> np.ndarray:
        return np.array([p.get_inv_transform_jacobian() for p in self.parameters_free])

    @property
    def theta_log_jacobian(self) -> np.ScalarType:
        return np.sum(np.log(np.abs(self.theta_jacobian)))

    @property
    def eta(self) -> np.ndarray:
        return np.array([p.eta for p in self.parameters])

    @eta.setter
    def eta(self, x: Sequence[float]):
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
    def d_penalty(self) -> List:
        """Partial derivative of the penalty function

        Args:
            scaling: Scaling coefficient of the penalty function
        """
        SCALING = 1e-4
        return [SCALING * p.get_grad_penalty() for p in self.parameters_free]

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
    def ids(self) -> List[str]:
        return list(flatten(self._parameters, reducer=make_reducer(delimiter="__")))

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
        return np.array([p.scale for p in self.parameters_free])

    @property
    def n_par(self) -> int:
        return np.sum(self.free)

    @property
    def parameters_free(self) -> List[Parameter]:
        return [p for p in self.parameters if p.free]

    def init_parameters(
        self,
        n_init: int = 1,
        method: Literal[
            "unconstrained", "prior", "zero", "fixed", "value"
        ] = "unconstrained",
        hpd: float = 0.95,
    ) -> np.ndarray:
        """Random initialization of the parameters

        Parameters
        ----------
        n_init: int, default 1
            Number of random initialization
        method: str, default 'unconstrained'
            - **unconstrained**: Uniform draw between [-1, 1] in the uncsontrained
            space
            - **prior**: Uniform draw from the prior distribution
            - **zero**: Set the unconstrained parameters to 0
            - **fixed**: The current parameter values are used
            - **value**: Uniform draw between the parameter value +/- 25%
        hpd: bool, default False
            Highest Prior Density to draw sample from (True for unimodal
            distribution)

        Returns
        -------
        eta0: ndarray of shape (n_par, n_init)
            Array of unconstrained parameters, where n_par is the
            number of free parameters and n_init the number of random initialization
        """

        if not isinstance(n_init, int) or n_init <= 0:
            raise TypeError("`n_init` must an integer greater or equal to 1")

        available_methods = ["unconstrained", "prior", "zero", "fixed", "value"]
        if method not in available_methods:
            raise ValueError(
                f"`method` must be one of the following {available_methods}"
            )

        if not (0.0 < hpd <= 1.0):
            raise ValueError("`hpd` must be between ]0, 1]")

        n_par = len(self.eta_free)
        if method == "unconstrained":
            eta0 = np.random.uniform(-1, 1, (n_par, n_init))
        elif method == "zero":
            eta0 = np.zeros((n_par, n_init))
        else:
            eta0 = np.zeros((n_par, n_init))
            for n in range(n_init):
                if method == "prior":
                    self.prior_init(hpd=hpd)
                elif method == "value":
                    value = np.asarray(self.theta_sd)
                    lb = value - 0.25 * value
                    ub = value + 0.25 * value
                    self.theta_sd = np.random.uniform(lb, ub)

                eta0[:, n] = self.eta_free

        return np.squeeze(eta0)
