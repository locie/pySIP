from collections import defaultdict
from dataclasses import field
from enum import Enum
from itertools import chain
from typing import Literal, NamedTuple, Optional, Sequence, Tuple, Union
from typing_extensions import Self

import numpy as np
import pandas as pd
from pydantic import ConfigDict, conint
from pydantic.dataclasses import dataclass
from jinja2 import Template

from ..params import Parameters
from ..utils.math import nearest_cholesky
from ..utils.misc import Namespace
from . import discretization


model_registry = Namespace()


class Io(Enum):
    """Input/Output data type"""

    TEMPERATURE = ("Temperature", "°C")
    POWER = ("Power", "W")
    ANY = ("Any type", "any")


class Par(Enum):
    """Parameter type"""

    THERMAL_RESISTANCE = ("Thermal resistance", "°C/W")
    THERMAL_TRANSMITTANCE = ("Thermal transmittance", "W/°C")
    THERMAL_CAPACITY = ("Thermal capacity", "J/°C")
    SOLAR_APERTURE = ("Solar aperture", "m²")
    STATE_DEVIATION = ("State deviation", "any")
    MEASURE_DEVIATION = ("Measure deviation", "any")
    INITIAL_DEVIATION = ("Initial deviation", "any")
    INITIAL_MEAN = ("Initial mean", "any")
    COEFFICIENT = ("Coefficient", "any")
    MAGNITUDE_SCALE = ("Magnitude scale", "any")
    LENGTH_SCALE = ("Length scale", "any")
    PERIOD = ("Period", "any")


@dataclass
class Node:
    """Description of model variables

    Parameters
    ----------
    category : Io or Par
        Category of the node
    name : str
        Name of the node
    description : str
        Description of the node
    """

    category: Union[Io, Par] = field()
    name: str = field(default="")
    description: str = field(default="")

    def __post_init__(self):
        if isinstance(self.category, str):
            try:
                self.category = Io[self.category]
            except KeyError:
                self.category = Par[self.category]

    def unpack(self):
        """Unpack the Node"""
        return self.category.name, self.name, self.description


def statespace(cls, *args, **kwargs):
    if cls not in model_registry:
        return None
    if args or kwargs:
        return model_registry[cls](*args, **kwargs)
    return model_registry[cls]


class MetaStateSpace(type):
    """Metaclass for the state-space models

    Mainly used to build the documentation of the models from the sections
    (inputs, outputs, states, params) available as class attributes. Will also add
    the class to the registry.
    """

    def __new__(cls, name, bases, attrs):
        # Add the class to the registry
        new_class = super(MetaStateSpace, cls).__new__(cls, name, bases, attrs)
        if "__base__" not in attrs:
            model_registry[attrs.get("__name__") or attrs["__qualname__"]] = new_class
        return new_class

    def _format_section(self, section_name):
        if not (nodes := getattr(self, section_name, False)):
            return None
        nodes = [Node(*x) for x in nodes]
        return self._section_template.render(section_name=section_name, nodes=nodes)

    @property
    def _sections(self) -> dict:
        return {
            section: [Node(*x) for x in getattr(self, section, [])]
            for section in ["inputs", "outputs", "states"]
        }

    @property
    def _parameters_cat(self) -> dict:
        parameters_cat = defaultdict(list)
        for par in getattr(self, "params", []):
            par = Node(*par)
            parameters_cat[par.category].append(par)
        return dict(parameters_cat)

    def _build_doc(self):
        _doc_template_str = """{{base_doc}}

Model Variables
---------------

{% for section, nodes in sections.items() %}
- {{section.capitalize()}}
{% for node in nodes %}
    - ``{{node.name}}``: {{node.description}} `({{node.category.value[1]}})`
{% endfor %}
{% endfor %}

{% if parameters_cat %}

Model Parameters
----------------

{% for category, parameters in parameters_cat.items() %}
- {{category.value[0]}}
{% for parameter in parameters %}
    - ``{{parameter.name}}``: {{parameter.description}}
      `({{parameter.category.value[1]}})`
{% endfor %}
{% endfor %}
{% endif %}
    """
        _doc_template = Template(
            _doc_template_str,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=False,
        )

        return _doc_template.render(
            base_doc=self.__doc__,
            sections=self._sections,
            parameters_cat=self._parameters_cat,
        )

    def __init__(self, name, bases, attr):
        self.__doc__ = self._build_doc()


def prepare_data(
    df: pd.DataFrame,
    inputs: Union[str, list],
    outputs: Union[str, list, bool],
    time_scale: str = "s",
):
    """Prepare data for the state-space model

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the data
    inputs : str or list
        Inputs names
    outputs : str or list
        Outputs names
    time_scale : str, optional
        Time scale, by default "s"

    Returns
    -------
    dt : pandas.Series
        Time steps
    u : pandas.DataFrame
        Inputs
    dtu : pandas.DataFrame
        Inputs time derivative
    y : pandas.DataFrame
        Outputs
    """
    df = df.copy()
    time = df.index.to_series()
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a dataframe")
    time_scale = pd.to_timedelta(1, time_scale)

    # diff and forward-fill the nan-last value
    dt = time.diff().shift(-1)
    dt.iloc[-1] = dt.iloc[-2]
    # deal with numerical approximation that lead to almost equal values
    if np.isclose(dt.to_numpy(), dt.iloc[0]).all():
        dt[:] = dt.iloc[0]
    if isinstance(df.index, pd.DatetimeIndex):
        dt = dt / time_scale
    else:
        dt = dt.astype(float)

    u = pd.DataFrame(df[inputs])

    dtu = u.diff().shift(-1) / dt.to_numpy()[:, None]
    dtu.iloc[-1, :] = 0

    for subdf in [dt, u, dtu]:
        if not np.all(np.isfinite(subdf)):
            raise ValueError(f"{subdf} contains undefinite values")

    if outputs:
        y = pd.DataFrame(df[outputs])
    else:
        y = pd.DataFrame(data=np.full(len(df), np.nan), index=df.index)
    if not np.all(np.isnan(y[~np.isfinite(y)])):
        raise TypeError("The output vector must contains numerical values or numpy.nan")
    return dt, u, dtu, y


class ContinuousStates(NamedTuple):
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    Q: np.ndarray
    R: np.ndarray


class DiscreteStates(NamedTuple):
    A: np.ndarray
    B0: np.ndarray
    B1: np.ndarray
    Q: np.ndarray


@dataclass(config=ConfigDict(arbitrary_types_allowed=True), repr=False)
class StateSpace(metaclass=MetaStateSpace):
    """Linear Gaussian Continuous-Time State-Space Model"""

    parameters: Parameters = field(default=None, repr=False)
    hold_order: conint(ge=0, le=1) = 0
    method: str = "mfd"
    name: str = ""

    def _coerce_attributes(self):
        for attr in ["states", "params", "inputs", "outputs"]:
            setattr(self, attr, [Node(*s) for s in getattr(self, attr)])
        self.states: Sequence[Node]
        self.params: Sequence[Node]
        self.inputs: Sequence[Node]
        self.outputs: Sequence[Node]

        if self.parameters:
            if not isinstance(self.parameters, Parameters):
                self.parameters = Parameters(self.parameters)
        else:
            self.parameters = Parameters([p.name for p in self.params])
        self.parameters.name = self.name

    def _init_states(self):
        self.A = np.zeros((self.nx, self.nx))
        self.B = np.zeros((self.nx, self.nu))
        self.C = np.zeros((self.ny, self.nx))
        self.D = np.zeros((self.ny, self.nu))
        self.Q = np.zeros((self.nx, self.nx))
        self.R = np.zeros((self.ny, self.ny))
        self.x0 = np.zeros((self.nx, 1))
        self.P0 = np.zeros((self.nx, self.nx))
        self._diag = np.diag_indices_from(self.A)
        self.set_constant_continuous_ssm()

    def __repr__(self):
        tpl = Template(
            """{{name}}
{{'-' * name|length}}
States:
{% for state in states %}
- {{state.name}}: {{state.description}}
{% endfor %}

Inputs:
{% for input in inputs %}
- {{input.name}}: {{input.description}}
{% endfor %}

Outputs:
{% for output in outputs %}
- {{output.name}}: {{output.description}}
{% endfor %}

Parameters:
{% for parameter in parameters %}
- {{parameter.name}}: {{parameter.value}}
{% endfor %}""",
            trim_blocks=True,
            lstrip_blocks=True,
        )
        return tpl.render(
            name=self.name,
            states=self.states,
            inputs=self.inputs,
            outputs=self.outputs,
            parameters=self.parameters,
        )

    def __post_init__(self):
        if self.name == "":
            self.name = self.__class__.__name__
        self._coerce_attributes()
        self._init_states()

    @property
    def nx(self):
        return len(self.states)

    @property
    def ny(self):
        return len(self.outputs)

    @property
    def nu(self):
        return len(self.inputs)

    def set_constant_continuous_ssm(self):
        """Set constant values in state-space model"""
        pass

    def update_continuous_ssm(self):
        """Update the state-space model with the constrained parameters"""
        pass

    def update(self):
        """Update the state-space model with the constrained parameters"""
        self.update_continuous_ssm()

    def get_discrete_ssm(self, dt: float) -> DiscreteStates:
        """Return the updated discrete state-space model

        Parameters
        ----------
        dt : float
            Sampling time

        Returns
        -------
        DiscreteStates
            Discrete state-space model, a 4-elements namedtuple containing
            - A: Discrete state matrix
            - B0: Discrete input matrix (zero order hold)
            - B1: Discrete input matrix (first order hold)
            - Q: Upper Cholesky factor of the process noise covariance matrix
        """
        self.update_continuous_ssm()
        return DiscreteStates(*self.discretization(dt))

    def discretization(
        self, dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Discretization of LTI state-space model. Should be overloaded by subclasses.

        Parameters
        ----------
        dt : float
            Sampling time

        Returns
        -------
        Tuple[np.ndarray, ...]
            a 4-elements tuple containing
            - A: Discrete state matrix
            - B0: Discrete input matrix (zero order hold)
            - B1: Discrete input matrix (first order hold)
            - Q: Upper Cholesky factor of the process noise covariance matrix
        """

        if self.nu == 0:
            Ad = discretization.state(self.A, dt)
            B0d = np.zeros((self.nx, self.nu))
            B1d = B0d
        else:
            Ad, B0d, B1d = discretization.state_input(
                self.A, self.B, dt, self.hold_order, "expm"
            )

        Qd = nearest_cholesky(
            discretization.diffusion_mfd(self.A, self.Q.T @ self.Q, dt)
        )

        return Ad, B0d, B1d, Qd

    def prepare_data(
        self,
        df: pd.DataFrame,
        inputs: Optional[Sequence[str]] = None,
        outputs: Optional[Union[Sequence[str], Literal[False]]] = None,
        time_scale: str = "s",
    ):
        """Prepare data for training

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the data
        inputs : Optional[Sequence[str]], optional
            List of input variables, by default None. If None, the inputs defined in
            the model are used.
        outputs : Optional[Sequence[str] or False], optional
            List of output variables, by default None. If None, the outputs defined in
            the model are used. If False, no output is returned.

        Returns
        -------
        DataFrame:
            time steps
        DataFrame:
            input data
        DataFrame:
            derivative of input data
        DataFrame:
            output data (only if outputs is not False)
        """
        if inputs is None:
            inputs = [par.name for par in self.inputs]
        if outputs is None:
            outputs = [par.name for par in self.outputs]
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(outputs, str):
            outputs = [outputs]
        if outputs is False:
            outputs = []
        for key in chain(inputs, outputs):
            if key not in df.columns:
                raise KeyError(
                    f"Missing column {key}. Use `inputs` and `outputs` parameters to "
                    "specify the inputs and outputs if they diverge from the model "
                    "definition."
                )
        return prepare_data(df, inputs, outputs, time_scale=time_scale)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True), repr=False)
class RCModel(StateSpace):
    """Dynamic thermal model"""

    latent_forces: str = ""

    def __post_init__(self):
        super().__post_init__()
        eig = np.real(np.linalg.eigvals(self.A))
        if np.all(eig < 0) and eig.max() / eig.min():
            self._method = "analytic"
        else:
            self._method = "mfd"

    def __le__(self, gp):
        """Create a Latent Force Model"""
        from .latent_force_model import LatentForceModel

        return LatentForceModel(self, gp, self.latent_forces)

    def discretization(
        self, dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Discretization of RC model

        Parameters
        ----------
        dt : float
            Sampling time

        Returns
        -------
        Tuple[np.ndarray, ...]
            a 4-elements tuple containing
            - A: Discrete state matrix
            - B0: Discrete input matrix (zero order hold)
            - B1: Discrete input matrix (first order hold)
            - Q: Upper Cholesky factor of the process noise covariance matrix
        """

        if self.method == "analytic":
            Ad, B0d, B1d = discretization.state_input(
                self.A, self.B, dt, self.hold_order, "analytic"
            )
            Qd = nearest_cholesky(
                discretization.diffusion_lyap(self.A, self.Q.T @ self.Q, Ad)
            )
        else:
            Ad, B0d, B1d = discretization.state_input(
                self.A, self.B, dt, self.hold_order, "expm"
            )
            Qd = nearest_cholesky(
                discretization.diffusion_mfd(self.A, self.Q.T @ self.Q, dt)
            )

        return Ad, B0d, B1d, Qd


class GPModel(StateSpace):
    """Gaussian Process"""

    def __post_init__(self):
        if hasattr(self, "J"):
            self.states = self.states_block * int(self.J + 1)
        super().__post_init__()

    def __mul__(self, gp: Self):
        if not isinstance(gp, GPModel):
            raise TypeError("`gp` must be an GPModel instance")
        # TODO: refactor to avoid circular import
        from .gaussian_process import GPProduct

        return GPProduct(self, gp)

    def __add__(self, gp: Self):
        if not isinstance(gp, GPModel):
            raise TypeError("`gp` must be an GPModel instance")
        # TODO: refactor to avoid circular import
        from .gaussian_process import GPSum

        return GPSum(self, gp)

    def discretization(
        self, dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Discretization of the temporal Gaussian Process

        Parameters
        ----------
        dt : float
            Sampling time

        Returns
        -------
        Tuple[np.ndarray, ...]
            a 4-elements tuple containing
            - A: Discrete state matrix
            - B0: Discrete input matrix (zero order hold)
            - B1: Discrete input matrix (first order hold)
            - Q: Upper Cholesky factor of the process noise covariance matrix
        """
        Ad = discretization.state(self.A, dt)
        B0d = np.zeros((self.nx, self.nu))
        Qd = nearest_cholesky(
            discretization.diffusion_stationary(self.P0.T @ self.P0, Ad)
        )

        return Ad, B0d, B0d, Qd
