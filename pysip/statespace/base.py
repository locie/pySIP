from dataclasses import field
from itertools import chain
from typing import NamedTuple, Optional, Sequence, Tuple, Union
from typing_extensions import Self

import numpy as np
import pandas as pd
from pydantic import ConfigDict, conint
from pydantic.dataclasses import dataclass

from ..params import Parameters
from ..utils.math import nearest_cholesky
from . import discretization
from .meta import MetaStateSpace
from .nodes import Node


def _check_data(dt, u, dtu, y):
    for df in [dt, u, dtu]:
        if not np.all(np.isfinite(df)):
            raise ValueError(f"{df} contains undefinite values")
    if not np.all(np.isnan(y[~np.isfinite(y)])):
        raise TypeError("The output vector must contains numerical values or numpy.nan")


def prepare_data(
    df: pd.DataFrame,
    inputs: Union[str, list],
    outputs: Union[str, list],
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

    y = pd.DataFrame(df[outputs])

    _check_data(dt, u, dtu, y)
    return dt, u, dtu, y


class States(NamedTuple):
    C: np.ndarray
    D: np.ndarray
    R: np.ndarray
    A: np.ndarray
    B0: np.ndarray
    B1: np.ndarray
    Q: np.ndarray


class DiscreteStates(NamedTuple):
    A: np.ndarray
    B0: np.ndarray
    B1: np.ndarray
    Q: np.ndarray


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
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
        self.set_constant_continuous_ssm()
        self._diag = np.diag_indices_from(self.A)

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
            - **A**: Discrete state matrix
            - **B0**: Discrete input matrix (zero order hold)
            - **B1**: Discrete input matrix (first order hold)
            - **Q**: Upper Cholesky factor of the process noise covariance matrix
        """
        self.update_continuous_ssm()
        return self.discretization(dt)

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
        DiscreteStates
            Discrete state-space model, a 4-elements namedtuple containing
            - **A**: Discrete state matrix
            - **B0**: Discrete input matrix (zero order hold)
            - **B1**: Discrete input matrix (first order hold)
            - **Q**: Upper Cholesky factor of the process noise covariance matrix
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
        outputs: Optional[Sequence[str]] = None,
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
        outputs : Optional[Sequence[str]], optional
            List of output variables, by default None. If None, the outputs defined in
            the model are used.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Tuple containing the following dataframes:
            - **dt**: Time step
            - **u**: Input data
            - **dtu**: Derivative of input data
            - **y**: Output data
        """
        if inputs is None:
            inputs = [par.name for par in self.inputs]
        if outputs is None:
            outputs = [par.name for par in self.outputs]
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(outputs, str):
            outputs = [outputs]
        for key in chain(inputs, outputs):
            if key not in df.columns:
                raise KeyError(
                    f"Missing column {key}. Use `inputs` and `outputs` parameters to "
                    "specify the inputs and outputs if they diverge from the model "
                    "definition."
                )
        dt, u, dtu, y = prepare_data(df, inputs, outputs, time_scale=time_scale)
        return dt, u, dtu, y

@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class RCModel(StateSpace):
    """Dynamic thermal model"""

    latent_forces: str = ""

    def __post_init__(self):
        print(self.name)
        super().__post_init__()
        eig = np.real(np.linalg.eigvals(self.A))
        if np.all(eig < 0) and eig.max() / eig.min():
            self._method = "analytic"
        else:
            self._method = "mfd"

    def __repr__(self):
        return f"\n{self.__class__.__name__}" + "-" * len(self.__class__.__name__)

    def __le__(self, gp):
        """Create a Latent Force Model"""
        from .latent_force_model import LatentForceModel

        return LatentForceModel(self, gp, self.latent_forces)

    def discretization(self, dt: float) -> DiscreteStates:
        """Discretization of RC model

        Parameters
        ----------
        dt : float
            Sampling time

        Returns
        -------
        DiscreteStates
            Discrete state-space model, a 4-elements namedtuple containing
            - **A**: Discrete state matrix
            - **B0**: Discrete input matrix (zero order hold)
            - **B1**: Discrete input matrix (first order hold)
            - **Q**: Upper Cholesky factor of the process noise covariance matrix
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

        return DiscreteStates(Ad, B0d, B1d, Qd)


class GPModel(StateSpace):
    """Gaussian Process"""

    def __post_init__(self):
        if hasattr(self, "J"):
            self.states = self.states_block * int(self.J + 1)
        super().__post_init__()

    def __repr__(self):
        return f"\n{self.__class__.__name__}" + "-" * len(self.__class__.__name__)

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
        DiscreteStates
            Discrete state-space model, a 4-elements namedtuple containing
            - **A**: Discrete state matrix
            - **B0**: Discrete input matrix (zero order hold)
            - **B1**: Discrete input matrix (first order hold)
            - **Q**: Upper Cholesky factor of the process noise covariance matrix
        """
        Ad = discretization.state(self.A, dt)
        B0d = np.zeros((self.nx, self.nu))
        Qd = nearest_cholesky(
            discretization.diffusion_stationary(self.P0.T @ self.P0, Ad)
        )

        return Ad, B0d, B0d, Qd
