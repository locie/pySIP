import numpy as np
from ..base import StateSpace
from ..gaussian_process.matern import Matern12, Matern32, Matern52
from ..gaussian_process.periodic import Periodic


class QuasiPeriodic(StateSpace):
    """Quasi-periodic covariance function

    Kernel parameters
    -----------------
    period : repetition period
    mscale : magnitude scale
    lscale_p : characteristic length scale of the periodic covariance
    lscale_m : characteristic length scale of the Matérn covariance
    J : degree of approximation
    nu : Matérn smoothness

    Notes
    -----
    The stochastic differential equation representation works only for
    univariate temporal time series, e.g. the input is the time.

    The Matérn smoothness parameter `nu` are restricted to the following
    half integer values, 1/2, 3/2, 5/2, in order to have an exact
    stochastic differential equation representation.

    """

    def __init__(self, J=6, nu="1/2"):
        """QuasiPeriodic is a StateSpace instance"""

        if not isinstance(J, int):
            raise TypeError("The degree of approximation `J` must an integer")

        if J <= 0:
            raise ValueError(f"The degree of approximation `J` must be "
                             "strictly positive")

        available_nu = ["1/2", "3/2", "5/2"]
        if nu not in available_nu:
            raise ValueError(f"The Matérn smoothness parameter `nu` must "
                             f"be either {available_nu}")

        self._names = ["period", "mscale", "lscale", "sigv", "lscale_m"]

        # periodic covariance function
        self._periodic = Periodic(J=J)

        # Matérn covariance function
        if nu == "1/2":
            self._matern = Matern12()
        elif nu == "3/2":
            self._matern = Matern32()
        else:
            self._matern = Matern52()

        self.Nx = self._periodic.Nx * self._matern.Nx
        self.Nu = 0
        self.Ny = 1

        self._Ip = np.eye(self._periodic.Nx)
        self._Im = np.eye(self._matern.Nx)

        super().__post_init__()

    def update(self):
        """Evaluate the state-space model with the parameter values

        Notes
        -----
        The unpacking of the parameters must be in the same order
        as in `names`.

        """
        self._periodic.parameters.theta = (
            self.parameters.theta_by_name(self._periodic._names))
        # Matérn covariance: mscale=1, lscale = lscale_m, sigv = 0
        self._matern.parameters.theta = \
            [1, self.parameters.theta_by_name(["lscale_m"])[0], 0]

        self._periodic.update()
        self._matern.update()

        # state matrix
        self.A = np.kron(self._periodic.A, self._Im) + \
            np.kron(self._Ip, self._matern.A)

        # output matrix
        self.C = np.kron(self._periodic.C, self._matern.C)

        # scaling matrix of the Wiener process
        self.Q = np.kron(self._periodic.P0, self._matern.Q)

        # standard deviation of the measurement noise
        self.R = self._periodic.R

        # initial standard deviation of the state vector
        self.P0 = np.kron(self._periodic.P0, self._matern.P0)

        if self.jacobian:
            '''missing dR'''

            self.dA["period"] = np.kron(self._periodic.dA["period"], self._Im)
            self.dA["lscale_m"] = np.kron(self._Ip, self._matern.dA["lscale"])

            self.dQ["mscale"] = np.kron(self._periodic.dP0["mscale"],
                                        self._matern.Q)
            self.dQ["lscale"] = np.kron(self._periodic.dP0["lscale"],
                                        self._matern.Q)
            self.dQ["lscale_m"] = np.kron(self._periodic.P0,
                                          self._matern.dQ["lscale"])

            self.dP0["mscale"] = np.kron(self._periodic.dP0["mscale"],
                                         self._matern.P0)
            self.dP0["lscale"] = np.kron(self._periodic.dP0["lscale"],
                                         self._matern.P0)
            self.dP0["lscale_m"] = np.kron(self._periodic.P0,
                                           self._matern.dP0["lscale"])
