from typing import Callable, Tuple

import numpy as np

from .metrics import EuclideanMetric


class EuclideanHamiltonian:
    """Hamiltonian System

    .. math::
       :nowrap:

        \\begin{align*}
            H(q, p) &= -\\log \\pi(p \\mid q) -\\log \\pi(q) \\\\
                    &= K(p, q) + V(q)
        \\end{align*}

    where :math:`H(q, p)` is the Hamiltonian function with :math:`q` and :math:`p`,
    the position and momentum variables in phase space. :math:`K(p, q)` is the Kinetic energy
    and :math:`V(q)` is the potential energy. The potential energy is completely determined by the
    target distribution while the kinetic energy is specified by the implementation.

    Hamiltonian's equations

    .. math::
       :nowrap:

        \\begin{align*}
            \\frac{\\mathrm{d}q}{\\mathrm{d}t} &= \\frac{\\partial H}{\\partial p} =
                \\frac{\\partial K}{\\partial p} \\\\
            \\frac{\\mathrm{d}p}{\\mathrm{d}t} &= -\\frac{\\partial H}{\\partial q} =
                -\\frac{\\partial K}{\\partial q} - \\frac{\\partial V}{\\partial q}
        \\end{align*}


    For Euclidean-Gaussian kinetic energy

    .. math::
       :nowrap:

        \\begin{align*}
            K(p) &= \\frac{1}{2} p M^{-1} p + \\log \\lvert M \\rvert + \\text{const.} \\\\
            &\\propto \\frac{1}{2} p M^{-1} p
        \\end{align*}


    with :math:`M` the mass matrix, the Hamiltonian system is separable such that

    .. math::
       :nowrap:

        \\begin{align*}
            H(q, p) &= -\\log \\pi(p) -\\log \\pi(q) \\\\
                    &= K(p) + V(q)
        \\end{align*}


    which simplifies the Hamiltonian's equations to

    .. math::
       :nowrap:

        \\begin{align*}
            \\frac{\\mathrm{d}q}{\\mathrm{d}t} &= \\frac{\\partial H}{\\partial p} =
                \\frac{\\partial K}{\\partial p}\\\\
            \\frac{\\mathrm{d}p}{\\mathrm{d}t} &= -\\frac{\\partial H}{\\partial q} =
                -\\frac{\\partial V}{\\partial q}
        \\end{align*}


    Args:
        potential: Function which evaluate the potential energy and the gradient at a given position
        metric: Diagonal or Dense Euclidean metric

    References:
        Betancourt, M., 2017. A conceptual introduction to Hamiltonian Monte Carlo.
        arXiv preprint arXiv:1701.02434.
    """

    def __init__(self, potential: Callable, metric: EuclideanMetric):
        self._V_dV = potential
        self._metric = metric

        if not isinstance(metric, EuclideanMetric):
            raise TypeError("`metric` must be an EuclideanMetric")

    def V(self, q: np.array) -> float:
        """Evaluate the potential energy function :math:`V(q)`at the position `q`

        Args:
            q: Position variable

        Returns:
            Potential energy
        """
        return self._V_dV(q)[0]

    def dV(self, q: np.array) -> np.ndarray:
        """Evaluate the gradient of the potential energy function
        :math:`\\frac{\\partial V(q)} {\\partial q}` at the position `q`

        Args:
            q: Position variable

        Returns:
            Gradient of the potential energy
        """
        return self._V_dV(q)[1]

    def V_and_dV(self, q: np.array) -> Tuple[float, np.ndarray]:
        """Evaluate the potential energy function :math:`V(q)` and the gradient of the potential
        energy function :math:`\\frac{\\partial V(q)} {\\partial q}` at the position `q`

        Args:
            q: Position variable

        Returns:
            2-element tuple containing
                - Potential energy
                - Gradient of the potential energy
        """
        return self._V_dV(q)

    def K(self, p: np.array) -> float:
        """Evaluate the kinetic energy function :math:`K(p)` at the momentum variable `p`

        Args:
            p: Momentum variable

        Returns:
            Kinetic energy
        """
        return self._metric.kinetic_energy(momentum=p)

    def dK(self, p: np.array) -> np.ndarray:
        """Evaluate the gradient of the kinetic energy function
        :math:`\\frac{\\partial K}{\\partial p}` at the momentum variable `p`

        Args:
            p: Momentum variable

        Returns:
            Gradient of the kinetic energy
        """
        return self._metric.gradient_kinetic_energy(momentum=p)

    def sample_p(self) -> np.ndarray:
        """Sample momentum

        Returns:
            Momentum variables
        """
        return self._metric.sample_momentum()

    def H(self, q: np.array, p: np.array) -> float:
        """Evaluate the energy, i.e. the Hamiltonian in phase space :math:`H(q, p)`

        Args:
            q: Position variable
            p: Momentum variable

        Returns:
            Energy
        """
        return self.K(p) + self.V(q)

    @property
    def inverse_mass_matrix(self) -> np.ndarray:
        """Return the inverse of the mass matrix `M`"""
        return self._metric.get_inverse_metric()

    @inverse_mass_matrix.setter
    def inverse_mass_matrix(self, M):
        """Set the inverse of the mass matrix `M`

        Args:
            M: Mass matrix (Full matrix or only diagonal elements)
        """
        return self._metric.set_inverse_metric(M)
