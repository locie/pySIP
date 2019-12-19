"""Hamiltonian Definition"""
import numpy as np
import scipy.linalg as sla
from typing import Callable


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


    For Euclidean-Gaussian kinetic energie

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
        V: Function for evaluating the potential energy
        dV: Function for evaluating the gradient of the potential energy
        M: Mass matrix

    References:
        Betancourt, M., 2017. A conceptual introduction to Hamiltonian Monte Carlo.
        arXiv preprint arXiv:1701.02434.

    Notes:
        Specify if dV return the log-posterior and the gradient. It is the default in DynamicHMC,
        a refactoring is required.
    """

    def __init__(self, V: Callable, dV: Callable, M: np.ndarray):
        self._V = V
        self._dV = dV
        self._M = None
        self._cholM = None
        self.M = M
        self.dim = M.shape[0]

    def V(self, q: np.array) -> float:
        """Potential energy :math:`V(q)`"""
        return self._V(q)

    def dV(self, q: np.array) -> np.ndarray:
        """Gradient of potential energy :math:`\\frac{\\partial V(q)} {\\partial q}`"""
        return self._dV(q)

    def K(self, p: np.array) -> float:
        """Kinetic energy :math:`K(p)`"""
        return 0.5 * np.sum(p * sla.cho_solve((self._cholM, True), p))

    def dK(self, p: np.array) -> np.ndarray:
        """Gradient of kinetic energy :math:`\\frac{\\partial K}{\\partial p}`"""
        return sla.cho_solve((self._cholM, True), p)

    def sample_p(self) -> np.ndarray:
        """Sample momentum"""
        return self._cholM @ np.random.randn(self.dim)

    def H(self, q: np.array, p: np.array) -> float:
        """The value of the Hamiltonian in phase space is called the energy at that point"""
        return self.K(p) + self.V(q)

    @property
    def M(self) -> np.ndarray:
        """Return the mass matrix `M`"""
        return self._M

    @M.setter
    def M(self, x):
        """Change the mass matrix `M`

        Args:
            mass_matrix: New mass matrix
        """
        if not len(x.shape) == 2:
            raise ValueError('The mass matrix must be 2-dimensional')

        if not np.allclose(x, x.T):
            raise ValueError('The mass matrix must be symmetric')

        try:
            self._M = x
            self._cholM = sla.cholesky(x, lower=True)
        except sla.LinAlgError as e:
            if 'Singular matrix' in str(e):
                raise ValueError('The mass matrix is not positive definite')

    @property
    def cholM(self) -> np.ndarray:
        """Return the lower triangular Cholesky factor the mass matrix `M`"""
        return self._cholM

    @cholM.setter
    def cholM(self, x):
        """Change the lower triangular Cholesky factor the mass matrix `M`"""
        if not len(x.shape) == 2:
            raise ValueError('The mass matrix must be 2-dimensional')
        self._cholM = x
        self._M = x @ x.T
