"""linear algebra utilities"""
import numpy as np


def pseudo_cholesky(A, rcond=1e-14):
    """Pseudo Cholesky factor for symmetric matrices

    A SoftAbs metric is used to ensure that the reciprocal of
    the condition number is `rcond`

    Parameters
    ----------
    rcond : float
        reciprocal of the condition number

    Return
    ------
    pchol : array_like
        regularized upper triangular Cholesky factor of `A`

    """
    if not isinstance(rcond, (int, float)):
        raise TypeError("`rcond` must be a float")

    if rcond < 0:
        raise ValueError("`rcond` must be positive")

    # ensure the symmetry of A
    A += A.T
    A /= 2

    # get the eigenvalues and eigenvectors
    w, v = np.linalg.eigh(A)

    alpha = 1. / (rcond * np.max(w))

    # SoftAbs
    for i in range(A.shape[0]):
        alpha_lambda = alpha * w[i]
        if np.abs(alpha_lambda) < 1e-4:
            w[i] = (1. + (1. / 3.) * alpha_lambda**2) / alpha
        elif np.abs(alpha_lambda) > 18:
            w[i] = np.abs(w[i])
        else:
            w[i] /= np.tanh(alpha_lambda)

    # get the pseudo square root in upper triangular form
    pchol = np.linalg.qr(np.diag(np.sqrt(w)) @ v.T, "r")

    return pchol


class MultivariateNormal(object):
    """Multivariate Normal Probability Density Function

    A SoftAbs metric can be used to ensure that the Hessian is PSD

    Reference
    ---------
    Betancourt, M., 2013, August.
    A general metric for Riemannian manifold Hamiltonian Monte Carlo.
    In International Conference on Geometric Science of Information
    (pp. 327-334). Springer, Berlin, Heidelberg.

    """

    def __init__(self, dim, alpha=1e16):
        """Initialize and create constant values

        Parameters
        ----------
        dim : int
            Number of random variables
        alpha : float
            SoftAbs hardness parameter, e.g. smooth the eigenvalues
            at 1 / `alpha`

        """
        if not isinstance(dim, int):
            raise TypeError("`dim` must be an integer")

        if not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be an integer or a float")

        self._alpha = float(alpha)
        self._dim = dim
        self._mean = np.zeros(dim)
        self._covm = np.eye(dim)
        self._sqrt = np.eye(dim)
        self._prem = np.eye(dim)
        self._ln_det = 0.0
        self._Ix = np.eye(dim)
        self._ln_2PI = np.log(2 * np.pi)

    @property
    def mean(self):
        """Return the mean vector"""
        return self._mean

    @mean.setter
    def mean(self, x):
        """Set a mean vector"""
        self._mean = x

    @property
    def covariance(self):
        """Return the covariance matrix"""
        return self._covm

    @covariance.setter
    def covariance(self, x):
        """Set a covariance matrix"""

        # check PSD and return tolerance
        w, v = self._SoftAbs(x)

        # covariance matrix
        self._covm = x

        # square root covariance matrix
        self._sqrt = v @ np.diag(np.sqrt(w)) @ v.T

        # precision matrix
        self._prem = v @ np.diag(1 / w) @ v.T

        # logarithm determinant of the covariance matrix
        self._ln_det = np.sum(np.log(w))

    @property
    def precision(self):
        """Return the precision matrix"""
        return self._prem

    @precision.setter
    def precision(self, x):
        """Set the precision matrix"""

        # check PSD and return tolerance
        w, v = self._SoftAbs(x)

        # precision matrix
        self._prem = x

        # square root covariance matrix
        self._sqrt = v @ np.diag(np.sqrt(1 / w)) @ v.T

        # covariance matrix
        self._covm = self._sqrt @ self._sqrt

        # logarithm determinant of the covariance matrix
        self._ln_det = - np.sum(np.log(w))

    @property
    def square_root(self):
        """Return the square root of the covariance matrix"""
        return self._sqrt

    def _SoftAbs(self, x):
        """SoftAbs metric

        Smooth absolute value of the eigenvalues at 1 / `alpha`

        Parameters
        ----------
        x : array_like
            A symmetric matrix

        Return
        ------
        w : array_like
            eigenvalues
        v : array_like
            eigenvectors

        """
        # ensure the symmetry of the covariance matrix
        x += x.T
        x /= 2

        # get the eigenvalues and eigenvectors
        w, v = np.linalg.eigh(x)

        # soft absolute value at the threshold 1/alpha of the eigenvalues
        if self._alpha is not None:
            for i in range(self._dim):
                alpha_lambda = self._alpha * w[i]
                if np.abs(alpha_lambda) < 1e-4:
                    w[i] = (1. + (1. / 3.) * alpha_lambda**2) / self._alpha
                elif np.abs(alpha_lambda) > 18:
                    w[i] = np.abs(w[i])
                else:
                    w[i] /= np.tanh(alpha_lambda)

        return w, v

    def log_pdf(self, x):
        """logarithm of the probability density function at `x`"""

        e = (x - self._mean).squeeze()
        return -0.5 * (self._dim * self._ln_2PI + self._ln_det + e @ self._prem @ e)

    def random(self, n=None):
        """Generate random samples

        Parameters
        ----------
        n : int
            Number of samples to generate

        Return
        ------
        rvs : array_like
            random samples

        """
        if n is None:
            rvs = self._mean + self._sqrt @ np.random.randn(self._dim)
        else:
            rvs = (self._mean[:, np.newaxis]
                   + self._sqrt @ np.random.randn(self._dim, int(n)))

        return rvs
