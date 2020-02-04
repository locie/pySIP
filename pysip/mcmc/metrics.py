import numpy as np
from scipy.linalg import LinAlgError, cholesky, cho_solve
from ..utils.math import cholesky_inverse


class EuclideanMetric:
    """Abstract class for Euclidean-Gaussian kinetic energy metric"""

    def get_inverse_metric(self):
        """Return the inverse of the metric"""
        raise NotImplementedError

    def set_inverse_metric(self, inverse_metric: np.ndarray):
        """Change the metric"""
        raise NotImplementedError

    def kinetic_energy(self, momentum: np.ndarray) -> float:
        """Evaluate the kinetic energy"""
        raise NotImplementedError

    def gradient_kinetic_energy(self, momentum: np.ndarray) -> np.ndarray:
        """Evaluate the gradient of the kinetic energy"""
        raise NotImplementedError

    def sample_momentum(self) -> np.ndarray:
        """Sample momentum"""
        raise NotImplementedError


class Diagonal(EuclideanMetric):
    """The metric is a diagonal matrix"""

    def __init__(self, inverse_metric: np.ndarray):
        """
        Args:
            inverse_metric: Inverse of the metric diagonal elements
        """
        self.set_inverse_metric(inverse_metric)

    def get_inverse_metric(self) -> np.ndarray:
        """Get the inverse of the metric

        Returns:
            Inverse of the metric
        """
        return self._inv_metric

    def set_inverse_metric(self, inverse_metric: np.ndarray):
        """Change the metric

        Args:
            inverse_metric: Inverse of the metric
        """
        if not isinstance(inverse_metric, np.ndarray):
            raise TypeError('`inverse_metric` must be an numpy ndarray')

        if not inverse_metric.ndim == 1:
            raise ValueError('`inverse_metric` must be 1-dimensional')

        if not np.all(inverse_metric > 0.0):
            raise ValueError('All `inverse_metric` elements must be positive')

        self._inv_metric = inverse_metric
        self._sqrt_metric = np.sqrt(np.reciprocal(inverse_metric))

    def kinetic_energy(self, momentum: np.ndarray) -> float:
        """Evaluate the kinetic energy at `momentum`"""
        return 0.5 * (self._inv_metric * momentum) @ momentum

    def gradient_kinetic_energy(self, momentum: np.ndarray) -> np.ndarray:
        """Evaluate the gradient of the kinetic energy at `momentum`"""
        return self._inv_metric * momentum

    def sample_momentum(self):
        """Sample momentum"""
        return self._sqrt_metric * np.random.randn(self._sqrt_metric.shape[0])


class Dense(EuclideanMetric):
    """The metric is a dense matrix"""

    def __init__(self, inverse_metric: np.ndarray):
        """
        Args:
            inverse_metric: Inverse dense mass matrix
        """
        self.set_inverse_metric(inverse_metric)

    def get_inverse_metric(self) -> np.ndarray:
        """Get the inverse of the metric

        Returns:
            Inverse of the metric
        """
        return self._inv_metric

    def set_inverse_metric(self, inverse_metric: np.ndarray):
        """Change the metric

        Args:
            inverse_metric: Inverse of the metric
        """
        if not isinstance(inverse_metric, np.ndarray):
            raise TypeError('`inverse_metric` must be an numpy ndarray')

        if not inverse_metric.ndim == 2:
            raise ValueError('`inverse_metric` must be 2-dimensional')

        if not np.allclose(inverse_metric, inverse_metric.T):
            raise ValueError('`inverse_metric` must be symmetric')

        self._inv_metric = inverse_metric
        self._sqrt_metric = cholesky_inverse(inverse_metric)

    def kinetic_energy(self, momentum: np.ndarray) -> float:
        """Evaluate the kinetic energy at `momentum`"""
        return 0.5 * momentum @ self._inv_metric @ momentum

    def gradient_kinetic_energy(self, momentum: np.ndarray) -> np.ndarray:
        """Evaluate the gradient of the kinetic energy at `momentum`"""
        return self._inv_metric @ momentum

    def sample_momentum(self):
        """Sample momentum"""
        return self._sqrt_metric @ np.random.randn(self._sqrt_metric.shape[0])
