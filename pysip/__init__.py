import importlib.metadata

__version__ = importlib.metadata.version(__package__ or __name__)

from .statespace_estimator import KalmanQR
from .statespace import Models
from .regressors import Regressor

__all__ = ["Models", "Regressor", "KalmanQR"]