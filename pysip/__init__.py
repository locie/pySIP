import importlib.metadata

__version__ = importlib.metadata.version(__package__ or __name__)

from .statespace_estimator import KalmanQR
from .regressors import Regressor
from . import statespace as Models

__all__ = ["Regressor", "Models", "KalmanQR", "Models"]
