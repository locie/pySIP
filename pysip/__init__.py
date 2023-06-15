import importlib.metadata

__version__ = importlib.metadata.version(__package__ or __name__)

from .statespace_estimator import KalmanQR
from .statespace import Models
from .regressors import FreqRegressor, BayesRegressor

__all__ = ["Models", "FreqRegressor", "BayesRegressor", "KalmanQR"]