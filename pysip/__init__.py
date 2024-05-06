import numpy as np

try:
    np.distutils.__config__.blas_opt_info = np.distutils.__config__.blas_ilp64_opt_info
except Exception:
    pass
import importlib.metadata

__version__ = importlib.metadata.version(__package__ or __name__)

from .statespace_estimator import KalmanQR
from .regressors import Regressor
from . import statespace as Models

__all__ = ["Regressor", "Models", "KalmanQR", "Models"]
