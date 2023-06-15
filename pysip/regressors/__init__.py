from .frequentist import Regressor
from .bayesian import BayesRegressor

FreqRegressor = Regressor

__all__ = ["Regressor", "BayesRegressor", "FreqRegressor"]
