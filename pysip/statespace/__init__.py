from .base import GPModel, RCModel, StateSpace, model_registry
from .gaussian_process import GPProduct, GPSum
from .latent_force_model import LatentForceModel, R2C2_Qgh_Matern32
from .gaussian_process import * # noqa, FIXME: depreciation cycle + force Models
from .thermal_network import * # noqa, FIXME: depreciation cycle + force Models

Models = model_registry

__all__ = [
    "GPModel",
    "RCModel",
    "StateSpace",
    "GPProduct",
    "GPSum",
    "LatentForceModel",
    "R2C2_Qgh_Matern32",
    "model_registry"
]
