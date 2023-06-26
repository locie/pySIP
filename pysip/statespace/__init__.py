from .base import GPModel, RCModel, StateSpace, model_registry
from .gaussian_process import GPProduct, GPSum
from .latent_force_model import LatentForceModel, R2C2_Qgh_Matern32

from inspect import isclass
from pkgutil import iter_modules
from pathlib import Path
from importlib import import_module

package_dir = str(Path(__file__).resolve().parent)
for _, module_name, _ in iter_modules([package_dir]):
    module = import_module(f"{__name__}.{module_name}")
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)

        if isclass(attribute) and issubclass(attribute, StateSpace):
            globals()[attribute_name] = attribute

Models = model_registry

__all__ = [
    "GPModel",
    "RCModel",
    "StateSpace",
    "GPProduct",
    "GPSum",
    "LatentForceModel",
    "R2C2_Qgh_Matern32",
    "Models"
]
