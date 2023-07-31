from .base import GPModel, RCModel, StateSpace, model_registry
from .latent_force_model import LatentForceModel

from .gaussian_process import __all__ as gaussian_process
from .thermal_network import __all__ as thermal_network


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
    "StateSpace",
    "Models",
    *thermal_network,
    *gaussian_process,
]