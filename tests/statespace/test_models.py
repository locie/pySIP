from itertools import chain

import pytest

from pysip.statespace import RCModel
from pysip.statespace.meta import model_registry, statespace

rc_models = {
    k: v for k, v in model_registry.items() if 'pysip.statespace.thermal_network' in v.__module__
}
# gp_models = {k: v for k, v in model_registry.items() if 'bopt.statespace.gp' in v.__module__}

all_models = rc_models

model = pytest.mark.parametrize('model', all_models.values())
name_model = pytest.mark.parametrize('name,model', all_models.items())

model_base = pytest.mark.parametrize('model,base', [(m, RCModel) for m in rc_models.values()])


def test_registry():
    assert isinstance(model_registry, dict)
    assert model_registry


@name_model
def test_factory(name, model):
    assert statespace(name) is model


@model
def test_model_has_correct_attributes(model):
    variables = ['inputs', 'outputs', 'states', 'params']
    methods = [
        'set_constant_continuous_ssm',
        'set_constant_continuous_dssm',
        'update_continuous_ssm',
        'update_continuous_dssm',
    ]
    for attr in variables + methods:
        assert hasattr(model, attr)


@model_base
def test_inheritance(model, base):
    assert base in model.__bases__


@model
def test_docstring(model):
    sections = ['Inputs', 'Outputs', 'States', 'Model parameters']
    subsections = [
        'Thermal capacity',
        'Initial deviation',
        'Measure deviation',
        'Initial mean',
        'State deviation',
        'Thermal resistance',
    ]

    assert model.__doc__

    for sec in chain(sections, subsections):
        assert sec in model.__doc__
