import numpy as np
import pytest

from pysip.params import Parameters


@pytest.fixture
def parameters():
    return Parameters(["a", "b", "c"], name="alpha")


@pytest.fixture
def parameters_beta():
    return Parameters(["c", "d", "e"], name="beta")


@pytest.fixture
def parameters_with_transforms():
    parameters = [
        {"name": "a", "value": 1.0, "transform": "log"},
        {"name": "b", "value": 2.0, "transform": "logit", "bounds": (1.0, 3.0)},
    ]

    return Parameters(parameters=parameters, name="transform")


def test_parameters(parameters):
    assert np.allclose(parameters.theta, [0, 0, 0])


def test_repr(parameters, parameters_beta, parameters_with_transforms):
    print(parameters.__repr__())
    assert (
        parameters.__repr__()
        == """Parameters alpha
name=a value=0.000e+00 transform=none bounds=(None, None) prior=None
name=b value=0.000e+00 transform=none bounds=(None, None) prior=None
name=c value=0.000e+00 transform=none bounds=(None, None) prior=None
"""
    )

    s = parameters + parameters_beta + parameters_with_transforms
    assert (
        s.__repr__()
        == """Parameters alpha__beta__transform
* alpha__beta
    * alpha
        name=a value=0.000e+00 transform=none bounds=(None, None) prior=None
        name=b value=0.000e+00 transform=none bounds=(None, None) prior=None
        name=c value=0.000e+00 transform=none bounds=(None, None) prior=None
    * beta
        name=c value=0.000e+00 transform=none bounds=(None, None) prior=None
        name=d value=0.000e+00 transform=none bounds=(None, None) prior=None
        name=e value=0.000e+00 transform=none bounds=(None, None) prior=None
* transform
    name=a value=1.000e+00 transform=log bounds=(None, None) prior=None
    name=b value=2.000e+00 transform=logit bounds=(1.0, 3.0) prior=None
"""
    )


def test_transform(parameters_with_transforms):
    assert parameters_with_transforms.eta == pytest.approx([0.0, 0.0])


def test_set_parameter(parameters):
    assert parameters.parameters[0].name == "a"
    assert parameters.parameters[0].value == 0.0

    parameters.set_parameter("a", value=3.0)

    assert parameters.parameters[0].value == 3.0


def test_set_theta(parameters):
    assert np.allclose(parameters.theta, [0, 0, 0])
    parameters.theta = [1.0, 0.0, 0.0]
    assert np.allclose(parameters.theta, [1.0, 0.0, 0.0])


def test_add(parameters, parameters_with_transforms):
    new_parameters = parameters + parameters_with_transforms

    assert isinstance(new_parameters, Parameters)
    assert np.allclose(new_parameters.theta, [0.0, 0.0, 0.0, 1.0, 2.0])


def test_add_recursive(parameters, parameters_beta, parameters_with_transforms):
    new_parameters = parameters + parameters_beta
    new_parameters = new_parameters + parameters_with_transforms

    assert isinstance(new_parameters, Parameters)
    assert np.allclose(new_parameters.theta, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0])


def test_add_set_theta(parameters, parameters_beta):
    new_parameters = parameters + parameters_beta

    assert np.allclose(new_parameters.theta, [0, 0, 0, 0, 0, 0])

    new_parameters.theta = [1.0, 0.0, 0.0, 2.0, 3.0, 0.0]

    assert np.allclose(new_parameters.theta, [1.0, 0.0, 0.0, 2.0, 3.0, 0.0])
    assert np.allclose(parameters.theta, [1.0, 0.0, 0.0])
    assert np.allclose(parameters_beta.theta, [2.0, 3.0, 0.0])


def test_add_set_parameters(parameters, parameters_with_transforms):
    parameters = parameters + parameters_with_transforms

    assert parameters.parameters[0].name == "a"
    assert parameters.parameters[0].value == 0.0

    parameters.set_parameter("alpha", "a", value=3.0)

    assert parameters.parameters[0].value == 3.0


def test_unknown_parameter(parameters):
    parameters.set_parameter("lol", value=-1.0)

    assert "lol" not in parameters._parameters
