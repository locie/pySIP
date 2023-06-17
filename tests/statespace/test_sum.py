import pytest

from pysip.statespace import GPSum, Matern12, Matern32, Matern52


@pytest.fixture
def matern12():
    return Matern12(name="m12")


@pytest.fixture
def matern32():
    return Matern32()


@pytest.fixture
def matern52():
    return Matern52(name="m52")


@pytest.fixture
def add(matern12, matern32):
    add = GPSum(matern12, matern32)
    add.parameters.theta = list(range(1, 7))
    add.update()
    return add


@pytest.fixture
def add_double(add, matern52):
    add_double = add + matern52
    add_double.parameters.theta = list(range(1, 10))
    add_double.update()
    return add_double


matern_params = ["mscale", "lscale", "sigv"]


def suffix(s):
    return [s + "__" + x for x in matern_params]


def test_names(add, matern12, matern32):
    assert add.parameters.ids == suffix(matern12.name) + suffix(matern32.name)
    assert list(add.parameters._parameters.keys()) == ["m12", "Matern32"]
    assert list(add.parameters._parameters["m12"].keys()) == matern_params
    assert list(add.parameters._parameters["Matern32"].keys()) == matern_params


def test_names_double(add_double, add, matern12, matern32, matern52):
    assert add_double.parameters.ids == (
        suffix(add.name + "__" + matern12.name)
        + suffix(add.name + "__" + matern32.name)
        + suffix(matern52.name)
    )
    par = add_double.parameters._parameters
    assert list(par.keys()) == ["m12__+__Matern32", "m52"]
    assert list(par["m12__+__Matern32"].keys()) == ["m12", "Matern32"]
    assert list(par["m52"].keys()) == matern_params
    assert list(par["m12__+__Matern32"]["m12"].keys()) == matern_params
    assert list(par["m12__+__Matern32"]["Matern32"].keys()) == matern_params
