import numpy as np
import pytest

from pysip.statespace import GPProduct, Matern12, Matern32, Matern52


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
def product(matern12, matern32):
    product = GPProduct(matern12, matern32)
    product.parameters.theta = list(range(1, 7))
    product.update_continuous_dssm()
    return product


@pytest.fixture
def product_double(product, matern52):
    product_double = product * matern52
    product_double.parameters.theta = list(range(1, 10))
    product_double.update_continuous_dssm()
    return product_double


matern_params = ["mscale", "lscale", "sigv"]


def suffix(s):
    return [s + "__" + x for x in matern_params]


def test_names(product, matern12, matern32):
    assert product._names == suffix(matern12.name) + suffix(matern32.name)
    assert list(product.parameters._parameters.keys()) == ["m12", "Matern32"]
    assert list(product.parameters._parameters["m12"].keys()) == matern_params
    assert list(product.parameters._parameters["Matern32"].keys()) == matern_params


def test_names_double(product_double, product, matern12, matern32, matern52):
    assert product_double._names == (
        suffix(product.name + "__" + matern12.name)
        + suffix(product.name + "__" + matern32.name)
        + suffix(matern52.name)
    )
    par = product_double.parameters._parameters
    assert list(par.keys()) == ["m12__x__Matern32", "m52"]
    assert list(par["m12__x__Matern32"].keys()) == ["m12", "Matern32"]
    assert list(par["m52"].keys()) == matern_params
    assert list(par["m12__x__Matern32"]["m12"].keys()) == matern_params
    assert list(par["m12__x__Matern32"]["Matern32"].keys()) == matern_params


def test_jacobian(product, matern12, matern32):
    assert list(product.dP0.keys()) == suffix(matern12.name) + suffix(matern32.name)
    for k in product.dP0:
        assert product.dP0[k].shape == (
            matern12.nx * matern32.nx,
            matern12.nx * matern32.nx,
        )


def test_jacobian_double(product_double, product, matern12, matern32, matern52):
    assert list(product_double.dP0.keys()) == (
        suffix(product.name + "__" + matern12.name)
        + suffix(product.name + "__" + matern32.name)
        + suffix(matern52.name)
    )
    for k in product_double.dP0:
        assert product_double.dP0[k].shape == (
            matern12.nx * matern32.nx * matern52.nx,
            matern12.nx * matern32.nx * matern52.nx,
        )
