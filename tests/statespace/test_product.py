import pytest

import numpy as np

from bopt.statespace.gaussian_process.gp_product import GPProduct
from bopt.statespace.gaussian_process.matern import Matern12, Matern32, Matern52


@pytest.fixture
def matern12():
    return Matern12(name='m12')

@pytest.fixture
def matern32():
    return Matern32()

@pytest.fixture
def matern52():
    return Matern52(name='m52')


@pytest.fixture
def product(matern12, matern32):
    product = GPProduct(matern12, matern32)
    product.parameters.theta = list(range(1, 7))
    return product

@pytest.fixture
def product_double(product, matern52):
    product_double = product * matern52
    #product_double = GPProduct(product, matern52)
    product_double.parameters.theta = list(range(1, 10))
    return product_double

matern_params = ['mscale', 'lscale', 'sigv']
def suffix(s):
    return [s + '__' + x for x in matern_params]


def test_names(product, matern12, matern32):
    assert product._names == suffix(matern12.name) + suffix(matern32.name)

    assert list(product.parameters._parameters.keys()) == ['m12', 'Matern32']
    assert list(product.parameters._parameters['m12'].keys()) == matern_params
    assert list(product.parameters._parameters['Matern32'].keys()) == matern_params


def test_names_double(product_double, product, matern12, matern32, matern52):
    assert product_double._names == (
        suffix(product.name + '__' + matern12.name) 
        + suffix(product.name + '__' + matern32.name)
        + suffix(matern52.name)
    )

    assert list(product_double.parameters._parameters.keys()) == ['GPProduct', 'm52']
    assert list(product_double.parameters._parameters['GPProduct'].keys()) == ['m12', 'Matern32']
    
    assert list(product_double.parameters._parameters['m52'].keys()) == matern_params
    assert list(product_double.parameters._parameters['GPProduct']['m12'].keys()) == matern_params
    assert list(product_double.parameters._parameters['GPProduct']['Matern32'].keys()) == matern_params


def test_jacobian(product, matern12, matern32):
    assert list(product.dB.keys()) == suffix(matern12.name) + suffix(matern32.name)
    for k in product.dB:
        assert product.dB[k].shape == (matern12.Nx * matern32.Nx, 0)


def test_jacobian_double(product_double, product, matern12, matern32, matern52):
    assert list(product_double.dB.keys()) == (
        suffix(product.name + '__' + matern12.name) 
        + suffix(product.name + '__' + matern32.name)
        + suffix(matern52.name)
    )
    for k in product_double.dB:
        assert product_double.dB[k].shape == (matern12.Nx * matern32.Nx * matern52.Nx, 0)


@pytest.mark.xfail
def test_update(product):
    product.update()

    np.testing.assert_equal(product.dA['m12__lscale'], np.array([])) # TODO
    np.testing.assert_equal(product.dA['Matern32__lscale'], np.array([])) # TODO

    np.testing.assert_equal(product.dB['m12__lscale'], np.array([])) # TODO
    np.testing.assert_equal(product.dB['Matern32__lscale'], np.array([])) # TODO


@pytest.mark.xfail
def test_update_double(product_double):
    product_double.update()

    np.testing.assert_equal(product_double.dA['m52__lscale'], np.array([])) # TODO
    np.testing.assert_equal(product_double.dB['m52__lscale'], np.array([])) # TODO