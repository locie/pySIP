import pytest
import numpy as np
import pandas as pd
import numdifftools as nd

from pysip.utils import generate_sine, ned
from pysip.regressors import FreqRegressor as Regressor
from pysip.statespace import Matern12, Matern32, Matern52
from pysip.statespace import Periodic, QuasiPeriodic12, QuasiPeriodic32


@pytest.fixture
def data_Matern():
    """Generate artificial data for Matern covariance"""
    np.random.seed(1)
    N = 50
    t = np.sort(np.random.rand(1, N), axis=1).flatten()
    y = np.sin(12 * t) + 0.66 * np.cos(25 * t) + np.random.randn(1, N) * 0.01
    y = y.flatten()
    return pd.DataFrame(index=t, data=y, columns=['y'])


@pytest.fixture
def data_Periodic():
    """Generate artificial data for Periodic covariance"""
    t, y = generate_sine(n=50, bounds=(0, 2), period=1.0, amplitude=1.0, random=True, noise_std=0.3)
    return pd.DataFrame(index=t, data=y, columns=['y'])


def generate_regressor(*statespaces):
    """State-space generator for gradient test"""
    values = np.random.uniform(0.3, 2.0, 3)
    p = [
        dict(name='mscale', value=values[0], transform='log'),
        dict(name='lscale', value=values[1], transform='log'),
        dict(name='sigv', value=values[2], transform='log'),
    ]
    for ssm in statespaces:
        yield Regressor(ss=ssm(parameters=p))


def generate_regressor_product(*statespaces):
    """State-space generator for gradient test"""
    values = np.random.uniform(1.0, 2.0, 5)

    par_Periodic = [
        dict(name='period', value=values[0], bounds=(0.0, None)),
        dict(name='mscale', value=values[1], bounds=(0.0, None)),
        dict(name='lscale', value=values[2], bounds=(0.0, None)),
        dict(name='sigv', value=values[3], bounds=(0.0, None)),
    ]

    par_Matern = [
        dict(name='mscale', value=1.0, transform='fixed'),
        dict(name='lscale', value=values[4], bounds=(0.0, None)),
        dict(name='sigv', value=0.0, transform='fixed'),
    ]

    for ssm in statespaces:
        yield Regressor(ss=Periodic(parameters=par_Periodic) * ssm(parameters=par_Matern))


def generate_regressor_sum(*statespaces):
    """State-space generator for gradient test"""
    values = np.random.uniform(0.3, 3.0, 6)

    par_Periodic = [
        dict(name="period", value=values[0], bounds=(0.0, None)),
        dict(name='mscale', value=values[1], bounds=(0.0, None)),
        dict(name='lscale', value=values[2], bounds=(0.0, None)),
        dict(name='sigv', value=values[3], bounds=(0.0, None)),
    ]

    par_Matern = [
        dict(name="mscale", value=values[4], bounds=(0.0, None)),
        dict(name="lscale", value=values[5], bounds=(0.0, None)),
        dict(name="sigv", value=0.0, transform='fixed'),
    ]

    for ssm in statespaces:
        yield Regressor(Periodic(par_Periodic) + ssm(par_Matern))


def check_grad_fd(data, reg):
    reg._use_penalty = False
    reg._use_jacobian = True
    reg.ss.method = 'mfd'
    dt, u, u1, y, _ = reg._prepare_data(data, None, 'y')
    grad = reg._eval_dlog_posterior(reg.ss.parameters.eta_free, dt, u, u1, y)[1]
    grad_fct = nd.Gradient(reg._eval_log_posterior)
    grad_fd = grad_fct(reg.ss.parameters.eta_free, dt, u, u1, y)
    print(f'grad: {grad}')
    print(f'R: {reg.ss.R}')

    assert ned(grad, grad_fd) < 1e-7
    assert np.all(np.sign(grad) == np.sign(grad_fd))
    assert grad == pytest.approx(grad_fd, rel=1e-6)


def check_grad(data, reg1, reg2):
    reg1._use_penalty = False
    reg1._use_jacobian = True
    reg2._use_penalty = False
    reg2._use_jacobian = True
    reg1.ss.method = 'mfd'
    reg2.ss.method = 'mfd'
    dt, u, u1, y, _ = reg1._prepare_data(data, None, 'y')
    grad1 = reg1._eval_dlog_posterior(reg1.ss.parameters.eta_free, dt, u, u1, y)[1]
    grad2 = reg2._eval_dlog_posterior(reg2.ss.parameters.eta_free, dt, u, u1, y)[1]
    assert ned(grad1, grad2) < 1e-7
    assert np.all(np.sign(grad1) == np.sign(grad2))
    assert grad1 == pytest.approx(grad2, rel=1e-6)


@pytest.mark.parametrize('regressor', generate_regressor(Matern12, Matern32, Matern52))
def test_Matern(data_Matern, regressor):
    check_grad_fd(data_Matern, regressor)


def test_Periodic(data_Periodic):
    p = [
        dict(name='period', transform='log'),
        dict(name='mscale', transform='log'),
        dict(name='lscale', transform='log'),
        dict(name='sigv', transform='log'),
    ]
    regressor = Regressor(ss=Periodic(parameters=p))
    regressor.ss.parameters.theta = np.random.uniform(0.3, 3.0, 4)
    check_grad_fd(data_Periodic, regressor)


@pytest.mark.parametrize('regressor', generate_regressor_sum(Matern12, Matern32, Matern52))
def test_gp_sum(data_Periodic, regressor):
    check_grad_fd(data_Periodic, regressor)


@pytest.mark.parametrize('regressor', generate_regressor_product(Matern12, Matern32, Matern52))
def test_gp_product(data_Periodic, regressor):
    check_grad_fd(data_Periodic, regressor)


def test_QuasiPeriodic(data_Periodic):
    p = [
        dict(name='period', transform='log'),
        dict(name='mscale', transform='log'),
        dict(name='lscale', transform='log'),
        dict(name='decay', transform='log'),
        dict(name='sigv', transform='log'),
    ]
    regressor = Regressor(ss=QuasiPeriodic12(parameters=p))
    regressor.ss.parameters.theta = np.random.uniform(0.3, 3.0, 5)
    check_grad_fd(data_Periodic, regressor)


@pytest.mark.parametrize('models', [[Matern12, QuasiPeriodic12], [Matern32, QuasiPeriodic32]])
def test_gp_product_creation(data_Periodic, models):
    m, qp = models
    period, mscale, lscale, decay, sigv = np.random.uniform(0.3, 3.0, 5)

    par_Periodic = [
        dict(name='period', value=period, transform='log'),
        dict(name='mscale', value=mscale, transform='log'),
        dict(name='lscale', value=lscale, transform='log'),
        dict(name='sigv', value=sigv, transform='log'),
    ]

    par_Matern = [
        dict(name='mscale', value=1.0, transform='fixed'),
        dict(name='lscale', value=decay, transform='log'),
        dict(name='sigv', value=0.0, transform='fixed'),
    ]

    par_QuasiPeriodic = [
        dict(name='period', value=period, transform='log'),
        dict(name='mscale', value=mscale, transform='log'),
        dict(name='lscale', value=lscale, transform='log'),
        dict(name='sigv', value=sigv, transform='log'),
        dict(name='decay', value=decay, transform='log'),
    ]

    reg1 = Regressor(ss=qp(par_QuasiPeriodic))
    reg2 = Regressor(ss=Periodic(par_Periodic) * m(par_Matern))
    check_grad(data_Periodic, reg1, reg2)
