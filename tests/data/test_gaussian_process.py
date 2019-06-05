import pytest

import numpy as np
import pandas as pd

from bopt.core.prior import InverseGamma as iGa
from bopt.regressors import Regressor
from bopt.statespace.gaussian_process.matern import Matern32


@pytest.fixture
def data():
    np.random.seed(1)
    N = 20

    t = np.sort(np.random.rand(1, N), axis=1).flatten()
    y = np.sin(12 * t) + 0.66 * np.cos(25 * t) + np.random.randn(1, N) * 0.01
    return pd.DataFrame(index=t, data={'y': y.flatten()})


@pytest.fixture
def matern_statespace():
    return Matern32([
        dict(name="mscale", value=0.5, transform="log", prior=iGa(3, 1)),
        dict(name="lscale", value=0.5, transform="log", prior=iGa(3, 1)),
        dict(name="sigv", value=0.1, transform="log", prior=iGa(3, 0.1))
    ])


@pytest.fixture
def regressor(matern_statespace):
    return Regressor(ss=matern_statespace)


def test_eval_log_likelihood(data, regressor):
    regressor.eval_log_likelihood(df=data, outputs='y')

    assert regressor.log_likelihood == pytest.approx(180.972, rel=1e-3)


def test_eval_dlog_likelihood(data, regressor):
    regressor.filter._compute_hessian = True
    data = regressor._prepare_data(df=data, inputs=None, outputs='y')
    regressor._eval_dlog_likelihood(*data)

    assert regressor.log_likelihood == pytest.approx(180.972, rel=1e-3)
    assert regressor.dlog_likelihood == pytest.approx([-450.385, 573.485, -1533.999], rel=1e-3)
    assert np.linalg.det(regressor.d2log_likelihood) == pytest.approx(1137422369.868, rel=1e-3)
    assert np.trace(regressor.d2log_likelihood) == pytest.approx(19790.282, rel=1e-3)


def test_eval_log_posterior(data, regressor):
    data = regressor._prepare_data(df=data, inputs=None, outputs='y')
    regressor._eval_log_posterior(regressor.ss.parameters.eta, *data)

    assert regressor.log_posterior == pytest.approx(180.204, rel=1e-3)


def test_eval_dlog_posterior(data, regressor):
    data = regressor._prepare_data(df=data, inputs=None, outputs='y')
    regressor._eval_dlog_posterior(regressor.ss.parameters.eta, *data)

    assert regressor.log_posterior == pytest.approx(180.204, rel=1e-3)
    assert regressor.dlog_posterior == pytest.approx([-223.193, 288.742, -150.399], rel=1e-3)


def test_eval_d2log_posterior(data, regressor):
    regressor.filter._compute_hessian = True
    data = regressor._prepare_data(df=data, inputs=None, outputs='y')
    regressor._eval_d2log_posterior(regressor.ss.parameters.eta, *data)

    assert regressor.log_posterior == pytest.approx(180.204, rel=1e-3)
    assert regressor.dlog_posterior == pytest.approx([-223.193, 288.742, -150.399], rel=1e-3)
    assert np.linalg.det(regressor.d2log_posterior) == pytest.approx(699760.890, rel=1e-3)
    assert np.trace(regressor.d2log_posterior) == pytest.approx(713.206, rel=1e-3)


def test_kalman_filter(data, regressor):
    _, P = regressor.estimate_states(df=data, outputs='y')
    residuals = regressor.eval_residuals(df=data, outputs='y')

    assert np.sum(P.trace(0, 1, 2)) == pytest.approx(36.541, rel=1e-3)
    assert np.sum(residuals) == pytest.approx(1.401, rel=1e-3)


def test_rts_smoother(data, regressor):
    _, P = regressor.estimate_states(df=data, outputs='y', smooth=True)

    assert np.sum(P.trace(0, 1, 2)) == pytest.approx(20.28508, rel=1e-3)


def test_fit_predict(data, regressor):
    summary, _ = regressor.fit(df=data, outputs='y')

    tpred = np.linspace(-0.1, 1.1, 100)
    _, ysf = regressor.predict(df=data, outputs="y", tpred=tpred, smooth=False)
    _, yss = regressor.predict(df=data, outputs="y", tpred=tpred, smooth=True)

    assert regressor.log_posterior == pytest.approx(-1.212, rel=1e-3)
    assert regressor.log_likelihood == pytest.approx(-0.09889, rel=1e-3)
    assert regressor.ss.parameters.theta == pytest.approx([1.044, 1.495e-1, 1.621e-2], rel=1e-3)
    assert np.array(summary.iloc[:, 1]) == pytest.approx([2.753e-1, 3.518e-2, 5.814e-3], rel=1e-3)
    assert np.all(summary.iloc[:, 3] < 1e-6)
    assert np.all(summary.iloc[:, 4] < 1e-14)

    assert ysf.sum() == pytest.approx(47.071, rel=1e-3)
    assert yss.sum() == pytest.approx(27.7837, rel=1e-3)
