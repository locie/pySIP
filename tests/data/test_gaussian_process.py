import numpy as np
import pandas as pd
import pytest

from pysip.core import InverseGamma as iGa
from pysip.regressors import FreqRegressor as Regressor
from pysip.statespace import Matern32


@pytest.fixture
def data_raw():
    np.random.seed(1)
    N = 20

    t = np.sort(np.random.rand(1, N), axis=1).flatten()
    y = np.sin(12 * t) + 0.66 * np.cos(25 * t) + np.random.randn(1, N) * 0.01
    return pd.DataFrame(index=t, data={'y': y.flatten()})


@pytest.fixture
def matern_statespace():
    return Matern32(
        [
            dict(name="mscale", value=0.5, transform="log", prior=iGa(3, 1)),
            dict(name="lscale", value=0.5, transform="log", prior=iGa(3, 1)),
            dict(name="sigv", value=0.1, transform="log", prior=iGa(3, 0.1)),
        ]
    )


@pytest.fixture
def regressor(matern_statespace):
    return Regressor(ss=matern_statespace)


@pytest.fixture
def data_prep(data_raw, regressor):
    return regressor._prepare_data(df=data_raw, inputs=None, outputs='y')[:-1]


def test_eval_log_likelihood(data_raw, regressor):
    ll_ = regressor.eval_log_likelihood(df=data_raw, outputs='y')
    assert ll_ == pytest.approx(180.972, rel=1e-3)


def test_eval_dlog_likelihood(data_prep, regressor):
    ll_, dll_ = regressor._eval_dlog_likelihood(*data_prep)
    assert ll_ == pytest.approx(180.972, rel=1e-3)
    assert dll_ == pytest.approx([-450.385, 573.485, -1533.999], rel=1e-3)


def test_eval_log_posterior(data_prep, regressor):
    lp_ = regressor._eval_log_posterior(regressor.ss.parameters.eta, *data_prep)
    assert lp_ == pytest.approx(180.204, rel=1e-3)


def test_eval_dlog_posterior(data_prep, regressor):
    lp_, dlp_ = regressor._eval_dlog_posterior(regressor.ss.parameters.eta, *data_prep)
    assert lp_ == pytest.approx(180.204, rel=1e-3)
    assert dlp_ == pytest.approx([-223.193, 288.742, -150.399], rel=1e-3)


def test_kalman_filter(data_raw, regressor):
    _, P = regressor.estimate_states(df=data_raw, outputs='y')
    residuals, *_ = regressor.eval_residuals(df=data_raw, outputs='y')
    assert np.sum(P.trace(0, 1, 2)) == pytest.approx(36.541, rel=1e-3)
    assert np.sum(residuals) == pytest.approx(1.401, rel=1e-3)


def test_rts_smoother(data_raw, regressor):
    _, P = regressor.estimate_states(df=data_raw, outputs='y', smooth=True)
    assert np.sum(P.trace(0, 1, 2)) == pytest.approx(20.28508, rel=1e-3)


def test_fit_predict(data_raw, regressor):
    summary, _, summary_scipy = regressor.fit(df=data_raw, outputs='y')
    log_likelihood = regressor.eval_log_likelihood(df=data_raw, outputs='y')

    tnew = np.linspace(-0.1, 1.1, 100)
    _, ysf = regressor.predict(df=data_raw, outputs="y", tnew=tnew, smooth=False)
    _, yss = regressor.predict(df=data_raw, outputs="y", tnew=tnew, smooth=True)

    assert summary_scipy.fun == pytest.approx(-1.212, rel=1e-3)
    assert log_likelihood == pytest.approx(-0.09889, rel=1e-3)
    assert regressor.ss.parameters.theta == pytest.approx([1.044, 1.495e-1, 1.621e-2], rel=1e-3)
    assert np.array(summary.iloc[:, 1]) == pytest.approx([2.753e-1, 3.518e-2, 5.814e-3], rel=1e-3)
    assert np.all(summary.iloc[:, 3] < 1e-6)
    assert np.all(summary.iloc[:, 4] < 1e-12)
    assert ysf.sum() == pytest.approx(47.071, rel=1e-3)
    assert yss.sum() == pytest.approx(27.7837, rel=1e-3)
