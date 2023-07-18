import numpy as np
import pandas as pd
import pytest

from pysip.params.prior import InverseGamma as iGa
from pysip.regressors import FreqRegressor as Regressor
from pysip.statespace import Matern32


@pytest.fixture
def data_raw():
    np.random.seed(1)
    N = 20

    t = np.sort(np.random.rand(1, N), axis=1).flatten()
    y = np.sin(12 * t) + 0.66 * np.cos(25 * t) + np.random.randn(1, N) * 0.01
    return pd.DataFrame(index=t, data={"y": y.flatten()})


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
    return Regressor(ss=matern_statespace, outputs="y")


@pytest.fixture
def data_prep(data_raw: pd.DataFrame, regressor: Regressor):
    return regressor._prepare_data(df=data_raw, inputs=None)[:-1]


def test_log_likelihood(data_raw: pd.DataFrame, regressor: Regressor):
    ll_ = regressor.log_likelihood(df=data_raw)
    assert ll_ == pytest.approx(180.972, rel=1e-3)


def test_log_posterior(data_raw, regressor: Regressor):
    lp_ = regressor.log_posterior(df=data_raw)
    assert lp_ == pytest.approx(180.204, rel=1e-3)


def test_kalman_filter(data_raw: pd.DataFrame, regressor: Regressor):
    ds_residuals = regressor.eval_residuals(df=data_raw)
    ds_filtered = regressor.estimate_states(df=data_raw)
    assert float(ds_residuals["residual"].sum()) == pytest.approx(1.401, rel=1e-3)
    assert float(ds_filtered["P"].data.trace(0, 1, 2).sum()) == pytest.approx(
        36.541, rel=1e-3
    )


def test_rts_smoother(data_raw: pd.DataFrame, regressor: Regressor):
    ds_filtered = regressor.estimate_states(df=data_raw, smooth=True)
    assert np.sum(ds_filtered["P"].data.trace(0, 1, 2).sum()) == pytest.approx(
        20.28508, rel=1e-3
    )


def test_fit_predict(data_raw: pd.DataFrame, regressor: Regressor):
    summary, _, summary_scipy = regressor.fit(df=data_raw)
    log_likelihood = regressor.log_likelihood(df=data_raw)

    tnew = np.linspace(-0.1, 1.1, 100)
    ds_filtered = regressor.predict(
        df=data_raw, tnew=tnew, smooth=False, use_outputs=True
    )
    ds_smoothed = regressor.predict(
        df=data_raw, tnew=tnew, smooth=True, use_outputs=True
    )

    assert summary_scipy.fun == pytest.approx(-1.212, rel=1e-2)
    assert log_likelihood == pytest.approx(-0.09889, rel=1e-2)
    assert regressor.ss.parameters.theta == pytest.approx(
        [1.044, 1.495e-1, 1.621e-2], rel=1e-2
    )
    assert np.array(summary.iloc[:, 1]) == pytest.approx(
        [2.753e-1, 3.518e-2, 5.814e-3], rel=1e-2
    )
    assert np.all(summary.iloc[:, 3] < 1e-4)
    assert np.all(summary.iloc[:, 4] < 1e-10)
    assert float(ds_filtered["y_std"].sum()) == pytest.approx(47.071, rel=1e-3)
    assert float(ds_smoothed["y_std"].sum()) == pytest.approx(27.7837, rel=1e-3)
