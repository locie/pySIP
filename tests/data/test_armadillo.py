import pytest
import numpy as np
import pandas as pd

from pysip.statespace.thermal_network import TwTi_RoRi
from pysip.regressors import FreqRegressor as Regressor
from pysip.core import Normal
from pysip.utils import check_model, fit


@pytest.fixture
def data_armadillo():
    return pd.read_csv('data/armadillo/armadillo_data_H2.csv').set_index('Time')


@pytest.fixture
def parameters_armadillo():
    return [
        dict(name="Ro", value=0.1, bounds=(0, None), prior=Normal(0.1, 0.04)),
        dict(name="Ri", value=0.01, bounds=(0, None), prior=Normal(0.05, 0.02)),
        dict(name="Cw", value=0.1, scale=1e8, bounds=(0, None), prior=Normal(0.1, 0.04)),
        dict(name="Ci", value=0.01, scale=1e8, bounds=(0, None), prior=Normal(0.05, 0.02)),
        dict(name="sigw_w", value=0.01, bounds=(0, None), prior=Normal(0.05, 0.02)),
        dict(name="sigw_i", value=0.0, transform='fixed'),
        dict(name="sigv", value=0.01, bounds=(0, None), prior=Normal(0.05, 0.02)),
        dict(name="x0_w", value=0.25, scale=1e2, bounds=(0, None), prior=Normal(0.25, 0.07)),
        dict(name="x0_i", value=26.70106194217502, transform='fixed'),
        dict(name="sigx0_w", value=1.0, transform='fixed'),
        dict(name="sigx0_i", value=1.0, transform='fixed'),
    ]


@pytest.fixture
def statespace_armadillo(parameters_armadillo):
    return TwTi_RoRi(parameters_armadillo, hold_order=1)


@pytest.fixture
def regressor_armadillo(statespace_armadillo):
    return Regressor(ss=statespace_armadillo)


def test_fit_predict(data_armadillo, regressor_armadillo):
    summary, corr, scipy_summary = regressor_armadillo.fit(
        df=data_armadillo, outputs='T_int', inputs=['T_ext', 'P_hea']
    )

    log_likelihood = regressor_armadillo.eval_log_likelihood(
        df=data_armadillo, outputs='T_int', inputs=['T_ext', 'P_hea']
    )

    ym, _ = regressor_armadillo.predict(df=data_armadillo, inputs=['T_ext', 'P_hea'])

    assert scipy_summary.fun == pytest.approx(-252.707, rel=1e-3)
    assert log_likelihood == pytest.approx(-244.283, rel=1e-3)
    assert fit(data_armadillo['T_int'].values, ym) == pytest.approx(0.807, rel=1e-2)
    assert check_model(regressor_armadillo, summary, corr, verbose=False)
