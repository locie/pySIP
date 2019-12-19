import pytest
import numpy as np
import pandas as pd

from pysip.statespace.thermal_network import TwTi_RoRi
from pysip.regressors import FreqRegressor as Regressor
from pysip.core import Normal, LogNormal
from pysip.utils.check import check_model
from pysip.utils.math import fit, rmse, mae, mad, smape, ned
from pysip.utils.statistics import aic, lrtest, check_ccf, check_cpgram, ccf, cpgram


@pytest.fixture
def data_armadillo():
    df = pd.read_csv('data/armadillo/armadillo_data_H2.csv').set_index('Time')
    df.drop(df.index[-1], axis=0, inplace=True)
    return df


@pytest.fixture
def parameters_armadillo():
    sT = 3600.0 * 24.0
    return [
        dict(name='Ro', scale=1e-2, bounds=(0, None), prior=LogNormal(1, 1)),
        dict(name='Ri', scale=1e-3, bounds=(0, None), prior=LogNormal(1, 1)),
        dict(name='Cw', scale=1e7 / sT, bounds=(0, None), prior=LogNormal(1, 1)),
        dict(name='Ci', scale=1e6 / sT, bounds=(0, None), prior=LogNormal(1, 1)),
        dict(name='sigw_w', scale=1e-2 * sT ** 0.5, bounds=(0, None), prior=LogNormal(1, 1)),
        dict(name='sigw_i', value=0, transform='fixed'),
        dict(name='sigv', scale=1e-2, bounds=(0, None), prior=LogNormal(1, 1)),
        dict(name='x0_w', loc=25, scale=5, prior=Normal(0, 1)),
        dict(name='x0_i', value=26.701, transform='fixed'),
        dict(name='sigx0_w', value=1, transform='fixed'),
        dict(name='sigx0_i', value=0.1, transform='fixed'),
    ]


@pytest.fixture
def statespace_armadillo(parameters_armadillo):
    return TwTi_RoRi(parameters_armadillo, hold_order=1)


@pytest.fixture
def regressor_armadillo(statespace_armadillo):
    return Regressor(ss=statespace_armadillo)


def test_fit_predict(data_armadillo, regressor_armadillo):
    sT = 3600.0 * 24.0
    data_armadillo.index /= sT

    summary, corr, scipy_summary = regressor_armadillo.fit(
        df=data_armadillo, outputs='T_int', inputs=['T_ext', 'P_hea']
    )

    loglik = regressor_armadillo.eval_log_likelihood(
        df=data_armadillo, outputs='T_int', inputs=['T_ext', 'P_hea']
    )

    ym, _ = regressor_armadillo.predict(df=data_armadillo, inputs=['T_ext', 'P_hea'])
    y = data_armadillo['T_int'].values

    res, _ = regressor_armadillo.eval_residuals(
        df=data_armadillo, inputs=['T_ext', 'P_hea'], outputs='T_int'
    )

    assert scipy_summary.fun == pytest.approx(-316.68812014562525, rel=1e-3)
    assert loglik == pytest.approx(-328.97507135305074, rel=1e-3)
    Np = int(np.sum(regressor_armadillo.ss.parameters.free))
    assert aic(loglik, Np) == pytest.approx(-643.9501427061015, rel=1e-3)
    assert lrtest(loglik - 2, loglik, Np + 1, Np) == pytest.approx(0.04550026389635857, rel=1e-3)
    assert fit(y, ym) == pytest.approx(0.8430719684902173, rel=1e-3)
    assert rmse(y, ym) == pytest.approx(0.7434264911761783, rel=1e-3)
    assert mae(y, ym) == pytest.approx(0.6578653810019472, rel=1e-3)
    assert mad(y, ym) == pytest.approx(1.3561205016928923, rel=1e-3)
    assert smape(y, ym) == pytest.approx(1.0027822035683103, rel=1e-3)
    assert ned(y, ym) == pytest.approx(0.011151665679293845, rel=1e-3)
    assert check_model(regressor_armadillo, summary, corr, verbose=False)
    assert res.mean() == pytest.approx(0, abs=5e-2)
    assert check_ccf(*ccf(res))[0]
    assert check_cpgram(*cpgram(res))[0]
