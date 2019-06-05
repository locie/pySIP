import pytest
import pandas as pd

from bopt.statespace.rc import TwTi_RoRi
from bopt.regressors import Regressor
from bopt.core import Normal


@pytest.fixture
def data():
    return pd.read_csv('data/armadillo/armadillo_data_H2.csv').set_index('Time')


@pytest.fixture
def parameters():
    YILOC = 26.70106194217502
    return [
        dict(name="Ro", value=0.1, transform="lowup", bounds=(1e-6, 0.3), prior=Normal(0.1, 0.04)),
        dict(name="Ri", value=0.01, transform="lowup", bounds=(1e-6, 0.3), prior=Normal(0.05, 0.02)),
        dict(name="Cw", value=0.1, transform="lowup", bounds=(1e-6, 0.3), prior=Normal(0.1, 0.04)),
        dict(name="Ci", value=0.01, transform="lowup", bounds=(1e-6, 0.3), prior=Normal(0.05, 0.02)),
        dict(name="sigw_w", value=0.01, transform="lowup", bounds=(1e-6, 0.3), prior=Normal(0.05, 0.02)),
        dict(name="sigw_i", value=0.0, transform="fixed"),
        dict(name="sigv", value=0.01, transform="lowup", bounds=(1e-6, 0.3), prior=Normal(0.05, 0.02)),
        dict(name="x0_w", value=0.25, transform="lowup", bounds=(0.1, 0.5), prior=Normal(0.25, 0.07)),
        dict(name="x0_i", value=(YILOC / 100.0), transform="fixed"),
        dict(name="sigx0_w", value=1.0, transform="fixed"),
        dict(name="sigx0_i", value=1.0, transform="fixed")
    ]


@pytest.fixture
def statespace(parameters):
    return TwTi_RoRi(parameters, hold_order='foh')


@pytest.fixture
def regressor(statespace):
    return Regressor(ss=statespace)
