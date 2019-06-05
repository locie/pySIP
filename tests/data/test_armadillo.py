import pytest

import numpy as np
import pandas as pd
from bopt.statespace.rc import TwTi_RoRi
from bopt.regressors import Regressor
from bopt.core import Normal


def test_fit_predict(data, regressor):
    regressor.fit(df=data, outputs='T_int', inputs=['T_ext', 'P_hea'])
    y_mean, _ = regressor.predict(df=data, inputs=['T_ext', 'P_hea'])

    assert regressor.log_posterior == pytest.approx(-252.707, rel=1e-3)
    assert regressor.log_likelihood == pytest.approx(-244.283, rel=1e-3)

    y = data['T_int'].values
    fit_mean = 1.0 - np.linalg.norm(y - y_mean, 2) / np.linalg.norm(y - np.mean(y), 2)

    assert fit_mean == pytest.approx(0.807, rel=1e-2)
