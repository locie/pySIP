import pytest

import numpy as np
import pandas as pd

from bopt.statespace import TwTi_RoRi
from bopt.regressors import Regressor


@pytest.fixture
def data():
    idx = pd.date_range('2019-01-01', periods=48, freq='1h')

    return pd.DataFrame(
        index=idx,
        data={
            'T_ext': [12.0 + 10.0 * np.sin(2.0 * np.pi / 24.0 * float(t)) for t, x in enumerate(idx)],
            'P_hea': [abs(200 * np.random.random()) for x in idx],
            'T_in': [19.0 + np.random.random() + 2 * np.sin(2.0 * np.pi / 24.0 * float(t)) for t, x in enumerate(idx)]
        }
    )


@pytest.fixture
def test():
    idx = pd.date_range('2019-01-01', periods=51, freq='1h')[-3:]
    t_ext = [12.0 + 10.0 * np.sin(2.0 * np.pi / 24.0 * float(t)) for t, x in enumerate(idx)]
    p_hea = [abs(200 * np.random.random()) for x in idx]

    data = pd.DataFrame(index=idx, data={'T_ext': t_ext, 'P_hea': p_hea})

    return data


@pytest.fixture
def test2():
    idx = pd.date_range('2019-01-01', periods=51, freq='1h')[-3:]
    t_ext = [12.0 + 10.0 * np.sin(2.0 * np.pi / 24.0 * float(t)) for t, x in enumerate(idx)]
    p_hea = [abs(200 * np.random.random()) for x in idx]
    t_in = [np.nan for x in idx]

    data = pd.DataFrame(index=idx, data={'T_ext': t_ext, 'P_hea': p_hea, 'T_in': t_in})

    return data


@pytest.fixture
def test3():
    idx = pd.date_range('2019-01-01', periods=51, freq='1h')[-3:]
    t_ext = [12.0 + 10.0 * np.sin(2.0 * np.pi / 24.0 * float(t)) for t, x in enumerate(idx)]
    p_hea = [abs(200 * np.random.random()) for x in idx]
    t_in = [50] + [np.nan for x in idx[1:]]

    data = pd.DataFrame(index=idx, data={'T_ext': t_ext, 'P_hea': p_hea, 'T_in': t_in})

    return data


@pytest.fixture
def data2():
    idx = pd.date_range('2019-01-01', periods=48, freq='1h')
    t_ext = [12.0 + 10.0 * np.sin(2.0 * np.pi / 24.0 * float(t)) for t, x in enumerate(idx)]
    p_hea = [abs(0 * np.random.random()) for x in idx]

    t_in = [19.0 + np.random.random() + 2 * np.sin(2.0 * np.pi / 24.0 * float(t)) for t, x in enumerate(idx)]

    data = pd.DataFrame(index=idx, data={'Temp_ext': t_ext, 'Power_hea': p_hea, 'Temp_in': t_in})

    return data


@pytest.fixture
def parameters():
    return [
        dict(name="Ro", value=0.1, transform="log"),
        dict(name="Ri", value=0.01, transform="log"),
        dict(name="Cw", value=0.1, transform="log"),
        dict(name="Ci", value=0.01, transform="log"),
        dict(name="sigw_w", value=0.01, transform="fixed"),
        dict(name="sigw_i", value=0.0, transform="fixed"),
        dict(name="sigv", value=0.01, transform="fixed"),
        dict(name="x0_w", value=0.25, transform="fixed"),
        dict(name="x0_i", value=(26 / 100.0), transform="fixed"),
        dict(name="sigx0_w", value=1.0, transform="fixed"),
        dict(name="sigx0_i", value=1.0, transform="fixed")
    ]


@pytest.fixture
def regressor(parameters):
    return Regressor(TwTi_RoRi(parameters))
