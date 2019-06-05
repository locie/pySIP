import pytest

import pandas as pd

from bopt.core import Normal
from bopt.statespace.rc import TwTi_RoRi
from bopt.regressors import Regressor
from bopt.regressors.meta import MetaRegressor


@pytest.fixture
def rooms():
    return ['kitchen', 'bedroom']


@pytest.fixture
def metadata(data, rooms):
    df = pd.concat([
        data.rename(columns={x: x + f'__{room}' for x in data.columns[1:]})
        for room in rooms
    ], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


@pytest.mark.xfail
def test_metaregressor(metadata, statespace, rooms):
    reg = MetaRegressor(rooms=rooms, conjoint=['T_ext'], ss=statespace,
                        reg=Regressor)

    reg.fit(metadata, inputs=['P_hea', 'T_ext'], outputs=['T_int'])

    assert len(reg.regressors) == len(rooms)
    for r in reg.regressors.values():
        assert r.log_posterior == pytest.approx(-252.707, rel=1e-3)
        assert r.log_likelihood == pytest.approx(-244.283, rel=1e-3)
