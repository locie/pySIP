import pytest

import pandas as pd


def test_fit_api_list(data, regressor):
    regressor.fit(data, inputs=['T_ext', 'P_hea'], outputs=['T_in'])


@pytest.mark.xfail
def test_fit_api_automatic(data, regressor):
    regressor.fit(data)


@pytest.mark.xfail
def test_predict_api_df(test, data, regressor):
    regressor.fit(data, inputs=['T_ext', 'P_hea'], outputs=['T_in'])
    preds = regressor.predict(test, inputs=['T_ext', 'P_hea'])

    assert isinstance(preds, pd.DataFrame)
    assert (preds.index == test.index).all()
    assert len(preds.index) == 3
    assert list(preds.columns) == ['T_in__lower', 'T_in', 'T_in__upper']


@pytest.mark.xfail
def test_predict_api_df_2(test2, data, regressor):
    regressor.fit(data, inputs=['T_ext', 'P_hea'], outputs=['T_in'])
    preds = regressor.predict(test2, inputs=['T_ext', 'P_hea'])

    assert isinstance(preds, pd.DataFrame)
    assert (preds.index == test2.index).all()
    assert len(preds.index) == 3
    assert list(preds.columns) == ['T_in__lower', 'T_in', 'T_in__upper']


@pytest.mark.xfail
def test_predict_api_df_3(test3, data, regressor):
    regressor.fit(data, inputs=['T_ext', 'P_hea'], outputs=['T_in'])
    preds = regressor.predict(test3, inputs=['T_ext', 'P_hea'])

    assert isinstance(preds, pd.DataFrame)
    assert (preds.index == test3.index).all()
    assert len(preds.index) == 3
    assert list(preds.columns) == ['T_in__lower', 'T_in', 'T_in__upper']
    assert preds['T_in'][0] == pytest.approx(50.0, 0.1)


@pytest.mark.xfail
def test_predict_api_df_and_t(regressor, data):

    from datetime import timedelta
    idx = pd.date_range('2019-01-01', periods=54, freq='1h')[-6:]
    t = (idx - timedelta(minutes=20))[1:]
    t_ext = [14.0 for t, x in enumerate(idx)]
    p_hea = [0 for x in idx]
    predictions = pd.DataFrame(index=idx, data={'T_ext': t_ext, 'P_hea': p_hea})

    regressor.fit(data, inputs=['T_ext', 'P_hea'], outputs=['T_in'])
    preds = regressor.predict(predictions, tpred=t)

    assert isinstance(preds, pd.DataFrame)
    assert (preds.index == t).all()
    assert len(preds.index) == 5
    assert list(preds.columns) == ['T_in__lower', 'T_in', 'T_in__upper']
