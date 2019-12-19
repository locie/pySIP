import os
import pytest
import numpy as np

from pysip.regressors import FreqRegressor as Regressor
from pysip.statespace import TwTi_RoRiAwAicv, TwTiTm_RoRiRmRbAwAicv, Matern12, Matern52, Periodic
from pysip.utils import save_model, load_model


@pytest.mark.parametrize(
    'reg',
    [
        pytest.param(TwTi_RoRiAwAicv(), id="rc_model"),
        pytest.param(Matern12(), id="gp_model"),
        pytest.param(Periodic() * Matern12(), id="gp_product"),
        pytest.param(Periodic() + Matern12(), id="gp_sum"),
        pytest.param(TwTi_RoRiAwAicv(latent_forces='Qv') <= Matern12(), id="lfm_rc_gp"),
        pytest.param(
            TwTi_RoRiAwAicv(latent_forces='Qv') <= Periodic() * Matern12(), id="lfm_rc_gp_product"
        ),
        pytest.param(
            TwTi_RoRiAwAicv(latent_forces='Qv') <= Periodic() + Matern12(), id="lfm_rc_gp_sum"
        ),
    ],
)
def test_save_model_to_pickle(reg):
    reg = Regressor(reg)
    reg.ss.parameters.theta = np.random.uniform(1e-1, 1, len(reg.ss.parameters.theta))
    reg.ss.update_continuous_dssm()
    dA = reg.ss.dA
    dB = reg.ss.dB
    dC = reg.ss.dC
    dD = reg.ss.dD
    dQ = reg.ss.dQ
    dR = reg.ss.dR
    dx0 = reg.ss.dx0
    dP0 = reg.ss.dP0

    save_model('test', reg)
    load_reg = load_model('test')

    for k in dA.keys():
        assert np.allclose(dA[k], load_reg.ss.dA[k])
        assert np.allclose(dB[k], load_reg.ss.dB[k])
        assert np.allclose(dC[k], load_reg.ss.dC[k])
        assert np.allclose(dD[k], load_reg.ss.dD[k])
        assert np.allclose(dQ[k], load_reg.ss.dQ[k])
        assert np.allclose(dR[k], load_reg.ss.dR[k])
        assert np.allclose(dx0[k], load_reg.ss.dx0[k])
        assert np.allclose(dP0[k], load_reg.ss.dP0[k])

    assert id(reg) != id(load_reg)
    for a, b in zip(reg.ss.parameters, load_reg.ss.parameters):
        assert a == b

    os.remove('test.pickle')


def test_save_version_number():
    from pysip import __version__

    save_model('test', Regressor(None))
    load_reg = load_model('test')

    assert load_reg.__version__ == __version__


@pytest.mark.parametrize(
    'ss,size_kb',
    [
        pytest.param(TwTi_RoRiAwAicv(), 4, id="rc_model"),
        pytest.param(Matern12(), 17, id="gp_model"),
        pytest.param(Periodic() * Matern12(), 32, id="gp_product"),
        pytest.param(
            TwTiTm_RoRiRmRbAwAicv(latent_forces='Qv') <= Periodic() * Matern52() + Matern52(),
            237,
            id="lfm_model",
        ),
    ],
)
def test_model_pickle_file_size_limit(ss, size_kb):
    model = Regressor(ss)
    model.ss.parameters.theta = np.random.uniform(1e-1, 1, len(model.ss.parameters.theta))
    model.ss.update_continuous_dssm()

    save_model('big', model)
    size = os.path.getsize('big.pickle')
    os.remove('big.pickle')

    assert size <= size_kb * 1000
