import numpy as np
import pytest

from bopt.statespace.rc.r2c2_qgh import R2C2Qgh
from bopt.statespace.gaussian_process.matern import Matern32
from bopt.statespace.latent_force.latent import LatentForceModel, R2C2_Qgh_Matern32


@pytest.fixture
def models():
    truth = R2C2_Qgh_Matern32()
    theta = np.random.random(len(truth._names))
    truth.parameters.theta = theta
    truth.update()
    lfm = LatentForceModel(R2C2Qgh(), Matern32(), 'Qgh')
    lfm.parameters.theta = theta
    lfm.update()
    return truth, lfm


def test_LFM_creation(models):
    truth, lfm = models
    assert truth.states == lfm.states
    assert truth.inputs == lfm.inputs
    assert truth.outputs == lfm.outputs
    assert truth.params == lfm.params

    assert np.allclose(truth.A, lfm.A)
    assert np.allclose(truth.B, lfm.B)
    assert np.allclose(truth.C, lfm.C)
    assert np.allclose(truth.D, lfm.D)
    assert np.allclose(truth.Q, lfm.Q)
    assert np.allclose(truth.R, lfm.R)
    assert np.allclose(truth.x0, lfm.x0)
    assert np.allclose(truth.P0, lfm.P0)
    for k in truth._names:
        assert np.allclose(truth.dA[k], lfm.dA[k])
        assert np.allclose(truth.dB[k], lfm.dB[k])
        assert np.allclose(truth.dC[k], lfm.dC[k])
        assert np.allclose(truth.dD[k], lfm.dD[k])
        assert np.allclose(truth.dQ[k], lfm.dQ[k])
        assert np.allclose(truth.dR[k], lfm.dR[k])
        assert np.allclose(truth.dx0[k], lfm.dx0[k])
        assert np.allclose(truth.dP0[k], lfm.dP0[k])
