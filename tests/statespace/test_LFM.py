from copy import deepcopy

import numdifftools as nd
import numpy as np
import pandas as pd
import pytest

from pysip.regressors import FreqRegressor as Regressor
from pysip.statespace import LatentForceModel, Matern32, R2C2_Qgh_Matern32, R2C2Qgh
from pysip.utils import generate_random_binary, generate_sine, ned

sT = 24.0 * 60.0
sC = 1e8 / sT


@pytest.fixture
def artificial_data():
    """Generate 50 data points spaced of 1 hour

    Ti: Indoor temperature
    To: Outdoor temperature
    Tb: Boundary temperature
    Qgh: Global horizontal solar radiation
    Qh: Heat supplied by heaters
    Qv: Heat supplied by ventilation
    """
    n = 50  # number of points
    df = pd.DataFrame(
        {
            'Ti': generate_sine(n=n, amplitude=3.0, phase=np.pi / 3.0, offset=20.0)[1],
            'To': generate_sine(n=n)[1],
            'Tb': generate_sine(n=n, amplitude=2.0, phase=np.pi / 4.0, offset=14.0)[1],
            'Qgh': generate_sine(n=n, amplitude=800.0, clip_to_0=True)[1],
            'Qh': generate_random_binary(n=n),
            'Qv': generate_sine(n=n, amplitude=200.0, offset=100.0)[1],
        }
    )
    df.index /= sT
    return df


@pytest.fixture
def models():

    p = np.random.uniform(-1, 1, 13)

    p_truth = [
        dict(name='R2C2Qgh__Ro', scale=1e-2, transform='log'),
        dict(name='R2C2Qgh__Ri', scale=1e-3, transform='log'),
        dict(name='R2C2Qgh__Cw', scale=sC, transform='log'),
        dict(name='R2C2Qgh__Ci', scale=sC, transform='log'),
        dict(name='R2C2Qgh__sigw_w', scale=np.sqrt(sT), transform='log'),
        dict(name='R2C2Qgh__sigw_i', scale=np.sqrt(sT), transform='log'),
        dict(name='R2C2Qgh__sigv', scale=1e-2, transform='log'),
        dict(name='R2C2Qgh__x0_w', scale=1e1, transform='log'),
        dict(name='R2C2Qgh__x0_i', scale=1e1, transform='log'),
        dict(name='R2C2Qgh__sigx0_w', scale=1.0, transform='log'),
        dict(name='R2C2Qgh__sigx0_i', scale=1.0, transform='log'),
        dict(name='Matern32__mscale', scale=1e3, transform='log'),
        dict(name='Matern32__lscale', scale=1.0, transform='log'),
    ]

    p_rc = [
        dict(name='Ro', scale=1e-2, transform='log'),
        dict(name='Ri', scale=1e-3, transform='log'),
        dict(name='Cw', scale=sC, transform='log'),
        dict(name='Ci', scale=sC, transform='log'),
        dict(name='sigw_w', scale=np.sqrt(sT), transform='log'),
        dict(name='sigw_i', scale=np.sqrt(sT), transform='log'),
        dict(name='sigv', scale=1e-2, transform='log'),
        dict(name='x0_w', scale=1e1, transform='log'),
        dict(name='x0_i', scale=1e1, transform='log'),
        dict(name='sigx0_w', scale=1.0, transform='log'),
        dict(name='sigx0_i', scale=1.0, transform='log'),
    ]

    p_gp = [
        dict(name='mscale', scale=1e3, transform='log'),
        dict(name='lscale', scale=1.0, transform='log'),
        dict(name='sigv', value=0.0, transform='fixed'),
    ]

    truth = R2C2_Qgh_Matern32(p_truth)
    truth.parameters.eta = p
    truth.update_continuous_ssm()
    truth.update_continuous_dssm()

    lfm = LatentForceModel(R2C2Qgh(p_rc), Matern32(p_gp), 'Qgh')
    lfm.parameters.eta = p
    lfm.update_continuous_ssm()
    lfm.update_continuous_dssm()

    return truth, lfm


@pytest.fixture
def models_by_operator():
    truth = R2C2_Qgh_Matern32()
    theta = np.random.random(len(truth._names))

    truth.parameters.theta_free = theta
    truth.update_continuous_ssm()
    truth.update_continuous_dssm()

    lfm = R2C2Qgh(latent_forces='Qgh') <= Matern32()
    lfm.parameters.theta_free = theta
    lfm.update_continuous_ssm()
    lfm.update_continuous_dssm()
    return truth, lfm


def test_LFM_creation_by_function(models):
    LFM_creation(models)


def test_LFM_creation_by_operator(models_by_operator):
    LFM_creation(models_by_operator)


def LFM_creation(models):
    truth, lfm = models
    assert truth.states == lfm.states
    assert truth.inputs == lfm.inputs
    assert truth.outputs == lfm.outputs
    assert truth.parameters.theta_free == lfm.parameters.theta_free

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


def test_LFM_gradient(artificial_data, models):
    reg_truth = Regressor(ss=models[0])
    reg_truth._use_penalty = False
    reg_truth._use_jacobian = True
    dt, u, u1, y, *_ = reg_truth._prepare_data(artificial_data, ['To', 'Qh'], 'Ti')

    reg_lfm = Regressor(ss=models[1])
    reg_lfm._use_penalty = False
    reg_lfm._use_jacobian = True

    eta_truth = deepcopy(reg_truth.ss.parameters.eta_free)
    eta_lfm = deepcopy(reg_lfm.ss.parameters.eta_free)

    grad_truth = reg_truth._eval_dlog_posterior(eta_truth, dt, u, u1, y)[1]
    grad_lfm = reg_lfm._eval_dlog_posterior(eta_lfm, dt, u, u1, y)[1]

    fct = nd.Gradient(reg_truth._eval_log_posterior)
    grad_truth_approx = fct(eta_truth, dt, u, u1, y)

    assert np.all(eta_truth == eta_lfm)
    assert ned(grad_truth, grad_lfm) < 1e-7
    assert ned(grad_truth, grad_truth_approx) < 1e-7
    assert np.all(np.sign(grad_truth) == np.sign(grad_truth_approx))
    assert np.all(np.sign(grad_truth) == np.sign(grad_lfm))
    assert grad_truth == pytest.approx(grad_truth_approx, rel=1e-6)
    assert grad_truth == pytest.approx(grad_lfm, rel=1e-6)
