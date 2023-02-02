""" To test with Gaussian Processes and Latent Force models (integrator) and time it !"""

import numpy as np
from pysip.statespace import (
    TwTi_RoRiAwAicv,
    TwTiTm_RoRiRmAwAicv,
    TwTiTh_RoRiRhAwAicv,
    Periodic,
    Matern32,
    QuasiPeriodic12,
    QuasiPeriodic32,
)
from pysip.statespace.discretization import *
from time import time
import pytest

sT = 3600.0 * 24.0

par_rc1 = [
    dict(name='Ro', scale=1e-2, bounds=(1e-4, None)),
    dict(name='Ri', scale=1e-3, bounds=(1e-4, None)),
    dict(name='Cw', scale=1e7 / sT, bounds=(1e-4, None)),
    dict(name='Ci', scale=1e6 / sT, bounds=(1e-4, None)),
    dict(name='Aw', scale=1e0, bounds=(1e-4, None)),
    dict(name='Ai', scale=1e0, bounds=(1e-4, None)),
    dict(name='cv', value=5e2, transform='fixed'),
    dict(name='sigw_w', scale=1e-3 * sT ** 0.5, bounds=(1e-4, None)),
    dict(name='sigw_i', scale=1e-3 * sT ** 0.5, bounds=(1e-4, None)),
    dict(name='sigv', scale=1e-2, bounds=(1e-4, None)),
    dict(name='x0_w', loc=15.0, scale=7.0, transform='none'),
    dict(name='x0_i', loc=20.0, scale=7.0, transform='none'),
    dict(name='sigx0_w', value=0.1, transform='fixed'),
    dict(name='sigx0_i', value=0.1, transform='fixed'),
]

par_rc2 = [
    dict(name='Ro', scale=1e-2, bounds=(1e-4, None)),
    dict(name='Ri', scale=1e-3, bounds=(1e-4, None)),
    dict(name='Rm', scale=1e-3, bounds=(1e-4, None)),
    dict(name='Cw', scale=1e7 / sT, bounds=(1e-4, None)),
    dict(name='Ci', scale=1e6 / sT, bounds=(1e-4, None)),
    dict(name='Cm', scale=1e7 / sT, bounds=(1e-4, None)),
    dict(name='Aw', scale=1e0, bounds=(1e-4, None)),
    dict(name='Ai', scale=1e0, bounds=(1e-4, None)),
    dict(name='cv', value=5e2, transform='fixed'),
    dict(name='sigw_w', scale=1e-3 * sT ** 0.5, bounds=(1e-4, None)),
    dict(name='sigw_i', scale=1e-3 * sT ** 0.5, bounds=(1e-4, None)),
    dict(name='sigw_m', scale=1e-3 * sT ** 0.5, bounds=(1e-4, None)),
    dict(name='sigv', scale=1e-2, bounds=(1e-4, None)),
    dict(name='x0_w', loc=15.0, scale=7.0, transform='none'),
    dict(name='x0_i', loc=20.0, scale=7.0, transform='none'),
    dict(name='x0_m', loc=20.0, scale=7.0, transform='none'),
    dict(name='sigx0_w', value=0.1, transform='fixed'),
    dict(name='sigx0_i', value=0.1, transform='fixed'),
    dict(name='sigx0_m', value=0.1, transform='fixed'),
]

par_rc3 = [
    dict(name='Ro', scale=1e-2, bounds=(1e-4, None)),
    dict(name='Ri', scale=1e-3, bounds=(1e-4, None)),
    dict(name='Rh', scale=1e-3, bounds=(1e-4, None)),
    dict(name='Cw', scale=1e7 / sT, bounds=(1e-4, None)),
    dict(name='Ci', scale=1e6 / sT, bounds=(1e-4, None)),
    dict(name='Ch', scale=1e6 / sT, bounds=(1e-4, None)),
    dict(name='Aw', scale=1e0, bounds=(1e-4, None)),
    dict(name='Ai', scale=1e0, bounds=(1e-4, None)),
    dict(name='cv', value=5e2, transform='fixed'),
    dict(name='sigw_w', scale=1e-3 * sT ** 0.5, bounds=(1e-4, None)),
    dict(name='sigw_i', scale=1e-3 * sT ** 0.5, bounds=(1e-4, None)),
    dict(name='sigw_h', scale=1e-3 * sT ** 0.5, bounds=(1e-4, None)),
    dict(name='sigv', scale=1e-2, bounds=(1e-4, None)),
    dict(name='x0_w', loc=15.0, scale=7.0, transform='none'),
    dict(name='x0_i', loc=20.0, scale=7.0, transform='none'),
    dict(name='x0_h', loc=20.0, scale=7.0, transform='none'),
    dict(name='sigx0_w', value=0.1, transform='fixed'),
    dict(name='sigx0_i', value=0.1, transform='fixed'),
    dict(name='sigx0_h', value=0.1, transform='fixed'),
]

par_p = [
    dict(name='period', bounds=(1e-4, None)),
    dict(name='mscale', bounds=(1e-4, None)),
    dict(name='lscale', bounds=(0.05, None)),
    dict(name='sigv', bounds=(1e-4, None)),
]

par_m = [
    dict(name='mscale', bounds=(1e-4, None)),
    dict(name='lscale', bounds=(1e-4, None)),
    dict(name='sigv', bounds=(1e-4, None)),
]

par_qp = [
    dict(name='period', bounds=(1e-4, None)),
    dict(name='mscale', bounds=(1e-4, None)),
    dict(name='lscale', bounds=(0.05, None)),
    dict(name='sigv', bounds=(1e-4, None)),
    dict(name='decay', bounds=(1e-4, None)),
]

model_list = [
    TwTi_RoRiAwAicv(par_rc1),
    # TwTi_RoRiAwAicv(par_rc1, latent_forces='Qv') <= Matern32(parameters=par_m),
    # TwTi_RoRiAwAicv(par_rc1, latent_forces='Qv') <= Periodic(parameters=par_p),
    # TwTi_RoRiAwAicv(par_rc1, latent_forces='Qv') <= QuasiPeriodic32(parameters=par_qp),
    TwTiTm_RoRiRmAwAicv(par_rc2),
    # TwTiTm_RoRiRmAwAicv(par_rc2, latent_forces='Qv') <= Matern32(parameters=par_m),
    # TwTiTm_RoRiRmAwAicv(par_rc2, latent_forces='Qv') <= Periodic(parameters=par_p),
    # TwTiTm_RoRiRmAwAicv(par_rc2, latent_forces='Qv') <= QuasiPeriodic32(parameters=par_qp),
    TwTiTh_RoRiRhAwAicv(par_rc3),
    # TwTiTh_RoRiRhAwAicv(par_rc3, latent_forces='Qv') <= Matern32(parameters=par_m),
    # TwTiTh_RoRiRhAwAicv(par_rc3, latent_forces='Qv') <= Periodic(parameters=par_p),
    # TwTiTh_RoRiRhAwAicv(par_rc3, latent_forces='Qv') <= QuasiPeriodic32(parameters=par_qp),
]


@pytest.fixture
def random_dt():
    dt_list = [1.0 / 24.0, 1.0 / (2.0 * 24.0), 1.0 / (4.0 * 24.0), 1.0 / (6.0 * 24.0)]
    return np.random.choice(dt_list)


@pytest.mark.parametrize('model', model_list)
@pytest.mark.parametrize('order_hold', [0, 1])
def test_disc_state_input(model, order_hold, random_dt):
    dt = random_dt
    model.parameters.eta = np.random.uniform(-1, 1, model.parameters.n_par)
    model.update_continuous_ssm()
    model.update_continuous_dssm()
    dA = np.array([model.dA[n] for n, f in zip(model._names, model.parameters.free) if f])
    dB = np.array([model.dB[n] for n, f in zip(model._names, model.parameters.free) if f])

    tic = time()
    Ad_expm, B0d_expm, B1d_expm = disc_state_input(model.A, model.B, dt, order_hold, 'expm')
    toc = time() - tic
    print(f'expm: {toc:.4e}')

    tic = time()
    Ad_expm_bis, B0d_expm_bis, B1d_expm_bis, dAd_expm, dB0d_expm, dB1d_expm = disc_d_state_input(
        model.A, model.B, dA, dB, dt, order_hold, 'expm'
    )
    toc = time() - tic
    print(f'dexpm: {toc:.4e}')

    tic = time()
    Ad, B0d, B1d = disc_state_input(model.A, model.B, dt, order_hold, 'analytic')
    toc = time() - tic
    print(f'analytic: {toc:.4e}')

    tic = time()
    Ad_bis, B0d_bis, B1d_bis, dAd, dB0d, dB1d = disc_d_state_input(
        model.A, model.B, dA, dB, dt, order_hold, 'analytic'
    )
    toc = time() - tic
    print(f'danalytic: {toc:.4e}')

    assert np.allclose(Ad_expm, Ad_expm_bis)
    assert np.allclose(B0d_expm, B0d_expm_bis)
    assert np.allclose(B1d_expm, B1d_expm_bis)

    assert np.allclose(Ad_expm, Ad)
    assert np.allclose(B0d_expm, B0d)
    assert np.allclose(B1d_expm, B1d)

    assert np.allclose(dAd_expm, dAd)
    assert np.allclose(dB0d_expm, dB0d)
    assert np.allclose(dB1d_expm, dB1d)
