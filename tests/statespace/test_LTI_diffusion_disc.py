from time import time

import numpy as np
import pandas as pd
import pytest

from pysip.statespace import GPModel
from pysip.statespace.discretization import *
from pysip.statespace.gaussian_process import *
from pysip.statespace.thermal_network import TwTi_RoRiAwAicv, TwTiTm_RoRiRmAwAicv

sT = 3600.0 * 24.0

par_rc2 = [
    dict(name="Ro", scale=1e-2, bounds=(1e-4, None)),
    dict(name="Ri", scale=1e-3, bounds=(1e-4, None)),
    dict(name="Cw", scale=1e7 / sT, bounds=(1e-4, None)),
    dict(name="Ci", scale=1e6 / sT, bounds=(1e-4, None)),
    dict(name="Aw", scale=1e0, bounds=(1e-4, None)),
    dict(name="Ai", scale=1e0, bounds=(1e-4, None)),
    dict(name="cv", value=5e2, transform="fixed"),
    dict(name="sigw_w", scale=1e-3 * sT**0.5, bounds=(1e-4, None)),
    dict(name="sigw_i", scale=1e-3 * sT**0.5, bounds=(1e-4, None)),
    dict(name="sigv", scale=1e-2, bounds=(1e-4, None)),
    dict(name="x0_w", loc=15.0, scale=7.0, transform="none"),
    dict(name="x0_i", loc=20.0, scale=7.0, transform="none"),
    dict(name="sigx0_w", value=0.1, transform="fixed"),
    dict(name="sigx0_i", value=0.1, transform="fixed"),
]

par_rc3 = [
    dict(name="Ro", scale=1e-2, bounds=(1e-4, None)),
    dict(name="Ri", scale=1e-3, bounds=(1e-4, None)),
    dict(name="Rm", scale=1e-3, bounds=(1e-4, None)),
    dict(name="Cw", scale=1e7 / sT, bounds=(1e-4, None)),
    dict(name="Ci", scale=1e6 / sT, bounds=(1e-4, None)),
    dict(name="Cm", scale=1e7 / sT, bounds=(1e-4, None)),
    dict(name="Aw", scale=1e0, bounds=(1e-4, None)),
    dict(name="Ai", scale=1e0, bounds=(1e-4, None)),
    dict(name="cv", value=5e2, transform="fixed"),
    dict(name="sigw_w", scale=1e-3 * sT**0.5, bounds=(1e-4, None)),
    dict(name="sigw_i", scale=1e-3 * sT**0.5, bounds=(1e-4, None)),
    dict(name="sigw_m", scale=1e-3 * sT**0.5, bounds=(1e-4, None)),
    dict(name="sigv", scale=1e-2, bounds=(1e-4, None)),
    dict(name="x0_w", loc=15.0, scale=7.0, transform="none"),
    dict(name="x0_i", loc=20.0, scale=7.0, transform="none"),
    dict(name="x0_m", loc=20.0, scale=7.0, transform="none"),
    dict(name="sigx0_w", value=0.1, transform="fixed"),
    dict(name="sigx0_i", value=0.1, transform="fixed"),
    dict(name="sigx0_m", value=0.1, transform="fixed"),
]

par_p = [
    dict(name="period", bounds=(1e-4, None)),
    dict(name="mscale", bounds=(1e-4, None)),
    dict(name="lscale", bounds=(0.05, None)),
    dict(name="sigv", bounds=(1e-4, None)),
]

par_m = [
    dict(name="mscale", bounds=(1e-4, None)),
    dict(name="lscale", bounds=(1e-4, None)),
    dict(name="sigv", bounds=(1e-4, None)),
]

par_qp = [
    dict(name="period", bounds=(1e-4, None)),
    dict(name="mscale", bounds=(1e-4, None)),
    dict(name="lscale", bounds=(0.05, None)),
    dict(name="sigv", bounds=(1e-4, None)),
    dict(name="decay", bounds=(1e-4, None)),
]

model_list = [
    TwTi_RoRiAwAicv(par_rc2),
    TwTiTm_RoRiRmAwAicv(par_rc3),
    Matern12(parameters=par_m),
    Matern32(parameters=par_m),
    Matern52(parameters=par_m),
    Periodic(parameters=par_p) + Matern12(parameters=par_m),
    Periodic(parameters=par_p) + Matern32(parameters=par_m),
    Periodic(parameters=par_p) + Matern52(parameters=par_m),
    QuasiPeriodic12(parameters=par_qp),
    QuasiPeriodic32(parameters=par_qp),
    Periodic(parameters=par_p) * Matern52(parameters=par_m),
    QuasiPeriodic12(parameters=par_qp) + Matern12(parameters=par_m),
    QuasiPeriodic32(parameters=par_qp) + Matern12(parameters=par_m),
    QuasiPeriodic32(parameters=par_qp) + Matern32(parameters=par_m),
    QuasiPeriodic32(parameters=par_qp) + Matern32(parameters=par_m),
    QuasiPeriodic12(parameters=par_qp) + Matern52(parameters=par_m),
    QuasiPeriodic32(parameters=par_qp) + Matern52(parameters=par_m),
    TwTi_RoRiAwAicv(par_rc2, latent_forces="Qv") <= Matern32(parameters=par_m),
    TwTi_RoRiAwAicv(par_rc2, latent_forces="Qv") <= Periodic(parameters=par_p),
    TwTi_RoRiAwAicv(par_rc2, latent_forces="Qv") <= QuasiPeriodic32(parameters=par_qp),
    TwTiTm_RoRiRmAwAicv(par_rc3, latent_forces="Qv") <= Matern32(parameters=par_m),
    TwTiTm_RoRiRmAwAicv(par_rc3, latent_forces="Qv") <= Periodic(parameters=par_p),
    TwTiTm_RoRiRmAwAicv(par_rc3, latent_forces="Qv")
    <= QuasiPeriodic32(parameters=par_qp),
]


@pytest.fixture
def random_dt():
    dt_list = [1.0 / 24.0, 1.0 / (2.0 * 24.0), 1.0 / (4.0 * 24.0), 1.0 / (6.0 * 24.0)]
    return np.random.choice(dt_list)


@pytest.mark.parametrize("model", model_list)
def test_disc_LTI(model, random_dt):
    dt = random_dt

    model.parameters.eta = np.random.uniform(-1, 1, model.parameters.n_par)
    ssm, dssm, _ = model.get_discrete_dssm(dt)

    dA = np.array(
        [model.dA[n] for n, f in zip(model._names, model.parameters.free) if f]
    )
    dQ = np.array(
        [model.dQ[n] for n, f in zip(model._names, model.parameters.free) if f]
    )

    QQ = model.Q.T @ model.Q
    dQQ = dQ.swapaxes(1, 2) @ model.Q + model.Q.T @ dQ

    Qd = ssm.Q[0].T @ ssm.Q[0]
    dQd = dssm.dQ[0].swapaxes(1, 2) @ ssm.Q[0] + ssm.Q[0].T @ dssm.dQ[0]

    Qd_mfd = disc_diffusion_mfd(model.A, QQ, dt)
    Qd_lyap = disc_diffusion_lyap(model.A, QQ, ssm.A[0])
    Qd_kron = disc_diffusion_kron(model.A, QQ, ssm.A[0])

    tic = time()
    Qd_mfd_bis, dQd_mfd = disc_d_diffusion_mfd(model.A, QQ, dA, dQQ, dt)
    toc = time() - tic
    print(f"mfd: {toc:.4f}")

    tic = time()
    Qd_lyap_bis, dQd_lyap = disc_d_diffusion_lyap(
        model.A, QQ, ssm.A[0], dA, dQQ, dssm.dA[0]
    )
    toc = time() - tic
    print(f"lyapunov: {toc:.4f}")

    tic = time()
    Qd_kron_bis, dQd_kron = disc_d_diffusion_kron(
        model.A, QQ, ssm.A[0], dA, dQQ, dssm.dA[0]
    )
    toc = time() - tic
    print(f"kron: {toc:.4f}")

    assert np.allclose(Qd_kron, Qd)
    assert np.allclose(Qd_lyap, Qd)
    assert np.allclose(Qd_mfd, Qd)

    assert np.allclose(Qd_mfd_bis, Qd)
    assert np.allclose(Qd_lyap_bis, Qd)
    assert np.allclose(Qd_kron_bis, Qd)

    assert np.allclose(dQd_mfd, dQd)
    assert np.allclose(dQd_lyap, dQd)
    assert np.allclose(dQd_kron, dQd)

    if isinstance(model, GPModel):
        Pinf = model.P0.T @ model.P0
        dP0 = np.array(
            [model.dP0[n] for n, f in zip(model._names, model.parameters.free) if f]
        )
        dPinf = dP0.swapaxes(1, 2) @ model.P0 + model.P0.T @ dP0

        Qd_statio = disc_diffusion_stationary(Pinf, ssm.A[0])
        Qd_statio_bis, dQd_statio = disc_d_diffusion_stationary(
            Pinf, ssm.A[0], dPinf, dssm.dA[0]
        )

        assert np.allclose(Qd_statio, Qd)
        assert np.allclose(Qd_statio_bis, Qd)
        assert np.allclose(dQd_statio, dQd)
