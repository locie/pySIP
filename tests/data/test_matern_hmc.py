import pytest
import numpy as np
import pandas as pd
import arviz as az
from pysip.statespace import Matern32
from pysip.regressors import BayesRegressor as Regressor
from pysip.core.prior import InverseGamma, Gamma

"""The samples are not cached if the fit is done inside the fixture"""


# @pytest.mark.skip(reason="to be unfixed")
def test_fit_hmc_m32():
    """Generate samples from the posterior distribution"""
    n_cpu = 1
    np.random.seed(1)
    N = 50
    t = np.linspace(0, 1, N)
    y = np.sin(12 * t) + 0.66 * np.cos(25 * t) + np.random.randn(N) * 0.1
    df = pd.DataFrame(index=t, data=y, columns=['y'])

    par = [
        dict(name='mscale', value=9.313e-01, bounds=(0, None), prior=Gamma(4, 4)),
        dict(name='lscale', value=1.291e-01, bounds=(0, None), prior=InverseGamma(3.5, 0.5)),
        dict(name='sigv', value=9.241e-02, bounds=(0, None), prior=InverseGamma(3.5, 0.5)),
    ]
    reg = Regressor(Matern32(par))
    fit = reg.fit(df=df, outputs='y', options={'init': 'fixed', 'n_cpu': n_cpu})
    # return df, reg, fit
    diagnostic = fit.diagnostic
    assert isinstance(diagnostic, pd.DataFrame)
    assert np.all(diagnostic['ebfmi'] > 0.8)
    assert np.all(diagnostic['mean accept_prob'] > 0.7)
    assert np.sum(diagnostic['sum diverging']) == 0
    assert np.sum(diagnostic['sum max_tree_depth']) == 0

    summary = az.summary(fit.posterior, round_to='none')
    assert isinstance(summary, pd.DataFrame)
    assert np.all(summary['r_hat'] < 1.01)
    assert np.all(summary[['ess_mean', 'ess_sd', 'ess_bulk', 'ess_tail']] > 1000)
    # mcse for ess_mean = 1000
    assert summary['mean']['mscale'] == pytest.approx(1.107023, abs=3 * 0.009261)
    assert summary['mean']['lscale'] == pytest.approx(0.146614, abs=3 * 0.001074)
    assert summary['mean']['sigv'] == pytest.approx(0.096477, abs=3 * 0.000515)
    assert summary['mean']['lp_'] == pytest.approx(2.919439, abs=3 * 0.038186)

    xm, xsd = reg.posterior_state_distribution(
        trace=fit.posterior, df=df, outputs='y', smooth=True, n_cpu=n_cpu
    )
    assert isinstance(xm, np.ndarray)
    assert isinstance(xsd, np.ndarray)
    assert xm.shape == (4000, len(df), reg.ss.nx)
    assert xsd.shape == (4000, len(df), reg.ss.nx)
    assert np.mean(np.mean((df['y'].values - xm[:, :, 0]) ** 2, axis=1) ** 0.5) == pytest.approx(
        5.839e-2, abs=1e-2
    )

    ym, ysd = reg.posterior_predictive(trace=fit.posterior, df=df, outputs='y', n_cpu=n_cpu)
    assert isinstance(ym, np.ndarray)
    assert isinstance(ysd, np.ndarray)
    assert ym.shape == (4000, len(df))
    assert ysd.shape == (4000, len(df))
    assert np.mean(np.mean((df['y'].values - ym) ** 2, axis=1) ** 0.5) == pytest.approx(
        3.728e-2, abs=1e-2
    )

    pw_loglik = reg.pointwise_log_likelihood(trace=fit.posterior, df=df, outputs='y', n_cpu=n_cpu)
    assert isinstance(pw_loglik, dict)
    assert pw_loglik['log_likelihood'].shape == (4, 1000, len(df))
    # 0.026 ~ pw_loglik['log_likelihood'].sum(axis=2).std() / np.sqrt(1000)
    assert pw_loglik['log_likelihood'].sum(axis=2).mean() == pytest.approx(-1.394, abs=3.256e-2)


# def test_hmc_diagnostic(fit_hmc_m32):
#     """HMC transitions statistics"""
#     _, _, fit = fit_hmc_m32
# diagnostic = fit.diagnostic
# assert isinstance(diagnostic, pd.DataFrame)
# assert np.all(diagnostic['ebfmi'] > 0.8)
# assert np.all(diagnostic['mean accept_prob'] > 0.7)
# assert np.sum(diagnostic['sum diverging']) == 0
# assert np.sum(diagnostic['sum max_tree_depth']) == 0


# def test_matern_posterior(fit_hmc_m32):
#     """Test the posterior distribution statistics"""
#     _, _, fit = fit_hmc_m32
#     summary = az.summary(fit.posterior, round_to='none')
#     assert isinstance(summary, pd.DataFrame)
#     assert np.all(summary['r_hat'] < 1.01)
#     assert np.all(summary[['ess_mean', 'ess_sd', 'ess_bulk', 'ess_tail']] > 1000)
#     # 4 times the mcse_mean and mcse_sd
#     assert summary['mean']['mscale'] == pytest.approx(1.162614, abs=4 * 0.006912)
#     assert summary['mean']['lscale'] == pytest.approx(0.155057, abs=4 * 0.000730)
#     assert summary['mean']['sigv'] == pytest.approx(0.101054, abs=4 * 0.000231)
#     assert summary['mean']['lp_'] == pytest.approx(4.630847, abs=4 * 0.026929)


# def test_state_posterior(fit_hmc_m32):
#     df, reg, fit = fit_hmc_m32
#     xm, xsd = reg.posterior_state_distribution(trace=fit.posterior, df=df, outputs='y', smooth=True)
#     assert isinstance(xm, np.ndarray)
#     assert isinstance(xsd, np.ndarray)
#     assert xm.shape == (4000, len(df), reg.ss.nx)
#     assert xsd.shape == (4000, len(df), reg.ss.nx)
#     assert np.mean(np.mean((df['y'].values - xm[:, :, 0]) ** 2, axis=1) ** 0.5) == pytest.approx(
#         6.045e-2, abs=1e-2
#     )


# def test_output_posterior(fit_hmc_m32):
#     df, reg, fit = fit_hmc_m32
#     ym, ysd = reg.posterior_predictive(trace=fit.posterior, df=df, outputs='y')
#     assert isinstance(ym, np.ndarray)
#     assert isinstance(ysd, np.ndarray)
#     assert ym.shape == (4000, len(df))
#     assert ysd.shape == (4000, len(df))
#     assert np.mean(np.mean((df['y'].values - ym) ** 2, axis=1) ** 0.5) == pytest.approx(
#         3.987e-2, abs=1e-2
#     )


# def test_pw_loglikelihood(fit_hmc_m32):
#     df, reg, fit = fit_hmc_m32
#     pw_loglik = reg.pointwise_log_likelihood(trace=fit.posterior, df=df, outputs='y')
#     assert isinstance(pw_loglik, dict)
#     assert pw_loglik['log_likelihood'].shape == (4, 1000, len(df))
#     # 0.026 ~ pw_loglik['log_likelihood'].sum(axis=2).std() / np.sqrt(1000)
#     assert pw_loglik['log_likelihood'].sum(axis=2).mean() == pytest.approx(-1.498, abs=0.037)
