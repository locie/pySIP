import arviz as az
import numpy as np
import pandas as pd
import pytest

from pysip.core.prior import Gamma, InverseGamma
from pysip.regressors import BayesRegressor as Regressor
from pysip.statespace import Matern32


@pytest.mark.xfail
@pytest.mark.parametrize('dense_mass_matrix', [False, True])
def test_fit_hmc_m32(dense_mass_matrix):
    """Generate samples from the posterior distribution"""
    n_cpu = 1
    np.random.seed(1)
    N = 50
    t = np.linspace(0, 1, N)
    y = np.sin(12 * t) + 0.66 * np.cos(25 * t) + np.random.randn(N) * 0.1
    df = pd.DataFrame(index=t, data=y, columns=['y'])

    par = [
        dict(name='mscale', value=1.11, bounds=(0, None), prior=Gamma(4, 4)),
        dict(name='lscale', value=0.15, bounds=(0, None), prior=InverseGamma(3.5, 0.5)),
        dict(name='sigv', value=0.1, bounds=(0, None), prior=InverseGamma(3.5, 0.5)),
    ]
    reg = Regressor(Matern32(par))
    fit = reg.fit(
        df=df,
        outputs='y',
        options={'init': 'fixed', 'n_cpu': n_cpu, 'dense_mass_matrix': dense_mass_matrix},
    )
    # return df, reg, fit
    diag_ = fit.diagnostic
    assert isinstance(diag_, pd.DataFrame)
    assert np.all(diag_['ebfmi'] > 0.9)
    assert np.all(diag_['mean accept_prob'] > 0.7)
    assert np.sum(diag_['sum diverging']) == 0
    assert np.sum(diag_['sum max_tree_depth']) == 0

    sum_ = az.summary(fit.posterior, round_to='none')
    assert isinstance(sum_, pd.DataFrame)
    assert np.all(sum_['r_hat'] < 1.01)
    assert np.all(sum_[['ess_mean', 'ess_sd', 'ess_bulk', 'ess_tail']] > 1000)
    assert sum_['mean']['mscale'] == pytest.approx(1.111, rel=1e-2)
    assert sum_['mean']['lscale'] == pytest.approx(1.468e-1, rel=1e-2)
    assert sum_['mean']['sigv'] == pytest.approx(9.625e-2, rel=1e-2)
    assert sum_['mean']['lp_'] == pytest.approx(2.909, rel=1e-2)

    xm, xsd = reg.posterior_state_distribution(
        trace=fit.posterior, df=df, outputs='y', smooth=True, n_cpu=n_cpu
    )
    assert isinstance(xm, np.ndarray)
    assert isinstance(xsd, np.ndarray)
    assert xm.shape == (4000, len(df), reg.ss.nx)
    assert xsd.shape == (4000, len(df), reg.ss.nx)
    assert np.mean(np.mean((df['y'].values - xm[:, :, 0]) ** 2, axis=1) ** 0.5) == pytest.approx(
        5.834e-2, rel=1e-2
    )

    ym, ysd = reg.posterior_predictive(trace=fit.posterior, df=df, outputs='y', n_cpu=n_cpu)
    assert isinstance(ym, np.ndarray)
    assert isinstance(ysd, np.ndarray)
    assert ym.shape == (4000, len(df))
    assert ysd.shape == (4000, len(df))
    assert np.mean(np.mean((df['y'].values - ym) ** 2, axis=1) ** 0.5) == pytest.approx(
        3.718e-2, rel=1e-2
    )

    pwloglik = reg.pointwise_log_likelihood(trace=fit.posterior, df=df, outputs='y', n_cpu=n_cpu)
    assert isinstance(pwloglik, dict)
    assert pwloglik['log_likelihood'].shape == (4, 1000, len(df))
    assert pwloglik['log_likelihood'].sum(axis=2).mean() == pytest.approx(-1.371, rel=1e-2)
