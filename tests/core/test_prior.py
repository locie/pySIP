import numpy as np
import pytest
import numdifftools as nd
from pysip.core import Beta, Gamma, InverseGamma, Normal, LogNormal
from scipy import stats


@pytest.fixture(name='N')
def n_samples():
    return 10000


@pytest.mark.parametrize('Prior', [Normal, Gamma, Beta, InverseGamma, LogNormal])
def test_prior_equality(Prior):
    assert Prior() == Prior()


@pytest.mark.parametrize(
    'Prior, log, dlog, scipy_fun',
    [
        (Normal, -0.9639385332046727, -0.3, stats.norm(loc=0, scale=1)),
        (Beta, 0.27990188513281833, 3.8095238095238098, stats.beta(a=3, b=3)),
        (Gamma, -3.4010927892118175, 5.666666666666667, stats.gamma(a=3, scale=1 / 1)),
        (InverseGamma, 0.7894107034104656, -2.222222222222223, stats.invgamma(a=3, scale=1)),
        (LogNormal, -0.43974098565696607, 0.6799093477531204, stats.lognorm(loc=0, s=1)),
    ],
)
def test_log_pdf(Prior, log, dlog, scipy_fun, N):
    prior = Prior()
    rvs = prior.random(N)
    x = np.random.uniform()

    assert prior.log_pdf(0.3) == pytest.approx(log)
    assert prior.dlog_pdf(0.3) == pytest.approx(dlog)
    assert prior.mean == pytest.approx(np.mean(rvs), abs=3 * np.std(rvs) / np.sqrt(N))
    assert prior.log_pdf(x) == pytest.approx(scipy_fun.logpdf(x))


@pytest.mark.parametrize('Prior', [Normal, Gamma, Beta, InverseGamma, LogNormal])
def test_dlog_pdf(Prior):
    prior = Prior()
    nd_prior = nd.Gradient(prior.log_pdf)
    x = np.random.uniform()
    assert prior.dlog_pdf(x) == pytest.approx(nd_prior(x), rel=1e-6)
