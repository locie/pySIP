import numpy as np
import pytest
from scipy import stats

from pysip.params.prior import Beta, Gamma, InverseGamma, LogNormal, Normal


@pytest.fixture(name="N")
def n_samples():
    return 10000


@pytest.mark.parametrize("Prior", [Normal, Gamma, Beta, InverseGamma, LogNormal])
def test_prior_equality(Prior):
    assert Prior() == Prior()


@pytest.mark.parametrize(
    "Prior, log, scipy_fun",
    [
        (Normal, -0.9639385332046727, stats.norm(loc=0, scale=1)),
        (Beta, 0.27990188513281833, stats.beta(a=3, b=3)),
        (Gamma, -3.4010927892118175, stats.gamma(a=3, scale=1 / 1)),
        (InverseGamma, 0.7894107034104656, stats.invgamma(a=3, scale=1)),
        (LogNormal, -0.43974098565696607, stats.lognorm(loc=0, s=1)),
    ],
)
def test_log_pdf(Prior, log, scipy_fun, N):
    prior = Prior()
    rvs = prior.random(N)
    x = np.random.uniform()

    assert prior.logpdf(0.3) == pytest.approx(log)
    assert prior.mean == pytest.approx(np.mean(rvs), abs=3 * np.std(rvs) / np.sqrt(N))
    assert prior.logpdf(x) == pytest.approx(scipy_fun.logpdf(x))
