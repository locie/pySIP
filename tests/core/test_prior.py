import numpy as np
import pytest
from pysip.core import Beta, Gamma, InverseGamma, Normal, LogNormal, MultivariateNormal


@pytest.fixture(name='N')
def n_samples():
    return 10000


@pytest.mark.parametrize('Prior', [Normal, Gamma, Beta, InverseGamma, LogNormal])
def test_prior_equality(Prior):
    assert Prior() == Prior()


@pytest.mark.parametrize(
    'Prior, log, dlog',
    [
        (Normal, -0.9639385332046727, -0.3),
        (Beta, 0.27990188513281833, 3.8095238095238098),
        (Gamma, -3.4010927892118175, 5.666666666666667),
        (InverseGamma, 0.7894107034104656, -2.222222222222223),
        (LogNormal, -0.43974098565696607, 0.6799093477531204),
    ],
)
def test_prior_normal(Prior, log, dlog, N):
    prior = Prior()
    rvs = prior.random(N)

    assert prior.log_pdf(0.3) == pytest.approx(log)
    assert prior.dlog_pdf(0.3) == pytest.approx(dlog)
    assert prior.mean == pytest.approx(np.mean(rvs), abs=3 * np.std(rvs) / np.sqrt(N))


def test_multivariate():
    mvn = MultivariateNormal(2)
    x = [0.3, 0.6]
    assert mvn.log_pdf(x) == pytest.approx(-2.0628770664093454)
