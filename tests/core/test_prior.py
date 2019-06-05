import numpy as np
import pytest

from bopt.core import Beta, Gamma, InverseGamma, Normal


@pytest.fixture(name='N')
def n_samples():
    return 10000


@pytest.mark.parametrize('Prior', [Normal, Gamma, Beta, InverseGamma])
def test_prior_equality(Prior):
    assert Prior() == Prior()


@pytest.mark.parametrize('Prior,log,dlog,d2log', [
    (Normal, -0.9639385332046727, -0.3, -1.0),
    (Beta, 0.27990188513281833, 3.8095238095238098, -26.30385487528345),
    (Gamma, 0.4011973816621559, -6.666666666666666, -11.11111111111111),
    (InverseGamma, -2.5462128007968396, -9.444444444444445, 29.62962962962963)
])
def test_prior_normal(Prior, log, dlog, d2log, N):
    prior = Prior()
    rvs = prior.random(N)

    assert prior.log_pdf(0.3) == pytest.approx(log)
    assert prior.dlog_pdf(0.3) == pytest.approx(dlog)
    assert prior.d2log_pdf(0.3) == pytest.approx(d2log)
    assert prior.mean == pytest.approx(np.mean(rvs), abs=3 * np.std(rvs) / np.sqrt(N))
