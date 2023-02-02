import numpy as np
import pytest

from numpy.testing import assert_allclose
from pysip.mcmc.adaptation import WelfordCovEstimator, WindowedAdaptation, CovAdaptation


@pytest.fixture
def mvn_samples(n_dim=3, n_samples=2000):
    np.random.seed(0)
    X = np.random.randn(n_dim, n_dim)
    cov = X @ X.T + np.identity(n_dim) * np.random.uniform(low=0.5, high=1.0)
    mean = np.random.randn(n_dim)
    samples = np.random.multivariate_normal(mean=mean, cov=cov, size=n_samples).T
    return samples, cov


@pytest.mark.parametrize('shrinkage', [False, True])
def test_var_adaptation(mvn_samples, shrinkage):
    samples, cov = mvn_samples
    n_dim, n_samples = samples.shape
    estimator = WelfordCovEstimator(dimension=n_dim, dense=False, shrinkage=shrinkage)
    for n in range(n_samples):
        estimator.add_sample(samples[:, n])

    var_hat = estimator.get_covariance()
    var_truth = cov.diagonal()
    assert_allclose(var_truth, var_hat, rtol=0.07)


@pytest.mark.parametrize('shrinkage', [False, True])
def test_covar_adaptation(mvn_samples, shrinkage):
    samples, cov = mvn_samples
    n_dim, n_samples = samples.shape
    estimator = WelfordCovEstimator(dimension=n_dim, dense=True, shrinkage=shrinkage)
    for n in range(n_samples):
        estimator.add_sample(samples[:, n])

    cov_hat = estimator.get_covariance()
    assert_allclose(cov, cov_hat, rtol=0.07)


def test_windowed_schedule():
    n_adapt = 1000
    schedule = WindowedAdaptation(n_adapt=n_adapt, init_buffer=75, term_buffer=50, window=25)
    index = []
    for _ in range(n_adapt):
        if schedule.end_adaptation_window:
            schedule.compute_next_window()
            schedule.increment_counter()
            index.append(schedule._counter)
        schedule.increment_counter()
    assert index == [100, 150, 250, 450, 950]


@pytest.mark.parametrize('shrinkage', [False, True])
def test_windowed_var_adaptation(mvn_samples, shrinkage):
    samples, cov = mvn_samples
    n_dim, n_samples = samples.shape
    estimator = WelfordCovEstimator(dimension=n_dim, dense=False, shrinkage=shrinkage)
    schedule = WindowedAdaptation(n_adapt=n_samples, init_buffer=75, term_buffer=100, window=25)
    adapter = CovAdaptation(estimator, schedule)
    index = []
    for n in range(n_samples):
        update, tmp = adapter.learn(samples[:, n])
        if update:
            index.append(n + 1)
            var_hat = tmp

    assert var_hat.shape == (n_dim,)
    assert index == [100, 150, 250, 450, 850, 1900]
    assert_allclose(cov.diagonal(), var_hat, rtol=0.07)


@pytest.mark.parametrize('shrinkage', [False, True])
def test_windowed_covar_adaptation(mvn_samples, shrinkage):
    samples, cov = mvn_samples
    n_dim, n_samples = samples.shape
    estimator = WelfordCovEstimator(dimension=n_dim, dense=True, shrinkage=shrinkage)
    schedule = WindowedAdaptation(n_adapt=n_samples, init_buffer=75, term_buffer=100, window=25)
    adapter = CovAdaptation(estimator, schedule)
    index = []
    for n in range(n_samples):
        update, tmp = adapter.learn(samples[:, n])
        if update:
            index.append(n + 1)
            cov_hat = tmp

    assert cov_hat.shape == (n_dim, n_dim)
    assert index == [100, 150, 250, 450, 850, 1900]
    assert_allclose(cov, cov_hat, rtol=0.07)
