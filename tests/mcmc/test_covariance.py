import numpy as np
import pytest
from pysip.mcmc.adaptation import WelfordCovEstimator


@pytest.fixture
def mvn_samples(n_dim=10, n_samples=500):
    # Generate random PSD matrix
    X = np.random.rand(n_dim, n_dim)
    U, _, V = np.linalg.svd(np.dot(X.T, X))
    cov = np.dot(np.dot(U, 1.0 + np.diag(np.random.rand(n_dim))), V)
    mean = np.zeros(n_dim)
    samples = np.random.multivariate_normal(mean=mean, cov=cov, size=n_samples).T
    return samples, cov


def test_var_adaptation(mvn_samples):
    samples, cov = mvn_samples
    n_dim, n_samples = samples.shape
    estimator = WelfordCovEstimator(dimension=n_dim, diagonal=True)
    for n in range(n_samples):
        estimator.add_sample(samples[:, n])

    err = cov.diagonal() - estimator.get_covariance(shrinkage=False)
    assert pytest.approx(err, abs=np.std(samples) / np.sqrt(n_samples))


def test_covar_adaptation(mvn_samples):
    samples, cov = mvn_samples
    n_dim, n_samples = samples.shape
    estimator = WelfordCovEstimator(dimension=n_dim, diagonal=False)
    for n in range(n_samples):
        estimator.add_sample(samples[:, n])

    err = cov - estimator.get_covariance(shrinkage=False)
    assert pytest.approx(err, abs=np.std(samples) / np.sqrt(n_samples))
