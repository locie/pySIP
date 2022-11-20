import numpy as np
import pytest

from pysip.mcmc.hamiltonian import EuclideanHamiltonian
from pysip.mcmc.hmc import DynamicHMC, Fit_Bayes
from pysip.mcmc.metrics import Dense, Diagonal


@pytest.fixture
def mvn_data(n_dim=50):
    """Multivariate Normal distribution data"""
    rng = np.random.RandomState(seed=1234)
    rnd_eigvec, _ = np.linalg.qr(rng.normal(size=(n_dim, n_dim)))
    rnd_eigval = np.exp(rng.normal(size=n_dim) * 2)
    prec = (rnd_eigvec / rnd_eigval) @ rnd_eigvec.T
    mean = rng.normal(size=n_dim)
    return mean, prec, n_dim, rng


@pytest.mark.parametrize("dense_mass_matrix", [False, True])
def test_mvn_dhmc(mvn_data, dense_mass_matrix):
    """Multivariate Normal distribution dHMC test"""
    mean, prec, n_dim, rng = mvn_data
    n_chains = 4

    def dV(q):
        e = q - mean
        v = 0.5 * e @ prec @ e
        dv = prec @ e
        return v, dv

    if dense_mass_matrix is True:
        metric = Dense(np.identity(n_dim))
    else:
        metric = Diagonal(np.ones(n_dim))

    dHMC = DynamicHMC(EuclideanHamiltonian(potential=dV, metric=metric))
    q0 = rng.normal(size=(n_dim, n_chains))

    chains, stats, options = dHMC.sample(
        q=q0,
        n_chains=n_chains,
        n_draws=2500,
        n_warmup=1000,
        options={"n_cpu": 1, "dense_mass_matrix": dense_mass_matrix},
    )
    fit = Fit_Bayes(chains=chains, stats=stats, options=options, n_warmup=1000)
    df = fit.diagnostic
    mean_rmse = np.mean((chains.mean(axis=(0, 2)) - mean) ** 2) ** 0.5

    assert mean_rmse < 5e-2
    assert np.all(df["ebfmi"] > 0.8)
    assert np.all(df["mean accept_prob"] > 0.7)
    assert np.sum(df["sum diverging"]) == 0
    assert np.sum(df["sum max_tree_depth"]) == 0
