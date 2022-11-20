import arviz as az
import numpy as np
import pandas as pd
import pytest

from pysip.core import Gamma, Normal
from pysip.regressors import BayesRegressor as Regressor
from pysip.statespace import TwTi_RoRi


@pytest.mark.skip(reason="Too log for the Gitlab CI")
@pytest.mark.parametrize("dense_mass_matrix", [False, True])
def test_armadillo_hmc(dense_mass_matrix):
    n_cpu = 1
    np.random.seed(4567)

    # Prepare data
    df = pd.read_csv("data/armadillo/armadillo_data_H2.csv").set_index("Time")
    df.drop(df.index[-1], axis=0, inplace=True)
    inputs = ["T_ext", "P_hea"]
    y0 = df["T_int"][0]
    sT = 3600.0 * 24.0
    df.index /= sT

    # Parameter settings for second order dynamic thermal model
    parameters = [
        dict(name="Ro", scale=1e-2, bounds=(0, None), prior=Gamma(2, 0.1)),
        dict(name="Ri", scale=1e-3, bounds=(0, None), prior=Gamma(2, 0.1)),
        dict(name="Cw", scale=1e7 / sT, bounds=(0, None), prior=Gamma(2, 0.1)),
        dict(name="Ci", scale=1e6 / sT, bounds=(0, None), prior=Gamma(2, 0.1)),
        dict(
            name="sigw_w", scale=1e-2 * sT**0.5, bounds=(0, None), prior=Gamma(2, 0.1)
        ),
        dict(name="sigw_i", value=0, transform="fixed"),
        dict(name="sigv", scale=1e-2, bounds=(0, None), prior=Gamma(2, 0.1)),
        dict(name="x0_w", loc=25, scale=7, prior=Normal(0, 1)),
        dict(name="x0_i", value=y0, transform="fixed"),
        dict(name="sigx0_w", value=0.1, transform="fixed"),
        dict(name="sigx0_i", value=0.1, transform="fixed"),
    ]

    reg = Regressor(TwTi_RoRi(parameters, hold_order=1))

    fit = reg.fit(
        df=df,
        inputs=inputs,
        outputs="T_int",
        options={"n_cpu": n_cpu, "dense_mass_matrix": dense_mass_matrix},
    )

    ym = reg.posterior_predictive(
        trace=fit.posterior, df=df, inputs=inputs, n_cpu=n_cpu
    )[0]

    pwloglik = reg.pointwise_log_likelihood(
        trace=fit.posterior, df=df, inputs=inputs, outputs="T_int", n_cpu=n_cpu
    )

    # Inference diagnosis
    diag_ = fit.diagnostic
    assert np.all(diag_["ebfmi"] > 0.9)
    assert np.all((diag_["mean accept_prob"] > 0.7) & (diag_["mean accept_prob"] < 0.9))
    assert np.sum(diag_["sum diverging"]) == 0
    assert np.sum(diag_["sum max_tree_depth"]) == 0

    # Convergence diagnosis
    sum_ = az.summary(fit.posterior, round_to="none")
    assert np.all(sum_["r_hat"] < 1.01)
    assert np.all(sum_[["ess_mean", "ess_sd", "ess_bulk", "ess_tail"]] > 1000)
    assert sum_["mean"]["Ro"] == pytest.approx(1.778e-02, rel=1e-2)
    assert sum_["mean"]["Ri"] == pytest.approx(2.001e-03, rel=1e-2)
    assert sum_["mean"]["Cw"] == pytest.approx(1.714e02, rel=1e-2)
    assert sum_["mean"]["Ci"] == pytest.approx(1.908e01, rel=1e-2)
    assert sum_["mean"]["sigw_w"] == pytest.approx(5.503e-01, rel=1e-2)
    assert sum_["mean"]["sigv"] == pytest.approx(3.470e-02, rel=1e-2)
    assert sum_["mean"]["x0_w"] == pytest.approx(2.659e01, rel=1e-2)
    assert sum_["mean"]["lp_"] == pytest.approx(-3.012e02, rel=1e-2)

    # Predictions tests
    assert np.mean(
        np.mean((df["T_int"].values - ym) ** 2, axis=1) ** 0.5
    ) == pytest.approx(8.358e-01, rel=5e-2)

    # Point-wise log-likelihood tests
    assert pwloglik["log_likelihood"].sum(axis=2).mean() == pytest.approx(
        3.274e02, rel=1e-2
    )
