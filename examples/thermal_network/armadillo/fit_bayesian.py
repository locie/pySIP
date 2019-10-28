import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

from pysip.utils import percentile_plot
from pysip.statespace import TwTi_RoRi
from pysip.regressors import BayesRegressor as Regressor
from pysip.core import Normal, Gamma

do_prior_pd = False
do_fit = True
do_posterior_pd = False
do_psis = False
do_save = False
do_plot = False

df = pd.read_csv('data/armadillo/armadillo_data_H2.csv').set_index('Time')
df.drop(df.index[-1], axis=0, inplace=True)
inputs = ['T_ext', 'P_hea']
outputs = 'T_int'

parameters = [
    dict(name='Ro', bounds=(0, None), prior=Normal(0.0505, 0.0213)),
    dict(name='Ri', bounds=(0, None), prior=Normal(0.0505, 0.0213)),
    dict(name='Cw', scale=1e8, bounds=(0, None), prior=Normal(0.5050, 0.2128)),
    dict(name='Ci', scale=1e8, bounds=(0, None), prior=Normal(0.0610, 0.0241)),
    dict(name='sigw_w', bounds=(0, None), prior=Gamma(2, 10)),
    dict(name='sigw_i', value=1e-8, transform='fixed'),
    dict(name='sigv', bounds=(0, None), prior=Gamma(2, 10)),
    dict(name='x0_w', scale=1e2, bounds=(0, None), prior=Normal(0.2750, 0.0537)),
    dict(name='x0_i', value=26.7, transform='fixed'),
    dict(name='sigx0_w', value=1.0, transform='fixed'),
    dict(name='sigx0_i', value=1.0, transform='fixed'),
]

reg = Regressor(TwTi_RoRi(parameters, hold_order=1))

if do_prior_pd:
    prior_draw, prior_pd = reg.prior_predictive(df=df, inputs=inputs, Nsim=10000, prob_mass=0.98)
    ymin = df['T_ext'].min()
    ymax = df['T_ext'].max() * 5.0
    idx_max = np.any(prior_pd['xi'].squeeze() > ymax, axis=1)
    idx_min = np.any(prior_pd['xi'].squeeze() < ymin, axis=1)
    ppd = prior_pd['xi'].squeeze().ravel()
    print(f'below lower bounds {np.sum(ppd < ymin) / ppd.shape[0]}')
    print(f'over upper bounds {np.sum(ppd > ymax) / ppd.shape[0]}')

if do_fit:
    fit = reg.fit(df=df, inputs=inputs, outputs=outputs, options={'init': 'prior', 'hpd': 0.5})

if do_posterior_pd:
    posterior_pd = reg.posterior_predictive(trace=fit.posterior, df=df, inputs=inputs)

if do_psis:
    pw_loglik = reg.pointwise_log_likelihood(
        trace=fit.posterior, df=df, inputs=inputs, outputs=outputs
    )

if do_save:
    data = az.from_dict(
        posterior=fit.posterior,
        posterior_predictive=posterior_pd,
        sample_stats=pw_loglik,
        prior=prior_draw,
        prior_predictive=prior_pd,
    )
    data.to_netcdf('fit_bayesian_Armadillo')

if do_plot:
    percentile_plot(
        df.index,
        prior_pd['xi'].squeeze(),
        n=10,
        percentile_min=2.5,
        percentile_max=97.5,
        plot_median=True,
        plot_mean=False,
        color='darkblue',
        line_color='navy',
    )
    plt.plot(df.index, ymin * np.ones(df.index.shape[0]), 'r')
    plt.plot(df.index, ymax * np.ones(df.index.shape[0]), 'r')

    percentile_plot(
        df.index,
        posterior_pd['xi'].squeeze(),
        n=10,
        percentile_min=2.5,
        percentile_max=97.5,
        plot_median=True,
        plot_mean=False,
        color='darkred',
        line_color='maroon',
    )

    plt.plot(df.index, df[outputs], '--k')
    plt.tight_layout()
