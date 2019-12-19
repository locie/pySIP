"""Warning !!!
The algorithm may fail because of the random initialization, try to re-run the example.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import seaborn as sns
from pysip.utils import percentile_plot
from pysip.statespace import TwTi_RoRi
from pysip.regressors import BayesRegressor as Regressor
from pysip.core import Normal, Gamma

# Load and prepare the data
df = pd.read_csv('data/armadillo/armadillo_data_H2.csv').set_index('Time')
df.drop(df.index[-1], axis=0, inplace=True)
inputs = ['T_ext', 'P_hea']
y0 = df['T_int'][0]
sT = 3600.0 * 24.0
df.index /= sT

# Parameter settings for second order dynamic thermal model
parameters = [
    dict(name='Ro', scale=1e-2, bounds=(0, None), prior=Gamma(2, 0.1)),
    dict(name='Ri', scale=1e-3, bounds=(0, None), prior=Gamma(2, 0.1)),
    dict(name='Cw', scale=1e7 / sT, bounds=(0, None), prior=Gamma(2, 0.1)),
    dict(name='Ci', scale=1e6 / sT, bounds=(0, None), prior=Gamma(2, 0.1)),
    dict(name='sigw_w', scale=1e-2 * sT ** 0.5, bounds=(0, None), prior=Gamma(2, 0.1)),
    dict(name='sigw_i', value=0, transform='fixed'),
    dict(name='sigv', scale=1e-2, bounds=(0, None), prior=Gamma(2, 0.1)),
    dict(name='x0_w', loc=25, scale=7, prior=Normal(0, 1)),
    dict(name='x0_i', value=y0, transform='fixed'),
    dict(name='sigx0_w', value=0.1, transform='fixed'),
    dict(name='sigx0_i', value=0.1, transform='fixed'),
]

# Instantiate the model and use the first order hold approximation
model = TwTi_RoRi(parameters, hold_order=1)
reg = Regressor(model)

# Dynamic Hamiltonian Monte Carlo with multinomial sampling
fit = reg.fit(df=df, inputs=inputs, outputs='T_int')

# Compute the posterior predictive distribution
ym = reg.posterior_predictive(trace=fit.posterior, df=df, inputs=inputs)[0]

sns.set_style('darkgrid')
sns.set_context('talk')
percentile_plot(
    df.index,
    ym,
    n=10,
    percentile_min=0,
    percentile_max=100,
    plot_median=True,
    plot_mean=False,
    color='darkblue',
    line_color='navy',
)
plt.plot(df.index, df['T_int'], color='darkred')
plt.tight_layout()
sns.despine()

# Compute Leave-one out cross validation with pareto smoothing importance sampling
loglik = reg.pointwise_log_likelihood(trace=fit.posterior, df=df, inputs=inputs, outputs='T_int')
az_data = az.from_dict(posterior=fit.posterior, sample_stats=loglik)
psis = az.loo(az_data, pointwise=True)
