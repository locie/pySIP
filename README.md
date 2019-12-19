# Stochastic State-Space Inference and Prediction in Python (pySIP)

Ordinary differential equations, determined from first-principles, can be used for modeling physical
systems. However, the exact dynamics of such systems are uncertain and only measured at
discrete-time instants through non-ideal sensors. In this case, stochastic differential equations
provide a modeling framework which is more robust to these uncertainties. The stochastic part of the
state-space model can accomodate for unmodeled disturbances, which do not have a significant
influence on the system dynamics. Otherwise, unmeasured disturbances can be modeled as temporal
**Gaussian Processes** with certain parametrized covariance structure. The resulting **Latent Force
Model** is a combination of parametric grey-box model and non-parametric Gaussian process model.

**pySIP** provides a framework for **infering continuous time linear stochastic state-space models**
. For that purpose, it is possible to chose between a **frequentist** and a **Bayesian** workflow.
Each workflow allows to estimate the parameters, assess the inference and model reliability, and
perform model selection.

**pySIP** is being developed in the perspective to build a library which gather models from
different engineering applications. Currently, applications involving dynamic thermal models
(RC network) and temporal Gaussian Processes are being prioritized. Nevertheless, any model
following the formalism of **pySIP** can benefit from the features.

**pySIP** is currently under development and in beta version. Please feel free to contact us if you
want to be involved in the current development process.

## Getting started

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pysip.statespace import TwTi_RoRi
from pysip.regressors import FreqRegressor as Regressor

# Load and prepare the data
df = pd.read_csv('data/armadillo/armadillo_data_H2.csv').set_index('Time')
df.drop(df.index[-1], axis=0, inplace=True)
inputs = ['T_ext', 'P_hea']
outputs = 'T_int'
sT = 3600.0 * 24.0
df.index /= sT  # Change time scale to days

# Parameter settings for second order dynamic thermal model
parameters = [
    dict(name='Ro', scale=1e-2, transform='log'),
    dict(name='Ri', scale=1e-3, transform='log'),
    dict(name='Cw', scale=1e7 / sT, transform='log'),
    dict(name='Ci', scale=1e6 / sT, transform='log'),
    dict(name='sigw_w', scale=1e-3 * sT ** 0.5, transform='log'),
    dict(name='sigw_i', value=0.0, transform='fixed'),
    dict(name='sigv', scale=1e-2, transform='log'),
    dict(name='x0_w', loc=25.0, scale=7.0, transform='none'),
    dict(name='x0_i', value=26.7, transform='fixed'),
    dict(name='sigx0_w', value=0.1, transform='fixed'),
    dict(name='sigx0_i', value=0.1, transform='fixed'),
]

# Instantiate the model and use the first order hold approximation
model = TwTi_RoRi(parameters, hold_order=1)
reg = Regressor(model)
fit_summary, corr_matrix, opt_summary = reg.fit(df=df, inputs=inputs, outputs=outputs)
print(f'\n{fit_summary}')

# Predict the indoor temperature each minute
dt = 60 / 3600
tnew = np.arange(df.index[0], df.index[-1], dt)
ym, ysd = reg.predict(df=df, inputs=inputs, tnew=tnew)

sns.set_style('darkgrid')
sns.set_context('talk')
plt.plot(df.index, df['T_int'], color='darkred', label='data')
plt.plot(tnew, ym, color='navy')
plt.fill_between(tnew, ym - 2 * ysd, ym + 2 * ysd, color='darkblue', alpha=0.2, label=r'95% CI')
plt.xlabel('time [days]')
plt.ylabel('temperature [°C]')
plt.tight_layout()
sns.despine()
plt.legend(loc='best', fancybox=True, framealpha=0.5)
```


## Contributors

* [Loïc Raillon](https://github.com/LoicRaillon) -
[Univ. Grenoble Alpes, Univ. Savoie Mont Blanc](https://www.locie.univ-smb.fr/en/home/), CNRS,
LOCIE, 73000 Chambéry, France,
* [Maxime Janvier](https://github.com/mjanv) - [Lancey Energy Storage](https://www.lancey.fr/en/)

## Funding

* Auvergne Rhône-Alpes, project HESTIA-Diag, habitat Econome avec Système Thermique Innovant Adapté
pour le Diagnostic
