

# Python Stochastic State-Space Inference and Prediction (pySIP)

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)

**pySIP** provides a framework for *infering continuous time stochastic state-space models*. 

This kind of models can be found in a broad class of applications involving time-varying phenomena which are only observed at discrete time instants. It allows to estimate the system dynamics and parameters given noisy measurements. pySIP provides a frequentist and a Bayesian workflow for model identification and prediction. The two workflows have methods for parameter estimation, model selection and convergence diagnosis. 

**pySIP** is being developed in the perspective to build a library which gather models from different engineering applications. Currently, two applications are being prioritized:
* dynamic thermal models (using RC networks)
* temporal Gaussian process

**pySIP** offers also the possibility to combine a physical model with a *Gaussian Process* to form a *Latent Force Model*, which can be used to model unknow input signals (latent forces) in physical systems.

**pySIP** is currently under development and in beta version. A first stable release will be available (hopefully) before the 2019 Christmas. Please feel free to contact us if you want to be involved in the current development process. 

## Getting started


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bopt.regressors import Regressor
from bopt.statespace.rc import TwTi_RoRi
from bopt.utils import save_model

# Load some data...
df = pd.read_csv('data/armadillo/armadillo_data_H2.csv').set_index('Time')
print(df.head())

# Fix the state-space model parameters
parameters = [
    dict(name='Ro', value=0.1, transform='log'),
    dict(name='Ri', value=0.01, transform='log'),
    dict(name='Cw', value=0.1, transform='log'),
    dict(name='Ci', value=0.01, transform='log'),
    dict(name='sigw_w', value=0.01, transform='log'),
    dict(name='sigw_i', value=0.0, transform='fixed'),
    dict(name='sigv', value=0.01, transform='log'),
    dict(name='x0_w', value=0.25, transform='log'),
    dict(name='x0_i', value=0.267, transform='fixed'),
    dict(name='sigx0_w', value=1.0, transform='fixed'),
    dict(name='sigx0_i', value=1.0, transform='fixed')
]

# You can choose any state-space model
ssm = TwTi_RoRi(parameters, hold_order='foh')

# Training
# Feed a regressor with your new state-space model and fit it with the Armadillo data
# `inputs` and `outputs` are column names in the dataframe `df`
reg = Regressor(ssm)
results = reg.fit(df=df, inputs=['T_ext', 'P_hea'], outputs='T_int')

print(reg.summary_)

# Save your model !
save_model('model_saved', reg)

# Prediction
# The training dataset is sampled at 30 minutes but the output 
# predictive distribution can be evaluated at **any** timesteps
dt = 60
tnew = np.arange(df.index[0], df.index[-1] + dt, dt)
y_mean, y_std = reg.predict(df=df, inputs=['T_ext', 'P_hea'], tpred=tnew)

plt.plot(df.index, df['T_int'], 'kx')
plt.plot(tnew, y_mean, 'C0', lw=2)
plt.fill_between(tnew,
                 y_mean - 1.96 * y_std,
                 y_mean + 1.96 * y_std,
                 color='C0', alpha=0.2)

plt.show()
```


## Contributors

* [Loïc Raillon](https://github.com/LoicRaillon) - [Univ. Grenoble Alpes, Univ. Savoie Mont Blanc](https://www.locie.univ-smb.fr/en/home/ ) CNRS, LOCIE, 73000 Chambéry, France,
* [Maxime Janvier](https://github.com/mjanv) - [Lancey Energy Storage]( https://www.lancey.fr/en/) 

## Funding

* Auvergne Rhône-Alpes, project HESTIA-Diag,habitat Econome avec Système Thermique Innovant Adapté pour le Diagnostic
