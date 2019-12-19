import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pysip.regressors import FreqRegressor as Regressor
from pysip.statespace import Periodic

# Generate artificial periodic data
np.random.seed(1)
xlim = (0, 7)
alim = (1, 3)
n = 20
period = 1
t = np.sort(xlim[0] + (xlim[1] - xlim[0]) * np.random.random(n))
amplitude = alim[0] + (alim[1] - alim[0]) * np.random.random()
noise = 0.2 * np.random.randn(n)
y = amplitude * np.sin(2.0 * np.pi / period * t) + noise
y[y <= 0] = 0.0
data = pd.DataFrame(index=t, data=y, columns=['y'])


# Parameter settings for the Periodic covariance function
parameters = [
    dict(name='period', value=1.0, transform='fixed'),
    dict(name='mscale', value=1.0, transform='log'),
    dict(name='lscale', value=1.0, transform='log'),
    dict(name='sigv', value=0.1, transform='log'),
]

# Instantiate regressor with the Periodic covariance function
reg = Regressor(Periodic(parameters))

fit_summary, corr_matrix, opt_summary = reg.fit(df=data, outputs='y')

# Fit results
print(f'\n{fit_summary}')

# Predict on test data
tnew = np.linspace(xlim[0], xlim[1] + 1, 500)
ym, ysd = reg.predict(df=data, outputs='y', tnew=tnew, smooth=True)

# Plot output mean and 95% credible intervals
sns.set_style('darkgrid')
sns.set_context('talk')
plt.plot(t, y, linestyle='', marker='+', mew=2, label='data', color='darkred')
plt.plot(tnew, ym, color='navy', label='mean')
plt.fill_between(tnew, ym - 2 * ysd, ym + 2 * ysd, color='darkblue', alpha=0.2, label=r'95% CI')
plt.tight_layout()
sns.despine()
plt.legend()
