import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pysip.statespace import Matern32
from pysip.regressors import FreqRegressor as Regressor

# Generate artificial data
np.random.seed(1)
N = 20
t = np.sort(np.random.rand(1, N), axis=1).flatten()
y = np.sin(12 * t) + 0.66 * np.cos(25 * t) + np.random.randn(1, N) * 0.01
y = y.flatten()
df = pd.DataFrame(index=t, data=y, columns=['y'])

# Parameter settings for the Matérn covariance function with smoothness = 3/2
parameters = [
    dict(name='mscale', value=0.5, transform='log'),
    dict(name='lscale', value=0.5, transform='log'),
    dict(name='sigv', value=0.1, transform='log'),
]

reg = Regressor(Matern32(parameters))
fit_summary, corr_matrix, opt_summary = reg.fit(df=df, outputs='y')

# Fit results
print(f'\n{fit_summary}')

# Predict on test data
tnew = np.linspace(-0.1, 1.1, 500)
ym, ysd = reg.predict(df=df, outputs='y', tnew=tnew, smooth=True)

# Plot output mean and 95% credible intervals
sns.set_style('darkgrid')
sns.set_context('talk')
plt.plot(t, y, linestyle='', marker='+', mew=2, label='data', color='darkred')
plt.plot(tnew, ym, color='navy', label='mean')
plt.fill_between(tnew, ym - 2 * ysd, ym + 2 * ysd, color='darkblue', alpha=0.2, label=r'95% CI')
plt.tight_layout()
sns.despine()
plt.legend()
