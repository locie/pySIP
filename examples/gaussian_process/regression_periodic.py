import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pysip.regressors import FreqRegressor as Regressor
from pysip.statespace import Periodic
from pysip.statespace.resonator import Resonator

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


parameters = [
    dict(name="period", value=1.0, transform="fixed"),
    dict(name="mscale", value=1.0, transform="log"),
    dict(name="lscale", value=1.0, transform="log"),
    dict(name="sigv", value=0.1, transform="log"),
]

pres = [
    dict(name="freq", value=1.0, transform="log"),
    dict(name="damp", value=1.0, transform="log"),
    dict(name="sigw", value=1.0, transform="log"),
    dict(name="sigv", value=1.0, transform="log"),
    dict(name="x0_f", value=1.0, transform="log"),
    dict(name="x0_df", value=1.0, transform="log"),
    dict(name="sigx0_f", value=1.0, transform="fixed"),
    dict(name="sigx0_df", value=1.0, transform="fixed"),
]

# reg = Regressor(Resonator(pres))
reg = Regressor(Periodic(parameters, J=7))

results = reg.fit(df=data, outputs='y')

Nnew = 500
tnew = np.linspace(xlim[0], xlim[1] + 3, Nnew)
ym, ystd = reg.predict(df=data, outputs='y', tnew=tnew, smooth=True)

fig = plt.figure(figsize=(9, 6), constrained_layout=True)
plt.rc('axes', linewidth=1.5)
plt.rc('legend', fontsize=14)

gs = fig.add_gridspec(1, 1)
axes = fig.add_subplot(gs[:, :])
axes.plot(t, y, 'kx', mew=2, label='data')
axes.plot(tnew, ym, color=sns.xkcd_rgb['windows blue'], lw=2, label='mean')
axes.fill_between(
    tnew,
    ym - 1.96 * ystd,
    ym + 1.96 * ystd,
    color=sns.xkcd_rgb['windows blue'],
    alpha=0.2,
    label=r'95% CI',
)

axes.legend(bbox_to_anchor=(1.1, 1.1), bbox_transform=axes.transAxes)
axes.set_xlabel('t', fontsize=14)
axes.set_ylabel('f(t)', fontsize=14)
axes.set_title(
    'Gaussian Process with periodic covariance', fontsize=14, fontweight='bold', loc='center'
)
axes.tick_params(axis='both', which='major', labelsize=14)
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
