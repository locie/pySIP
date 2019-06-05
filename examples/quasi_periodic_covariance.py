import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bopt.regressors import Regressor
from bopt.statespace.gaussian_process.gp_product import GPProduct
from bopt.statespace.gaussian_process.matern import Matern12
from bopt.statespace.gaussian_process.periodic import Periodic

np.random.seed(1)
xlim = (0, 7)
alim = (1.0, 5.0)
n = 20
period = 1
t = np.sort(xlim[0] + (xlim[1] - xlim[0]) * np.random.random(n))
np.random.seed(2)
amplitude = alim[0] + (alim[1] - alim[0]) * np.random.random()
noise = 0.2 * np.random.randn(len(t))
y = amplitude * np.sin(2.0 * np.pi / period * t) + noise
y[y <= 0] = 0.0
data = pd.DataFrame(index=t, data=y, columns=['y'])

par_Periodic = [
    dict(name="period", value=1.5, transform="log"),
    dict(name="mscale", value=1.0, transform="log"),
    dict(name="lscale", value=1.0, transform="log"),
    dict(name="sigv", value=0.1, transform="log")
]

par_Matern12 = [
    dict(name="mscale", value=1.0, transform="log"),
    dict(name="lscale", value=10.0, transform="log"),
    dict(name="sigv", value=0.1, transform="log")
]

quasi_periodic = GPProduct(Periodic(par_Periodic), Matern12(par_Matern12))

reg = Regressor(quasi_periodic)
results = reg.fit(df=data, outputs='y')

Nnew = 500
tnew = np.linspace(xlim[0], xlim[1] + 3, Nnew)
y_mean_f, y_std_f = reg.predict(df=data, outputs='y', tpred=tnew, smooth=False)
y_mean_s, y_std_s = reg.predict(df=data, outputs='y', tpred=tnew, smooth=True)

# plot filtered and smoothed output
plt.close("all")
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False)
fig.suptitle("filtered vs. smoothed state ")

axes[0].set_title("filtered")
axes[0].plot(t, y, 'kx', mew=2)
axes[0].plot(tnew, y_mean_f, 'C0', lw=2)
axes[0].fill_between(tnew,
                     y_mean_f - 1.96 * y_std_f,
                     y_mean_f + 1.96 * y_std_f,
                     color='C0', alpha=0.2)

axes[1].set_title("smoothed")
axes[1].plot(t, y, 'kx', mew=2)
axes[1].plot(tnew, y_mean_s, 'C0', lw=2)
axes[1].fill_between(tnew,
                     y_mean_s - 1.96 * y_std_s,
                     y_mean_s + 1.96 * y_std_s,
                     color='C0', alpha=0.2)

plt.show()
