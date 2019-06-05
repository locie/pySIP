import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bopt.statespace.gaussian_process.matern import Matern32
from bopt.regressors import Regressor


# Generate data
np.random.seed(1)
N = 20
t = np.sort(np.random.rand(1, N), axis=1).flatten()
y = np.sin(12 * t) + 0.66 * np.cos(25 * t) + np.random.randn(1, N) * 0.01
y = y.flatten()
df = pd.DataFrame(index=t, data=y, columns=['y'])

# Set Matern(3/2)
parameters = [
    dict(name="mscale", value=0.5, transform="log"),
    dict(name="lscale", value=0.5, transform="log"),
    dict(name="sigv", value=0.1, transform="log")
]

reg = Regressor(Matern32(parameters))

results = reg.fit(df=df, outputs='y')

# new data
Nnew = 500
tnew = np.linspace(-0.1, 1.1, Nnew)
ym_f, ys_f = reg.predict(df=df, outputs='y', tpred=tnew, smooth=False)
ym_s, ys_s = reg.predict(df=df, outputs='y', tpred=tnew, smooth=True)

# plot filtered and smoothed output
plt.close("all")
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
fig.suptitle("filtered vs. smoothed state ")
plt.xlim(-0.1, 1.1)

axes[0].set_title("filtered")
axes[0].plot(t, y, 'kx', mew=2)
axes[0].plot(tnew, ym_f, 'C0', lw=2)
axes[0].fill_between(tnew,
                     ym_f - 1.96 * ys_f,
                     ym_f + 1.96 * ys_f,
                     color='C0', alpha=0.2)

axes[1].set_title("smoothed")
axes[1].plot(t, y, 'kx', mew=2)
axes[1].plot(tnew, ym_s, 'C0', lw=2)
axes[1].fill_between(tnew,
                     ym_s - 1.96 * ys_s,
                     ym_s + 1.96 * ys_s,
                     color='C0', alpha=0.2)

plt.show()
