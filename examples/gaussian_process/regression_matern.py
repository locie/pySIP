"""Dependencies"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pysip.statespace import Matern12, Matern32, Matern52
from pysip.regressors import FreqRegressor as Regressor


# Generate df
np.random.seed(1)
N = 20
t = np.sort(np.random.rand(1, N), axis=1).flatten()
y = np.sin(12 * t) + 0.66 * np.cos(25 * t) + np.random.randn(1, N) * 0.01
y = y.flatten()
df = pd.DataFrame(index=t, data=y, columns=['y'])

parameters = [
    dict(name="mscale", value=0.5, transform="log"),
    dict(name="lscale", value=0.5, transform="log"),
    dict(name="sigv", value=0.1, transform="log"),
]

reg12 = Regressor(Matern12(parameters))
reg32 = Regressor(Matern32(parameters))
reg52 = Regressor(Matern52(parameters))

# fit
print("\nMatérn(1/2)")
print("-" * 11)
reg12.fit(df=df, outputs='y')

print("\nMatérn(3/2)")
print("-" * 11)
reg32.fit(df=df, outputs='y')

print("\nMatérn(5/2)")
print("-" * 11)
reg52.fit(df=df, outputs='y')

# new data
Nnew = 500
tnew = np.linspace(-0.1, 1.1, Nnew)

# interpolate
ym12, yd12 = reg12.predict(df=df, outputs='y', tnew=tnew, smooth=True)
ym32, yd32 = reg32.predict(df=df, outputs='y', tnew=tnew, smooth=True)
ym52, yd52 = reg52.predict(df=df, outputs='y', tnew=tnew, smooth=True)

# plot different kernels
plt.close("all")
fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
fig.suptitle("Matérn covariances")
plt.xlim(-0.1, 1.1)

axes[0].set_title("smoothness = 1/2")
axes[0].plot(t, y, 'kx', mew=2)
axes[0].plot(tnew, ym12, 'C0', lw=2)
axes[0].fill_between(tnew, ym12 - 1.96 * yd12, ym12 + 1.96 * yd12, color='C0', alpha=0.2)

axes[1].set_title("smoothness = 3/2")
axes[1].plot(t, y, 'kx', mew=2)
axes[1].plot(tnew, ym32, 'C0', lw=2)
axes[1].fill_between(tnew, ym32 - 1.96 * yd32, ym32 + 1.96 * yd32, color='C0', alpha=0.2)

axes[2].set_title("smoothness = 5/2")
axes[2].plot(t, y, 'kx', mew=2)
axes[2].plot(tnew, ym52, 'C0', lw=2)
axes[2].fill_between(tnew, ym52 - 1.96 * yd52, ym52 + 1.96 * yd52, color='C0', alpha=0.2)

plt.show()
