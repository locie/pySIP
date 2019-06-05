import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bopt.regressors import Regressor
from bopt.statespace.gaussian_process.periodic import Periodic

np.random.seed(1)
xlim = (0, 7)
alim = (1, 3)
n = 20
period = 1
t = np.sort(xlim[0] + (xlim[1] - xlim[0]) * np.random.random(n))
amplitude = alim[0] + (alim[1] - alim[0]) * np.random.random()
noise = 0.2 * np.random.randn(len(t))
y = amplitude * np.sin(2.0 * np.pi / period * t) + noise
y[y <= 0] = 0.0
data = pd.DataFrame(index=t, data=y, columns=['y'])


parameters = [
    dict(name="period", value=1.0, transform="fixed"),
    dict(name="mscale", value=1.0, transform="log"),
    dict(name="lscale", value=1.0, transform="log"),
    dict(name="sigv", value=0.1, transform="log")
]

reg = Regressor(Periodic(parameters, J=6))

results = reg.fit(df=data, outputs='y')

Nnew = 500
tnew = np.linspace(xlim[0], xlim[1] + 3, Nnew)
y_mean_f, y_std_f = reg.predict(df=data, outputs='y', tpred=tnew, smooth=False)
y_mean_s, y_std_s = reg.predict(df=data, outputs='y', tpred=tnew, smooth=True)

# plot filtered and smoothed output
# plt.close("all")
# fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False)
# fig.suptitle("filtered vs. smoothed state ")

# axes[0].set_title("filtered")
# axes[0].plot(t, y, 'kx', mew=2)
# axes[0].plot(tnew, y_mean_f, 'C0', lw=2)
# axes[0].fill_between(tnew,
#                      y_mean_f - 1.96 * y_std_f,
#                      y_mean_f + 1.96 * y_std_f,
#                      color='C0', alpha=0.2)

# axes[1].set_title("smoothed")
# axes[1].plot(t, y, 'kx', mew=2)
# axes[1].plot(tnew, y_mean_s, 'C0', lw=2)
# axes[1].fill_between(tnew,
#                      y_mean_s - 1.96 * y_std_s,
#                      y_mean_s + 1.96 * y_std_s,
#                      color='C0', alpha=0.2)

# plt.show()

# print("\n")
# print("{0:^12} {1:^12} {2:^12} {3:^12} {4:^12} {5:^12}".format(
#         "name", "\u03B8", "\u03C3(\u03B8)", "pvalue", "|g(\u03B7)|",
#         "|dpen(\u03B7)|"))
# print("{0:^12} {0:^12} {0:^12} {0:^12} {0:^12} {0:^12}".format(
#         "-----------"))
# for i in range(results[0].shape[0]):
#     print("{0:^12} {1:^12.3e} {2:^12.3e} {3:^12.3e} {4:^12.3e} "
#             "{5:^12.3e}".format(results[0].index[i],
#                                 results[0].iloc[i, 0],
#                                 results[0].iloc[i, 1],
#                                 results[0].iloc[i, 2],
#                                 results[0].iloc[i, 3],
#                                 results[0].iloc[i, 4]))

import seaborn as sns

fig = plt.figure(figsize=(9, 6), constrained_layout=True)
plt.rc('axes', linewidth=1.5)
plt.rc('legend', fontsize=14)

gs = fig.add_gridspec(1, 1)
axes = fig.add_subplot(gs[:, :])
axes.plot(t, y, 'kx', mew=2, label='data')
axes.plot(tnew, y_mean_s, color=sns.xkcd_rgb['windows blue'], lw=2, label='mean')
axes.fill_between(tnew,
                  y_mean_s - 1.96 * y_std_s,
                  y_mean_s + 1.96 * y_std_s,
                  color=sns.xkcd_rgb['windows blue'],
                  alpha=0.2, label='$95\%$ CI')

axes.legend(bbox_to_anchor=(1.1, 1.1), bbox_transform=axes.transAxes)
axes.set_xlabel('t', fontsize=14)
axes.set_ylabel('f(t)', fontsize=14)
axes.set_title('Gaussian Process with periodic covariance', fontsize=14, fontweight='bold', loc='center')
axes.tick_params(axis='both', which='major', labelsize=14)
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)

plt.savefig('periodic_kernel', dpi = 1200)