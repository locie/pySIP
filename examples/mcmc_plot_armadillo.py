import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from bopt.utils import load_model
from bopt.utils import ccf, plot_ccf, autocovf
from bopt.regressors import Regressor
from bopt.statespace.rc import TwTi_RoRi
from statsmodels.distributions.empirical_distribution import ECDF

plt.close('all')

# load sMMALA instance
mh = load_model('mcmc_armadillo')

# set warmup manually and display diagnostic
mh.warmup = 50
mh.diagnostic(warmup_estimation=False)
'''See diagnostic_mcmc.png'''

# remove warm-up samples
chains = mh.trace[:, :, mh.warmup:]

# split each chains into a first and second half of the same dimension
N = int((mh._N - mh.warmup) / 2)
M = mh._M * 2
chains = np.concatenate((chains[:, :, -2 * N: -N], chains[:, :, -N:]))

# compute the estimate IACT for each parameter
rho_hat = np.empty((mh._Np + 1, N))
for i in range(mh._Np + 1):
    # compute the autocovariance function for the M chains
    acov = np.asarray([autocovf(chains[m, i, :]) for m in range(M)])

    # between chain variance divided by N
    B_over_N = np.var(chains[:, i, :].mean(axis=1), ddof=1)

    # within chain variance (note: the autocorrelation function at the
    # lag 0 is equal to the sample variance of the chain)
    W = np.mean(acov[:, 0] * N / (N - 1.))

    # mixture of the within-chain and cross-chain sample variances
    var_hat = (1 - 1 / N) * W + B_over_N

    # combined autocorrelation
    rho_hat[i, :] = 1. - (W - np.mean(acov * N / (N - 1), axis=0)) / var_hat

'''Choose a parameter'''
parameter = 'Cw'
for p, name in enumerate(mh._par.names_free):
    if name == parameter:
        break

if parameter in ['Cw', 'Ci']:
    print('Must be multiplied by 1e8')

# Choose a number of lags to plot
lags = 50
cc = ["dusty purple", "amber", "faded green", "windows blue"]

fig = plt.figure(figsize=(9, 6), constrained_layout=True)
plt.rc('axes', linewidth=1.5)
plt.rc('legend', fontsize=14)
gs = fig.add_gridspec(2, 5)
ax1 = fig.add_subplot(gs[0, :3])
ax2 = fig.add_subplot(gs[1, :3])
ax3 = fig.add_subplot(gs[:, -2:])

ax1.plot(mh.trace[0, p, :], color=sns.xkcd_rgb[cc[0]], linewidth=2, label='chain 1')
ax1.plot(mh.trace[1, p, :], color=sns.xkcd_rgb[cc[1]], linewidth=2, label='chain 2')
ax1.plot(mh.trace[2, p, :], color=sns.xkcd_rgb[cc[2]], linewidth=2, label='chain 3')
ax1.plot(mh.trace[3, p, :], color=sns.xkcd_rgb[cc[3]], linewidth=2, label='chain 4')
# ax1.legend()
ax1.set_ylabel('Parameter value', fontsize=14)
ax1.set_xlabel('Iterations', fontsize=14)
ax1.set_title('Markov Chain traces', fontsize=14, fontweight='bold')
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

markerline, stemlines, baseline = ax2.stem(rho_hat[p, :lags], linefmt='-', markerfmt='None')
plt.setp(baseline, color=sns.xkcd_rgb[cc[-1]], linewidth=2)
plt.setp(stemlines, color=sns.xkcd_rgb[cc[-1]], linewidth=2)
ax2.set_xlabel('Lags', fontsize=14)
ax2.set_ylabel('Correlation', fontsize=14)
ax2.set_title('Autocorrelation', fontsize=14, fontweight='bold')
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

sns.distplot(mh.trace[:, p, mh.warmup:].ravel(), ax=ax3, hist=False, kde=True, color=sns.xkcd_rgb[cc[-1]],
             kde_kws = {'bw': 'scott', 'shade': True, 'linewidth': 1.5})
ax3.set_xlabel('Parameter value', fontsize=14)
ax3.set_ylabel('Density', fontsize=14)
ax3.set_title('Posterior distribution', fontsize=14, fontweight='bold')
ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)

plt.savefig('mcmc_trace_armadillo', dpi = 1200)

'''Manual posterior distribution, it will be integrated in the pySIP soon'''
df = pd.read_csv('data/armadillo/armadillo_data_H2.csv').set_index('Time')

parameters = [
    dict(name="Ro", value=0.1),
    dict(name="Ri", value=0.01),
    dict(name="Cw", value=0.1),
    dict(name="Ci", value=0.01),
    dict(name="sigw_w", value=0.01),
    dict(name="sigw_i", value=0.0, transform="fixed"),
    dict(name="sigv", value=0.01),
    dict(name="x0_w", value=0.25),
    dict(name="x0_i", value=0.2670106194217502, transform="fixed"),
    dict(name="sigx0_w", value=1.0, transform="fixed"),
    dict(name="sigx0_i", value=1.0, transform="fixed")
]
reg = Regressor(TwTi_RoRi(parameters, hold_order='foh'))

Nsamples = mh._N - mh.warmup
yhat = np.empty((Nsamples, df.index.shape[0]))

chains = np.empty((mh._Np+1, Nsamples*mh._M))
for i in range(mh._Np+1):
    chains[i, :] = mh.trace[:, i, mh.warmup:].ravel()

# chains[-1, :] = np.abs(chains[-1, :])
# posterior = ( chains[-1, :] - chains[-1, :].min() ) / ( chains[-1, :].max() - chains[-1, :].min() )
ecdf = ECDF(chains[-1, :])
min_ecdf = ecdf.x[np.where(ecdf.y <= 0.025)[0][-1]]
max_ecdf = ecdf.x[np.where(ecdf.y >= 0.975)[0][0]]

for i in range(Nsamples):
    reg.ss.parameters.eta = chains[:-1, i]
    yhat[i, :], _ = reg.predict(df=df, inputs=['T_ext', 'P_hea'])

fig2 = plt.figure(figsize=(9, 6), constrained_layout=True)
gs = fig2.add_gridspec(1, 1)
axes = fig2.add_subplot(gs[:, :])
PuBu = np.asarray(sns.color_palette("PuBu", 10))

for i in range(Nsamples):
    # idx = PuBu[int(posterior[i]*10), :]
    if chains[-1, i] > min_ecdf or chains[-1, i] < max_ecdf:
        axes.plot(df.index/3600, yhat[i, :], color=sns.xkcd_rgb[cc[-1]], lw=1.5)

axes.plot(df.index/3600, df['T_int'], color='k', lw=2, label='measured indoor temperature °C')
axes.legend()
axes.set_xlabel('Time (hours)', fontsize=14)
axes.set_ylabel('Temperature (°C)', fontsize=14)
axes.set_title('Output predictive distribution', fontsize=14, fontweight='bold')
axes.tick_params(axis='both', which='major', labelsize=14)
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)

plt.savefig('mcmc_pred_armadillo', dpi = 1200)
