"""
Test with both ventilation vent_H, vent2_H
add scaling for the ventilation
try to add cellar or attic or both as a boundary zone
try to remove globe temperature

Model Selection
---------------

TwTi_RoRiAwAi:
  - Np: 9
  - fit: ok
  - log-lik fit: -4002.216554
  - log-lik pred: -2769.345141
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bopt.statespace.rc import TwTi_RoRiAwAi
from bopt.regressors import Regressor


# Load and prepare data
xl = pd.read_excel("data/twin houses/Twin_house_exp2_10min.xlsx")
xl = xl.drop([0]).replace('-', np.nan)
index = xl.apply(lambda x: x['Date'].replace(hour=x['Time'].hour,
                                             minute=x['Time'].minute), axis=1)
xl = xl.drop(columns=['Date', 'Time']).set_index(pd.DatetimeIndex(index))

start_fit, stop_fit = '2014-04-29 01:00:00', '2014-05-14 01:00:00'
start_valid, stop_valid = '2014-05-14 01:00:00', '2014-05-23 17:00:00'
df = xl[start_fit:stop_valid].copy()

north_T_header = ['kitchen_AT', 'doorway_AT', 'parents_AT']
south_T_header = ['living_h010cm_AT', 'living_h110cm_AT', 'living_h170cm_AT',
                  'corridor_AT', 'bath_h010_AT', 'bath_h110_AT',
                  'bath_h170_AT', 'child_h010_AT', 'child_h110_AT',
                  'child_h170_AT']
south_H_header = ['heat_elP_living_room', 'heat_elP_bath_room',
                  'heat_elP_children_room']
attic_T = ['attic_west_h010_AT', 'attic_west_h110_AT', 'attic_west_h170_AT',
           'attic_east_h010_AT', 'attic_east_h110_AT', 'attic_east_h170_AT']


def pca(X, nc=1):
    """pca reduction to `nc` components"""
    Xstd = (X - X.mean(axis=0)) / X.std(axis=0)
    w, v = np.linalg.eig(np.cov(Xstd.T))
    idx = w.argsort()[::-1]
    w, v = w[idx], v[:, idx]
    explained_variance = np.cumsum(w[:nc] / w.sum())
    print(f"explain {explained_variance.squeeze():.3f} % of the variance")
    return np.dot(X, v[:, :nc] / v[:, :nc].sum())


df['north_T'] = pca(df[north_T_header].values)
df['south_T'] = pca(df[south_T_header].values)
df['attic_T'] = pca(df[attic_T].values)
df['south_H'] = df[south_H_header].sum(axis=1)

df['vent_H'] = (1005.0 * 1.2
                * ((df['vent_SUA_VFR'] / 3600.0) * df['living_SUA_AT']
                    - (df['vent_EHA_VFR'] / 3600.0) * df['vent_EHA_AT']))

df['vent2_H'] = (1005.0 * 1.2
                 * ((df['vent_SUA_VFR'] / 3600.0) * df['vent_SUA_AT']
                    - (df['vent_EHA_VFR'] / 3600.0) * df['vent_EHA_AT']))

y0 = 0.29806018786231947
y0p = 23.53273693772878

parameters = [
    dict(name="Ro", value=1e-2, transform="log"),
    dict(name="Ri", value=1e-3, transform="log"),
    dict(name="Cw", value=1e-2, transform="log"),
    dict(name="Ci", value=1e-3, transform="log"),
    dict(name="Aw", value=1.0, transform="log"),
    dict(name="Ai", value=1.0, transform="log"),
    dict(name="sigw_w", value=0.1, transform="log"),
    dict(name="sigw_i", value=1e-8, transform="fixed"),
    dict(name="sigv", value=0.1, transform="log"),
    dict(name="x0_w", value=0.28, transform="log"),
    dict(name="x0_i", value=y0, transform="fixed"),
    dict(name="sigx0_w", value=1.0, transform="fixed"),
    dict(name="sigx0_i", value=1.0, transform="fixed")
]

reg = Regressor(TwTi_RoRiAwAi(parameters, hold_order='foh'))
inputs = ['Ambient temperature',
          'Solar radiation: global horizontal',
          'south_H']
outputs = 'south_T'

out = reg.fit(df=df[start_fit:stop_fit],
              inputs=inputs,
              outputs=outputs)

# compute log-likelihood on validation dataset
x, P = reg.estimate_states(df=df[start_fit:stop_fit],
                           inputs=inputs,
                           outputs=outputs)

x0v = x[-1, :, :]
P0v = np.linalg.cholesky(P[-1, :, :]).T

reg.eval_log_likelihood(df=df[start_valid:stop_valid],
                        inputs=inputs,
                        outputs=outputs,
                        x0=x0v,
                        P0=P0v)


y_mean, y_std = reg.predict(df=df[start_valid:stop_valid],
                            inputs=inputs,
                            x0=x0v,
                            P0=P0v)

dfp = df[start_valid:stop_valid].copy()
dfp['yp'] = y_mean
dfp['lower'] = y_mean - 1.96 * y_std
dfp['upper'] = y_mean + 1.96 * y_std

fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
axes.plot(dfp.index, dfp['south_T'], 'k', lw=2)
axes.plot(dfp.index, dfp['yp'], 'C0', lw=2)
axes.fill_between(dfp.index, dfp['lower'], dfp['upper'], color='C0', alpha=0.2)
plt.show()
