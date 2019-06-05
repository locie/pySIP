"""
Test with both ventilation vent_H, vent2_H
add scaling for the ventilation
try to add cellar or attic or both as a boundary zone
try to remove globe temperature

Model Selection
---------------

TwTi_RoRiRsiAwAich:
  - Np:
  - fit: not ok, pvalue, Rsi, Aw, ch
  - log-lik fit:
  - log-lik pred:

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bopt.statespace.rc.twti_rorirsiawaich import TwTi_RoRiRsiAwAich
from bopt.regressors import Regressor


# Load data
xl = pd.read_excel("data/twin houses/Twin_house_exp2_10min.xlsx")

# Remove first line (unit informations); Deal with NaN values
xl = xl.drop([0]).replace('-', np.nan)

# Remove Date and Time columns and set index
index = xl.apply(lambda x: x['Date'].replace(hour=x['Time'].hour,
                                             minute=x['Time'].minute), axis=1)
xl = xl.drop(columns=['Date', 'Time']).set_index(pd.DatetimeIndex(index))

start_fit, stop_fit = '2014-04-29 01:00:00', '2014-05-14 01:00:00'
start_valid, stop_valid = '2014-05-14 01:00:00', '2014-05-23 17:00:00'

df = xl[start_fit:stop_valid].copy()

north_T_header = ['kitchen_AT', 'doorway_AT', 'parents_AT']
north_H_header = ['heat_elP_kitchen', 'heat_elP_doorway',
                  'heat_elP_parents_room']

south_T_header = ['living_h010cm_AT', 'living_h110cm_AT', 'living_h170cm_AT',
                  'living_GT', 'corridor_AT', 'bath_h010_AT', 'bath_h110_AT',
                  'bath_h170_AT', 'child_h010_AT', 'child_h110_AT',
                  'child_h170_AT']
south_H_header = ['heat_elP_living_room', 'heat_elP_bath_room',
                  'heat_elP_children_room']

sr_horizontal = ['Solar radiation: global horizontal',
                 'Solar radiation: diffuse horizontal']
sr_vertical = ['Solar radiation: north vertical', 'Solar radiation: east vertical',
               'Solar radiation: south vertical', 'Solar radiation: west vertical']
solar_radiation = sr_horizontal + sr_vertical
solar_radiation.append('long-wave downward radiation ')

cellar_T = 'cellar_AT'
attic_T = ['attic_west_h010_AT', 'attic_west_h110_AT', 'attic_west_h170_AT',
           'attic_east_h010_AT', 'attic_east_h110_AT', 'attic_east_h170_AT']

# south_T.remove('living_GT')


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
df['north_H'] = df[north_H_header].sum(axis=1)

df['south_T'] = pca(df[south_T_header].values)
df['south_H'] = df[south_H_header].sum(axis=1)

df['attic_T'] = pca(df[attic_T].values)

df['vent_H'] = (1005.0 * 1.2
                * ((df['vent_SUA_VFR'] / 3600.0) * df['living_SUA_AT']
                    - (df['vent_EHA_VFR'] / 3600.0) * df['vent_EHA_AT']))

df['vent2_H'] = (1005.0 * 1.2
                 * ((df['vent_SUA_VFR'] / 3600.0) * df['vent_SUA_AT']
                    - (df['vent_EHA_VFR'] / 3600.0) * df['vent_EHA_AT']))


y = df['south_T'][start_fit:stop_fit].values[np.newaxis, :]
y0 = y[0, 0] / 100.0
u = df[['Ambient temperature', 'Solar radiation: global horizontal',
        'south_H']][start_fit:stop_fit].values.T

dt = 600
t = np.arange(0, dt * y.shape[1], dt)

reg = Regressor(TwTi_RoRiRsiAwAich())

reg.ss.parameters.set_parameter("Ro", value=1e-2, transform="log")
reg.ss.parameters.set_parameter("Ri", value=1e-3, transform="log")
reg.ss.parameters.set_parameter("Rsi", value=1e-3, transform="log")
reg.ss.parameters.set_parameter("Cw", value=1e-2, transform="log")
reg.ss.parameters.set_parameter("Ci", value=1e-3, transform="log")
reg.ss.parameters.set_parameter("Aw", value=1.0, transform="log")
reg.ss.parameters.set_parameter("Ai", value=1.0, transform="log")
reg.ss.parameters.set_parameter("ch", value=0.5, transform="log")
reg.ss.parameters.set_parameter("sigw_w", value=0.1, transform="log")
reg.ss.parameters.set_parameter("sigw_i", value=1e-8, transform="fixed")
reg.ss.parameters.set_parameter("sigv", value=0.1, transform="log")
reg.ss.parameters.set_parameter("x0_w", value=0.28, transform="log")
reg.ss.parameters.set_parameter("x0_i", value=y0, transform="fixed")
reg.ss.parameters.set_parameter("sigx0_w", value=1.0, transform="fixed")
reg.ss.parameters.set_parameter("sigx0_i", value=1.0, transform="fixed")

out = reg.fit(t, y, u, 'foh')

# compute log-likelihood on validation dataset
yp = df['south_T'][start_valid:stop_valid].values[np.newaxis, :]
tp = np.arange(0, dt * yp.shape[1], dt)
reg.ss.parameters.set_parameter("x0_i", value=y[0, -1] / 100, transform="fixed")
up = df[['Ambient temperature', 'Solar radiation: global horizontal',
         'south_H']][start_valid:stop_valid].values.T
reg.eval_log_likelihood(tp, yp, up, 'foh')

reg.ss.parameters.set_parameter("x0_i", value=y0, transform="fixed")
upred = df[['Ambient temperature', 'Solar radiation: global horizontal',
            'south_H']][start_fit:stop_valid].values.T
df["south_T_pred"] = df["south_T"].copy()
df["south_T_pred"][start_valid:stop_valid] = np.nan
tpred = np.arange(0, dt * upred.shape[1], dt)
ypred = df["south_T_pred"].values[np.newaxis, :]

y_mean, y_std = reg.predict(tpred, tpred, ypred, upred, 'foh', smooth=False)

plt.close("all")
fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)

ytruth = df["south_T"][start_fit:stop_valid].values
axes.plot(tpred, ytruth, 'k')
axes.plot(tpred, y_mean, 'C0', lw=2)
axes.fill_between(tpred,
                  y_mean - 1.96 * y_std,
                  y_mean + 1.96 * y_std,
                  color='C0', alpha=0.2)
plt.show()
