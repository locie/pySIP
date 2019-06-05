"""Armadillo Box INES frequentist identification"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bopt.statespace.rc import TwTi_RoRi
from bopt.regressors import Regressor

df = pd.read_csv('data/armadillo/armadillo_data_H2.csv').set_index('Time')

y0 = 0.2670106194217502
parameters = [
    dict(name="Ro", value=0.1, transform="log"),
    dict(name="Ri", value=0.01, transform="log"),
    dict(name="Cw", value=0.1, transform="log"),
    dict(name="Ci", value=0.01, transform="log"),
    dict(name="sigw_w", value=0.01, transform="log"),
    dict(name="sigw_i", value=0.0, transform="fixed"),
    dict(name="sigv", value=0.01, transform="log"),
    dict(name="x0_w", value=0.25, transform="log"),
    dict(name="x0_i", value=y0, transform="fixed"),
    dict(name="sigx0_w", value=1.0, transform="fixed"),
    dict(name="sigx0_i", value=1.0, transform="fixed")
]

reg = Regressor(TwTi_RoRi(parameters, hold_order='foh'))
out = reg.fit(df=df, inputs=['T_ext', 'P_hea'], outputs='T_int')

dt = 60
tnew = np.arange(df.index[0], df.index[-1] + dt, dt)
y_mean_f, y_std_f = reg.predict(df=df,
                                inputs=['T_ext', 'P_hea'],
                                tpred=tnew)

y_mean_s, y_std_s = reg.predict(df=df,
                                outputs='T_int',
                                inputs=['T_ext', 'P_hea'],
                                tpred=tnew,
                                smooth=True)

# plot filtered and smoothed output
plt.close("all")
fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
fig.suptitle("filtered vs. smoothed state ")

axes.plot(df.index, df['T_int'], 'kx')
axes.plot(tnew, y_mean_f, 'C0', lw=2)
axes.fill_between(tnew,
                  y_mean_f - 1.96 * y_std_f,
                  y_mean_f + 1.96 * y_std_f,
                  color='C0', alpha=0.2)

axes.plot(tnew, y_mean_s, 'C3', lw=2)
axes.fill_between(tnew,
                  y_mean_s - 1.96 * y_std_s,
                  y_mean_s + 1.96 * y_std_s,
                  color='C3', alpha=0.2)

plt.show()
