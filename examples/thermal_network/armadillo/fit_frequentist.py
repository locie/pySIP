import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pysip.statespace import TwTi_RoRi
from pysip.regressors import FreqRegressor as Regressor

df = pd.read_csv('data/armadillo/armadillo_data_H2.csv').set_index('Time')
df.drop(df.index[-1], axis=0, inplace=True)
inputs = ['T_ext', 'P_hea']
outputs = 'T_int'
sT = 3600.0 * 24.0
df.index /= sT

parameters = [
    dict(name='Ro', value=1.0, scale=0.1, transform='log'),
    dict(name='Ri', value=1.0, scale=0.01, transform='log'),
    dict(name='Cw', value=1.0, scale=1e7 / sT, transform='log'),
    dict(name='Ci', value=1.0, scale=1e6 / sT, transform='log'),
    dict(name='sigw_w', value=1.0, scale=0.01 * sT ** 0.5, transform='log'),
    dict(name='sigw_i', value=0.0, transform='fixed'),
    dict(name='sigv', value=1.0, scale=0.01, transform='log'),
    dict(name='x0_w', value=1.0, scale=25.0, transform='log'),
    dict(name='x0_i', value=26.7, transform='fixed'),
    dict(name='sigx0_w', value=1.0, transform='fixed'),
    dict(name='sigx0_i', value=1.0, transform='fixed'),
]

reg = Regressor(TwTi_RoRi(parameters, hold_order=1))
out = reg.fit(df=df, inputs=inputs, outputs=outputs)

dt = 60 / 3600
tnew = np.arange(df.index[0], df.index[-1], dt)
ym, ysd = reg.predict(df=df, inputs=inputs, tnew=tnew)

plt.close('all')
sns.set_style('darkgrid')
sns.set_context('talk')

plt.plot(df.index, df['T_int'], 'k')
plt.plot(tnew, ym, 'navy', lw=2)
plt.fill_between(tnew, ym - 1.96 * ysd, ym + 1.96 * ysd, color='darkblue', alpha=0.2)
sns.despine()
plt.xlabel('time [days]')
plt.ylabel('temperature [°C]')
plt.tight_layout()
plt.show()
