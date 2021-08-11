import data_manager
import matplotlib.pyplot as plt
import numpy as np

# from statsmodels.tsa.stattools import adfuller
import pandas as pd
# import os
# import tensorflow as tf
# out = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95%conf', 'corr'])
# df = data_manager.ks200
# for d in np.linspace(0,1,11):
#     df1 = np.log(df)
#     df2 = data_manager.fracDiff_FFD(df1,d,0.01)
#     corr = np.corrcoef(df1.loc[df2.index], df2)[0,1]
#     df2 = adfuller(df2, maxlag=1, regression='c', autolag=None)
#     out.loc[d] = list(df2[:4]) + [df2[4]['5%']] + [corr]
# out[['adfStat', 'corr']].plot(secondary_y='adfStat')
# plt.axhline(out['95%conf'].mean(), linewidth=1, color='r', linestyle='dotted')
# df = (df - df.min()) / (df.max() - df.min())
# plt.plot(df)
# print(list(df.index).index(pd.Timestamp('20210630')))
# from datetime import datetime

w = data_manager.get_weights_FFD(0.7, 1e-5)
fpath = '005930'
sql = f"SELECT * FROM `{fpath}` ORDER BY `{fpath}`.date ASC;"
data = pd.read_sql(sql=sql, con=data_manager.conn, parse_dates=True)

data_del_na = data.set_index('date')['price_mod'].dropna().reset_index()
data = data.set_index('date').loc[data_del_na.date]
data['price_mod_temp'] = data['price_mod'].copy()
# data = data_manager.preprocessing(data.copy()[data_manager.COLUMNS_TRAINING_DATA], 0, len(data_manager.indexes)-1, False)
data = data.rolling(len(w)).apply(lambda x: (x*w).sum()).dropna()
data = data_manager.preprocessing(data.copy()[data_manager.COLUMNS_TRAINING_DATA], 0, len(data)-1, False)

print(data['price_mod'])
plt.plot(data['price_mod'])
plt.show()
