import data_manager
# import matplotlib.pyplot as plt
import numpy as np
#
# from statsmodels.tsa.stattools import adfuller
# import pandas as pd
# import os
# import tensorflow as tf
# out = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95%conf', 'corr'])
# fpath = '005930'
# sql = f"SELECT * FROM `{fpath}` ORDER BY `{fpath}`.date ASC;"
# df = pd.read_sql(sql=sql, con=data_manager.conn, parse_dates=True)['price_mod']
# for d in np.linspace(0,1,11):
#     df1 = np.log(df)
#     w = data_manager.get_weights_FFD(d, 1e-3)
#     df2_t = df.rolling(len(w)).apply(lambda x: (x*w).sum()).dropna()
#     corr = np.corrcoef(df1.loc[df2_t.index], df2_t)[0,1]
#     df2 = adfuller(df2_t, maxlag=1, regression='c', autolag=None)
#     out.loc[d] = list(df2[:4]) + [df2[4]['5%']] + [corr]
# out[['adfStat', 'corr']].plot(secondary_y='adfStat')
# plt.axhline(out['95%conf'].mean(), linewidth=1, color='r', linestyle='dotted')
# plt.show()



# print(df2_t)
# print(list(df.index).index(pd.Timestamp('20210630')))
from datetime import datetime

# w = data_manager.get_weights_FFD(0.3, 1e-3)
# fpath = '005930'
# sql = f"SELECT * FROM `{fpath}` ORDER BY `{fpath}`.date ASC;"
# data = pd.read_sql(sql=sql, con=data_manager.conn, parse_dates=True)
#
# data_del_na = data.set_index('date')['price_mod'].dropna().reset_index()
# data = data.set_index('date').loc[data_del_na.date]
# data['price_mod_temp'] = data['price_mod'].copy()
# # data = data_manager.preprocessing(data.copy()[data_manager.COLUMNS_TRAINING_DATA], 0, len(data_manager.indexes)-1, False)
# data_t = data.rolling(len(w)).apply(lambda x: (x*w).sum()).dropna()
# data_t = data_manager.preprocessing(data_t.copy()[data_manager.COLUMNS_TRAINING_DATA], 0, len(data_t)-1, False)
#
# for i in data.columns:
#     print(data_t[i])
#     plt.plot(data_t[i])
#     plt.show()
# for i in range(len(data_manager.capital.index)):
#     if len(data_manager.capital.iloc[i].dropna()) != 200:
#         print(i, len(data_manager.capital.iloc[i].dropna()))

print(len(data_manager.capital.iloc[0][:100].dropna()))