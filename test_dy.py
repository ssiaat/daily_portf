import data_manager
# import matplotlib.pyplot as plt
# import numpy as np
# from statsmodels.tsa.stattools import adfuller
import pandas as pd

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
a = pd.Timestamp('20151230')
b = pd.Timestamp('20201230')
n = 200
c = data_manager.capital
a_s = c.loc[a].sort_values(ascending=False).index[:n]
b_s = c.loc[b].sort_values(ascending=False).index[:n]
diff = list(set(a_s) - set(b_s))
print(len(diff))
# print(len(c[diff].loc[a].dropna()))
d = c.loc[b].rank(ascending=False)[diff]
print(len(d) - len(d.dropna()))
print(len(d[d>300]))