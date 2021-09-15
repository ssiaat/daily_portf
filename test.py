import numpy as np
import pandas as pd
import data_manager

# price = pd.read_csv('data/price_mod.csv', index_col='date', parse_dates=True)
# capital = pd.read_csv('data/ks200_cap.csv', index_col='date', parse_dates=True)
# ret = (price.shift(1) / price)[250:]
# ret_ks = pd.DataFrame(index=price.index[250:], columns=range(200))
# universe = None
# last_universe = None
# for i in price.index[250:]:
#     ticker = capital.loc[i].dropna()[1:].sort_values().index[:200]
#     last_universe = universe
#     universe = ticker
#     ret_universe = universe
#     if last_universe is not None and set(universe) != set(last_universe):
#         ret_universe = last_universe
#     ret_value = list(ret.loc[i][ret_universe].values)[:200]
#     if len(ret_value) < 200:
#         temp = [0 for _ in range(200 - len(ret_value))]
#         ret_value.extend(temp)
#     ret_ks.loc[i] = ret_value
# h = ret_ks[range(100)].mean(axis=1)
# l = ret_ks[range(100,200)].mean(axis=1)
# years = range(2005,2022,2)
# last_idx = None
# a = l.cumprod()
# for i in a:
#     print(i)
import time
import matplotlib.pyplot as plt
import data_manager
a = pd.read_csv('test.csv', index_col='date')
print(len(data_manager.w))
for i in a.columns:
    print(i)
    plt.plot(a[i])
    plt.show()