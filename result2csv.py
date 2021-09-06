import pandas as pd
import data_manager
fpath = 'models/'
need_to_change = 'dnn_7epoch'
fpath += need_to_change

ks200 = data_manager.capital.iloc[-1]
price = pd.read_csv('data/price_20210729.csv').T
portf_ratio = pd.read_csv(fpath+'/portf_ratio.csv').iloc[-2]
ret = pd.read_csv(fpath+'/test_result.csv')

portf_ratio.index = ['A' + i for i in portf_ratio.index]
price.index = portf_ratio.index
ks200.index = portf_ratio.index[1:]
portf_ratio = portf_ratio.dropna()[1:]
price = price.loc[portf_ratio.index]
ks200 = ks200[portf_ratio.index]

stock_name = []
code_df = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download', header=0)[0][['회사명', '종목코드']]
for i in price.index:
    stock_name.append(code_df.query(f'종목코드=={int(i[1:])}').values[0][0])

portf_ratio = portf_ratio.to_frame().reset_index()
portf_ratio.columns = ['ticker', 'portf_ratio']
portf_ratio['stock'] = stock_name
portf_ratio['ks200'] = ks200.values
portf_ratio['ret'] = (price[1].values - price[0].values) / price[0].values
portf_ratio = portf_ratio[['ticker', 'stock', 'portf_ratio', 'ks200', 'ret']].set_index('ticker')
portf_ratio.to_csv('result/dnn_7epoch.csv', encoding='cp949')

print(100 - ret.iloc[-2]['copy'])
print((portf_ratio['ret'] * portf_ratio['ks200']).sum())
print((portf_ratio['ret'] * portf_ratio['portf_ratio']).sum())