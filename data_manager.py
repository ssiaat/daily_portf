import pandas as pd
import numpy as np
from tqdm import tqdm

# db연결
import pymysql
conn = pymysql.connect(host='localhost', user='root', password='0000', db='ticker', port=3306, charset='utf8')

curs = conn.cursor()

# ks200 데이터는 따로 받아옴
sql = f"SELECT * FROM indexes ORDER BY indexes.date ASC;"
indexes = pd.read_sql(sql=sql, con=conn).set_index('date')
capital = pd.read_csv('data/capital.csv', index_col='date', parse_dates=True)

COLUMNS_CHART_DATA = ['date', 'price_mod', 'open', 'high', 'low', 'close', 'cap']
COLUMNS_TRAINING_DATA = ['open', 'high', 'low', 'close', 'price_mod', 'volume', 'cap', 'foreigner_rate', 'netbuy_institution', 'netbuy_foreigner', 'netbuy_individual', 'trs_amount']

def set_rebalance_date(start_year, end_year):
    index_temp = indexes.copy()
    index_temp['date_temp'] = index_temp.index
    date_df = index_temp.groupby(index_temp.index.year).last()
    date_df_needs = date_df[(date_df.index >= start_year) & (date_df.index < end_year + 1)]
    return date_df_needs.date_temp.values

# 특정 시점 기준 시총 상위 n개 ticker 가져옴
def get_stock_codes_yearly(n, criterion_date):
    last_idx = list(capital.index).index(criterion_date)
    stock_codes_last = capital.iloc[last_idx].dropna().sort_values(ascending=False).index
    start_idx = last_idx - 300 if last_idx >= 300 else 0
    stock_codes_start = capital.iloc[start_idx].dropna().index
    stock_codes = [x for x in stock_codes_last if x in stock_codes_start][:n]
    stock_codes = [i[1:] for i in stock_codes]
    return stock_codes

# 모든 시점에 대해서 ticker 가져옴 (get_stock_codes_year사용)
# ret: 연마다 처리된 stock codes, 전체 stock codes
def get_stock_codes(n, rebalance_date):
    stock_codes_yearly = []
    stock_codes = set()
    for rd in rebalance_date:
        temp = get_stock_codes_yearly(n, rd)
        stock_codes_yearly.append(temp)
        stock_codes.update(temp)
    return stock_codes_yearly, list(stock_codes)

# parameter 초기화를 he_normal
# input의 범위도 비슷하게 맞춰줌
def preprocessing(data, start_idx, end_idx, test=False):
    if 'close' in data.columns:
        data['open'] = data['open'] / data['close'] - 1
        data['high'] = data['high'] / data['close'] - 1
        data['low'] = data['low'] / data['close'] - 1
    for col in data.columns:
        max_num = data.iloc[start_idx:end_idx+1][col].max()
        min_num = data.iloc[start_idx:end_idx+1][col].min()
        if test:
            max_num = data.iloc[:start_idx][col].max()
            min_num = data.iloc[:start_idx][col].min()
        data[col] = (data[col] - min_num) / (max_num - min_num)
    return data

def get_weights_FFD(d, thres):
    w, k = [1.], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k+=1
    return np.array(w[::-1]).reshape(-1,)

w = get_weights_FFD(0.6, 1e-4)


def make_data(stock_codes, start_date, end_date, stationary, test):
    global indexes
    start_idx = list(indexes.index).index(start_date)
    if stationary:
        start_idx += len(w) - 1
    end_idx = list(indexes.index).index(end_date)
    date_idx = list(indexes.index)[start_idx:end_idx + 1]

    training_data_list = []
    training_data_idx = []
    price_df = pd.DataFrame(index=date_idx)
    cap_df = pd.DataFrame(index=date_idx)

    for stock_code in tqdm(stock_codes):
        # From local db,. 한종목씩
        price_data, cap_data, training_data = load_data_sql(stock_code, date_idx, start_idx, end_idx, stationary, test)
        training_data_idx.append(stock_code)
        training_data_list.append(training_data)
        price_df = pd.concat([price_df, price_data], axis=1)
        cap_df = pd.concat([cap_df, cap_data], axis=1)

    # training_df 는 3차원으로 설정
    # date -> stock code -> data
    training_df = pd.concat(training_data_list, keys=training_data_idx)
    training_df = training_df.swaplevel(0,1).sort_index(level=0)

    index_c = indexes.copy().ffill()
    indexe_ppc = preprocessing(index_c, start_idx, end_idx, test)
    if stationary:
        indexe_ppc = indexe_ppc.rolling(len(w)).apply(lambda x: (x*w).sum()).dropna()

    return price_df.fillna(0), cap_df.fillna(0), indexes.loc[price_df.index], indexe_ppc.loc[price_df.index], training_df.fillna(0)


# load_data_sql 한종목을 읽어오는것.
def load_data_sql(fpath, date_idx, start_idx, end_idx, stationary, test):
    # fpath는 stock_code 로 받음
    sql = f"SELECT * FROM `{fpath}` ORDER BY `{fpath}`.date ASC;"
    data = pd.read_sql(sql=sql, con=conn, parse_dates=True)

    data_del_na = data.set_index('date')['price_mod'].dropna().reset_index()
    data = data.set_index('date').loc[data_del_na.date]

    # 학습 데이터 분리, 전처리
    training_data = preprocessing(data.copy()[COLUMNS_TRAINING_DATA], start_idx, end_idx, test)
    if stationary:
        training_data = training_data.rolling(len(w)).apply(lambda x: (x*w).sum()).dropna()

    # index 조정
    temp = pd.DataFrame(index=date_idx)
    data = pd.merge(temp, data, how='left', left_on=temp.index, right_on=data.index).set_index('key_0')
    training_data = pd.merge(temp, training_data, how='left', left_on=temp.index, right_on=training_data.index).set_index('key_0')

    # 차트 데이터 분리
    price_data = data['price_mod']
    price_data.name = fpath
    cap_data = data['cap']
    cap_data.name = fpath

    # date 처리
    return price_data, cap_data, training_data
