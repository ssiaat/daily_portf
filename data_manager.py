import pandas as pd
import numpy as np

# db연결
import pymysql
conn = pymysql.connect(host='localhost', user='root', password='0000', db='ticker', port=3306, charset='utf8')

curs = conn.cursor()

# ks200 데이터는 따로 받아옴
sql = f"SELECT * FROM ks200 ORDER BY ks200.date ASC;"
ks200 = pd.read_sql(sql=sql, con=conn)

# v4 version
COLUMNS_CHART_DATA = ['date', 'price_mod', 'open', 'high', 'low', 'close', 'volume']

COLUMNS_TRAINING_DATA_V1 = [
    'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
    'close_lastclose_ratio', 'volume_lastvolume_ratio',
    'close_ma5_ratio', 'volume_ma5_ratio',
    'close_ma10_ratio', 'volume_ma10_ratio',
    'close_ma20_ratio', 'volume_ma20_ratio',
    'close_ma60_ratio', 'volume_ma60_ratio',
    'close_ma120_ratio', 'volume_ma120_ratio',
]

COLUMNS_TRAINING_DATA_V1_RICH = [
    'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
    'close_lastclose_ratio', 'volume_lastvolume_ratio',
    'close_ma5_ratio', 'volume_ma5_ratio',
    'close_ma10_ratio', 'volume_ma10_ratio',
    'close_ma20_ratio', 'volume_ma20_ratio',
    'close_ma60_ratio', 'volume_ma60_ratio',
    'close_ma120_ratio', 'volume_ma120_ratio',
    'inst_lastinst_ratio', 'frgn_lastfrgn_ratio',
    'inst_ma5_ratio', 'frgn_ma5_ratio',
    'inst_ma10_ratio', 'frgn_ma10_ratio',
    'inst_ma20_ratio', 'frgn_ma20_ratio',
    'inst_ma60_ratio', 'frgn_ma60_ratio',
    'inst_ma120_ratio', 'frgn_ma120_ratio',
]

COLUMNS_TRAINING_DATA_V2 = [
    'per', 'pbr', 'roe',
    'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
    'close_lastclose_ratio', 'volume_lastvolume_ratio',
    'close_ma5_ratio', 'volume_ma5_ratio',
    'close_ma10_ratio', 'volume_ma10_ratio',
    'close_ma20_ratio', 'volume_ma20_ratio',
    'close_ma60_ratio', 'volume_ma60_ratio',
    'close_ma120_ratio', 'volume_ma120_ratio',
    'market_kospi_ma5_ratio', 'market_kospi_ma20_ratio',
    'market_kospi_ma60_ratio', 'market_kospi_ma120_ratio',
    'bond_k3y_ma5_ratio', 'bond_k3y_ma20_ratio',
    'bond_k3y_ma60_ratio', 'bond_k3y_ma120_ratio'
]

numsearch_on = 0

if numsearch_on == 1:
    # 'netbuy_foreign',
    COLUMNS_TRAINING_DATA_V3 = [
        'volume', 'num_search', 'cap', 'foreigner_rate', 'netbuy_institution', 'netbuy_foreigner', 'ks200', 'target'
    ]
elif numsearch_on == 0:
    COLUMNS_TRAINING_DATA_V3 = [
        'volume', 'cap', 'foreigner_rate', 'netbuy_institution', 'netbuy_foreigner', 'ks200', 'target'
    ]

COLUMNS_TRAINING_DATA_V4 = [
    'volume', 'num_search', 'short_amount', 'cap', 'foreigner_rate', 'netbuy_institution', 'netbuy_foreigner', 'ks200',
    'target'
]

# v3에서만 실행
# parameter 초기화를 he_normal
# input의 범위도 비슷하게 맞춰줌
def preprocessing(data):
    for col in COLUMNS_TRAINING_DATA_V3:
        max_num = data[col].max()
        min_num = data[col].min()
        data[col] = (data[col] - min_num) / (max_num - min_num)
    return data

def transformation(data, ver='v1'):
    windows = [5, 10, 20, 60, 120]
    for window in windows:
        data['close_ma{}'.format(window)] = \
            data['close'].rolling(window).mean()
        data['volume_ma{}'.format(window)] = \
            data['volume'].rolling(window).mean()
        data['close_ma%d_ratio' % window] = \
            (data['close'] - data['close_ma%d' % window]) \
            / data['close_ma%d' % window]
        data['volume_ma%d_ratio' % window] = \
            (data['volume'] - data['volume_ma%d' % window]) \
            / data['volume_ma%d' % window]

        if ver == 'v1.rich':
            data['inst_ma{}'.format(window)] = \
                data['close'].rolling(window).mean()
            data['frgn_ma{}'.format(window)] = \
                data['volume'].rolling(window).mean()
            data['inst_ma%d_ratio' % window] = \
                (data['close'] - data['inst_ma%d' % window]) \
                / data['inst_ma%d' % window]
            data['frgn_ma%d_ratio' % window] = \
                (data['volume'] - data['frgn_ma%d' % window]) \
                / data['frgn_ma%d' % window]

    data['open_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'open_lastclose_ratio'] = \
        (data['open'][1:].values - data['close'][:-1].values) \
        / data['close'][:-1].values
    data['high_close_ratio'] = \
        (data['high'].values - data['close'].values) \
        / data['close'].values
    data['low_close_ratio'] = \
        (data['low'].values - data['close'].values) \
        / data['close'].values
    data['close_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'close_lastclose_ratio'] = \
        (data['close'][1:].values - data['close'][:-1].values) \
        / data['close'][:-1].values
    data['volume_lastvolume_ratio'] = np.zeros(len(data))
    data.loc[1:, 'volume_lastvolume_ratio'] = \
        (data['volume'][1:].values - data['volume'][:-1].values) \
        / data['volume'][:-1] \
            .replace(to_replace=0, method='ffill') \
            .replace(to_replace=0, method='bfill').values

    if ver == 'v1.rich':
        data['inst_lastinst_ratio'] = np.zeros(len(data))
        data.loc[1:, 'inst_lastinst_ratio'] = \
            (data['inst'][1:].values - data['inst'][:-1].values) \
            / data['inst'][:-1] \
                .replace(to_replace=0, method='ffill') \
                .replace(to_replace=0, method='bfill').values
        data['frgn_lastfrgn_ratio'] = np.zeros(len(data))
        data.loc[1:, 'frgn_lastfrgn_ratio'] = \
            (data['frgn'][1:].values - data['frgn'][:-1].values) \
            / data['frgn'][:-1] \
                .replace(to_replace=0, method='ffill') \
                .replace(to_replace=0, method='bfill').values

    return data


## 기본 ver = v2
def load_data(fpath, date_from, date_to, ver='v3'):
    header = None if ver == 'v1' else 0
    data = pd.read_csv(fpath, thousands=',', header=header,
                       converters={'date': lambda x: str(x)})

    if ver == 'v1':
        data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    # 날짜 오름차순 정렬
    data = data.sort_values(by='date').reset_index()

    # 데이터 전처리
    data = transformation(data)

    # 기간 필터링
    data['date'] = data['date'].str.replace('-', '')
    data = data[(data['date'] >= date_from) & (data['date'] <= date_to)]
    data = data.dropna()

    # 차트 데이터 분리
    chart_data = data[COLUMNS_CHART_DATA]

    # 학습 데이터 분리
    training_data = None
    if ver == 'v1':
        training_data = data[COLUMNS_TRAINING_DATA_V1]
    elif ver == 'v1.rich':
        training_data = data[COLUMNS_TRAINING_DATA_V1_RICH]
    elif ver == 'v2':
        # v2 폴더내에서,  해당 종목파일명의 csv 파일에서 ,  선택한 Feature. 만 전처리
        # data.loc[:, ['per', 'pbr', 'roe']] = \
        #     data[['per', 'pbr', 'roe']].apply(lambda x: x / 100)
        training_data = data[COLUMNS_TRAINING_DATA_V3]
        training_data = training_data.apply(np.tanh)
    elif ver == 'v3':
        # v2 폴더내에서,  해당 종목파일명의 csv 파일에서 ,  선택한 Feature. 만 전처리
        # data.loc[:, ['per', 'pbr', 'roe']] = \
        #     data[['per', 'pbr', 'roe']].apply(lambda x: x / 100)
        training_data = data[COLUMNS_TRAINING_DATA_V3]
        training_data = training_data.apply(np.tanh)
    elif ver == 'v4':
        # v2 폴더내에서,  해당 종목파일명의 csv 파일에서 ,  선택한 Feature. 만 전처리
        # data.loc[:, ['per', 'pbr', 'roe']] = \
        #     data[['per', 'pbr', 'roe']].apply(lambda x: x / 100)
        training_data = data[COLUMNS_TRAINING_DATA_V4]
        training_data = training_data.apply(np.tanh)

    else:
        raise Exception('Invalid version.')

    return chart_data, training_data


## 여기서 읽어온것 다 합쳐야함

def make_data(stock_codes, start_date, end_date):
    training_df = pd.DataFrame()
    price_df = pd.DataFrame()
    vol_df = pd.DataFrame()
    for stock_code in stock_codes:
        # From local db,. 한종목씩
        price_data, vol_data, training_data = load_data_sql(stock_code, start_date, end_date)  ## 인자 1 ## 인자 2, 3
        # columns
        # cols = [학습대상 특성들]
        # df_unit = df_unit[cols]
        training_df = pd.concat([training_df, training_data] , axis=1)  ## 옆으로 늘어뜨려붙이기
        price_df = pd.concat([price_df, price_data], axis=1)
        vol_df = pd.concat([vol_df, vol_data], axis=1)
    price_df.columns = stock_codes
    vol_df.columns = stock_codes
    vol_df = (vol_df.T / vol_df.sum(axis=1)).T

    return price_df, vol_df, training_df



## 기본 ver = v2
# load_data_sql 한종목을 읽어오는것.
def load_data_sql(fpath, date_from, date_to, ver='v3'):
    header = None if ver == 'v1' else 0
    # data = pd.read_csv(fpath, thousands=',', header=header,
    #                    converters={'date': lambda x: str(x)})
    # fpath는 stock_code 로 받음
    ##  테이블 명이    숫자인경우  :  ' 작은따옴표 아니라  ` (물결키) 로 감쌈
    sql = f"SELECT * FROM `{fpath}` ORDER BY `{fpath}`.date ASC;"
    data = pd.read_sql(sql=sql, con=conn)
    data['ks200'] = ks200.kospi.values


    # 데이터 전처리
    if ver!= 'v3':
        data = transformation(data)


    # 기간 필터링
    # date 만   string 으로   타입변환해야함
    data['date'] = data['date'].apply(lambda _: str(_))
    data['date'] = data['date'].str.replace('-', '')
    data['date'] = data['date'].str.split(' ').str[0]

    # processing nan
    data_del_na = data.set_index('date')['price_mod'].dropna().reset_index()
    data = data.set_index('date').loc[data_del_na.date].reset_index().fillna(0)
    data = data[(data['date'] >= date_from) & (data['date'] <= date_to)]

    # 차트 데이터 분리
    price_data = data[['date', 'price_mod']].set_index('date')
    vol_data = data[['date', 'volume']].set_index('date')

    # 학습 데이터 분리
    training_data = None
    if ver == 'v1':
        training_data = data[COLUMNS_TRAINING_DATA_V1]
    elif ver == 'v1.rich':
        training_data = data[COLUMNS_TRAINING_DATA_V1_RICH]
    elif ver == 'v2':
        # v2 폴더내에서,  해당 종목파일명의 csv 파일에서 ,  선택한 Feature. 만 전처리
        # data.loc[:, ['per', 'pbr', 'roe']] = \
        #     data[['per', 'pbr', 'roe']].apply(lambda x: x / 100)
        training_data = data[COLUMNS_TRAINING_DATA_V3]
        training_data = training_data.apply(np.tanh)
    elif ver == 'v3':
        # v2 폴더내에서,  해당 종목파일명의 csv 파일에서 ,  선택한 Feature. 만 전처리
        # data.loc[:, ['per', 'pbr', 'roe']] = \
        #     data[['per', 'pbr', 'roe']].apply(lambda x: x / 100)
        training_data = data[COLUMNS_TRAINING_DATA_V3]
        training_data = training_data.astype(float)
        training_data = preprocessing(training_data)

        training_data = training_data.apply(np.tanh)
    elif ver == 'v4':
        # v2 폴더내에서,  해당 종목파일명의 csv 파일에서 ,  선택한 Feature. 만 전처리
        # data.loc[:, ['per', 'pbr', 'roe']] = \
        #     data[['per', 'pbr', 'roe']].apply(lambda x: x / 100)
        training_data = data[COLUMNS_TRAINING_DATA_V4]
        training_data = training_data.astype(float)
        training_data = training_data.apply(np.tanh)

    else:
        raise Exception('Invalid version.')

    return price_data, vol_data, training_data


from sqlalchemy import create_engine

## 기본 ver = v2
def load_data_ec2(fpath, date_from, date_to, ver='v3'):
    header = None if ver == 'v1' else 0
    # data = pd.read_csv(fpath, thousands=',', header=header,
    #                    converters={'date': lambda x: str(x)})
    # fpath는 stock_code 로 받음
    ##  테이블 명이    숫자인경우  :  ' 작은따옴표 아니라  ` (물결키) 로 감쌈
    sql = f"SELECT * FROM rltrader_datas.`{fpath}` ORDER BY rltrader_datas.`{fpath}`.date ASC;"
    engine2 = create_engine("mysql+pymysql://root:" + "0000" + "@13.209.4.191:3306/rltrader_datas?charset=utf8",
                            encoding='utf-8')
    data = pd.read_sql(sql=sql, con=engine2)
    print('----------Load Data from EC2 ')

    if ver == 'v1':
        data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    # 날짜 오름차순 정렬
    data = data.sort_values(by='date').reset_index()

    # 데이터 전처리
    data = transformation(data, ver="v4")

    # 기간 필터링
    # date 만   string 으로   타입변환해야함
    data['date'] = data['date'].apply(lambda _: str(_))
    # print(data.dtypes)
    data['date'] = data['date'].str.replace('-', '')
    data['date'] = data['date'].str.split(' ').str[0]

    print(f'data _ Before data split : {data}')
    data = data[(data['date'] >= date_from) & (data['date'] <= date_to)]
    print(f' date_from : {date_from}\n date_to :{date_to}\n')
    # data = data.dropna()
    print(f'data _ After data split : {data}')

    # 차트 데이터 분리
    chart_data = data[COLUMNS_CHART_DATA]

    buy_hold_return = (chart_data.close.pct_change() + 1).fillna(1).cumprod() - 1
    try:
        print(f'\n\n\n\n--------------------------------------------------------------- : {data["Ticker"].iloc[1]}')
    except:
        print(f'--------------------------------------------------------------------Nope ')

    try:
        print(f'-----------------------------------------------Buy & Hold 100% : {100 * buy_hold_return.iloc[-1]} %')
    except:
        print('Nope')

    # 학습 데이터 분리
    training_data = None
    if ver == 'v1':
        training_data = data[COLUMNS_TRAINING_DATA_V1]
    elif ver == 'v1.rich':
        training_data = data[COLUMNS_TRAINING_DATA_V1_RICH]
    elif ver == 'v2':
        # v2 폴더내에서,  해당 종목파일명의 csv 파일에서 ,  선택한 Feature. 만 전처리
        # data.loc[:, ['per', 'pbr', 'roe']] = \
        #     data[['per', 'pbr', 'roe']].apply(lambda x: x / 100)
        training_data = data[COLUMNS_TRAINING_DATA_V3]
        training_data = training_data.apply(np.tanh)
    elif ver == 'v3':
        # v2 폴더내에서,  해당 종목파일명의 csv 파일에서 ,  선택한 Feature. 만 전처리
        # data.loc[:, ['per', 'pbr', 'roe']] = \
        #     data[['per', 'pbr', 'roe']].apply(lambda x: x / 100)
        training_data = data[COLUMNS_TRAINING_DATA_V3]
        training_data = training_data.astype(float)
        training_data = training_data.apply(np.tanh)
    elif ver == 'v4':
        # v2 폴더내에서,  해당 종목파일명의 csv 파일에서 ,  선택한 Feature. 만 전처리
        # data.loc[:, ['per', 'pbr', 'roe']] = \
        #     data[['per', 'pbr', 'roe']].apply(lambda x: x / 100)
        training_data = data[COLUMNS_TRAINING_DATA_V4]
        training_data = training_data.astype(float)
        training_data = training_data.apply(np.tanh)

    else:
        raise Exception('Invalid version.')

    return chart_data, training_data

