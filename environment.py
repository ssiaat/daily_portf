import numpy as np
import pandas as pd


class Environment:

    def __init__(self, price_data=None, cap_data=None, index_data=None, index_ppc=None,
                 training_data=None, num_ticker=10, num_steps=5, num_features=12):
        self.price_data = price_data
        self.cap_data = cap_data
        self.index_data = index_data
        self.index_ppc = index_ppc
        self.ks_data = index_data.ks200
        self.training_data = training_data

        self.num_ticker = num_ticker
        self.num_steps = num_steps
        self.num_features = num_features

        self.observe_price = None
        self.observe_cap = None
        self.observe_ks = None
        self.observe_index = None
        self.idx = -1
        self.date_list = price_data.index  # 3차원 데이터를 date로 접근해야해서 필요
        self.date = None
        self.stock_codes_idx = 0

        self.year = self.date_list[0].year  # test환경에서 year값이 변하면 종목을 변경해줘야함

        self.universe = list(self.cap_data.iloc[0].dropna().index)[:self.num_ticker]
        self.last_universe = list(self.cap_data.iloc[0].dropna().index)[:self.num_ticker]
        self.universe_history = pd.DataFrame(columns=range(200))

    def reset(self):
        self.idx = -1
        self.date = None
        self.universe = list(self.cap_data.iloc[0].dropna().index)[:self.num_ticker]
        self.last_universe = list(self.cap_data.iloc[0].dropna().index)[:self.num_ticker]

    def observe(self):
        if len(self.price_data) > self.idx + self.num_steps:
            self.idx += 1
            self.date = self.date_list[self.idx + self.num_steps - 1]
            diff_stock_codes = self.update_stock_codes()
            self.observe_price = self.price_data.loc[self.date][self.universe]
            self.observe_cap = self.cap_data.loc[self.date][self.universe]
            self.observe_ks = self.ks_data.loc[self.date]
            return diff_stock_codes, self.idx
        return None, None

    # test시 1년이 지나면 stock code변경
    def update_stock_codes(self):
        self.universe_history.loc[self.idx] = self.universe
        self.last_universe = self.universe.copy()
        today_stock_codes = self.cap_data.loc[self.date].dropna().index[:self.num_ticker]
        diff_universe = [x for x in today_stock_codes if x not in self.last_universe]
        diff_universe_idx = 0
        ret = None
        if len(diff_universe) != 0:
            ret = []
            for i, x in enumerate(self.last_universe):
                if x not in today_stock_codes:
                    self.universe[i] = diff_universe[diff_universe_idx]
                    diff_universe_idx += 1
                    ret.append(i)
        return ret

    def get_price(self):
        if self.observe_price is not None:
            return self.observe_price.values.reshape(-1,)
        return None

    def get_last_price(self):
        return self.price_data.iloc[self.idx + self.num_steps - 2][self.universe].values.reshape(-1,)

    # 종목 변경 시 지난 포트폴리오의 현재 가격 얻어올 때 사용
    def get_price_last_portf(self):
        return self.price_data.iloc[self.idx + self.num_steps - 1][self.last_universe].values.reshape(-1,)

    def get_cap(self):
        if self.observe_cap is not None:
            return self.observe_cap.fillna(0).values.reshape(-1,).astype(np.float32)
        return None

    def get_ks(self):
        if self.observe_ks is not None:
            return self.observe_ks
        return None

    def get_ks_to_reset(self):
        return self.ks_data.iloc[0]

    def get_training_data(self, idx):
        universe = self.universe_history.iloc[idx].values
        date_idx = self.date_list[idx:idx + self.num_steps]
        sample = self.training_data.loc[date_idx].reset_index().drop('key_0', axis=1).set_index('level_1')
        sample = sample.loc[universe].values
        sample_index = self.index_ppc.loc[date_idx].values
        ks_portf = self.cap_data.loc[date_idx][universe].fillna(0).values
        return [sample, sample_index, [ks_portf[-1]]]

    def transform_sample(self, sample):
        sample = sample.reshape(self.num_ticker, self.num_steps, self.num_features)
        next_sample = np.split(sample, self.num_ticker)
        return next_sample

