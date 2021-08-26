import numpy as np
import tensorflow as tf
from utils import softmax


class Agent:
    # 매매 수수료 및 세금
    TRADING_TAX = [0.001, 0.003]  # 거래세 매수, 매도

    # 시총 비중 대비 매수 매도 차이
    OVER_CAP = 0.05
    OVER_CAP_RANGE = 0.03

    # 행동
    ACTION_BUY = 2  # 매수
    ACTION_SELL = 1  # 매도
    ACTION_HOLD = 0  # 홀딩

    # 인공 신경망에서 확률을 구할 행동들
    ACTIONS = [ACTION_BUY, ACTION_SELL]
    NUM_ACTIONS = len(ACTIONS)  # 인공 신경망에서 고려할 출력값의 개수

    def __init__(
        self, environment=None, num_ticker=10, hold_criter=0., delayed_reward_threshold=.05):
        # Environment 객체
        # 현재 주식 가격을 가져오기 위해 환경 참조
        self.environment = environment
        self.num_ticker = num_ticker
        self.hold_criter = hold_criter
        # 지연보상 임계치
        self.delayed_reward_threshold = delayed_reward_threshold

        # Agent 클래스의 속성
        self.initial_balance = 0  # 초기 자본금
        self.balance = 0  # 현재 현금 잔고
        self.num_stocks = np.zeros((self.num_ticker,))  # 보유 주식 수
        self.portfolio_ratio = np.zeros((self.num_ticker,))
        self.last_portfolio_ratio = np.zeros((self.num_ticker,))

        # 포트폴리오 가치: balance + num_stocks * {현재 주식 가격}
        self.portfolio_value = 0 
        self.last_portfolio_value = 0  # 직전 학습 시점의 PV
        # 종목마다 포트폴리오 가치 (balance고려 안됨)
        self.portfolio_value_each = np.zeros((self.num_ticker,))

        self.num_buy = np.zeros((self.num_ticker,))  # 매수 횟수
        self.num_sell = np.zeros((self.num_ticker,))  # 매도 횟수
        self.num_hold = np.zeros((self.num_ticker,))  # 홀딩 횟수
        self.immediate_reward = 0  # 즉시 보상
        self.profitloss = 0
        self.last_profitloss = 0

        self.base_ks = self.environment.get_ks_to_reset()  # 기준 시점의 ks200지수, 초기값은 첫 ks200지수
        self.win_cnt = 0

    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = np.zeros((self.num_ticker,))
        self.portfolio_ratio = np.zeros((self.num_ticker,))
        self.portfolio_value = self.initial_balance
        self.portfolio_value_each = np.zeros((self.num_ticker,))
        self.last_portfolio_ratio = np.zeros((self.num_ticker,))
        self.last_portfolio_value = self.initial_balance
        self.last_profitloss = 0

        self.num_buy = np.zeros((self.num_ticker,))
        self.num_sell = np.zeros((self.num_ticker,))
        self.num_hold = np.zeros((self.num_ticker,))
        self.immediate_reward = 0

        self.base_ks = self.environment.get_ks_to_reset()
        self.win_cnt = 0

    def set_balance(self, balance):
        self.initial_balance = balance

    def set100(self, tensor):
        if tf.reduce_sum(tensor):
            return tensor / tf.reduce_sum(tensor)
        else:
            return tensor

    def penalty_diff_bm(self, ratio):
        curr_cap = self.environment.get_cap()
        curr_cap = self.set100(np.where(np.isnan(curr_cap) == True, 0., curr_cap))
        # ratio = tf.math.softmax(ratio)
        ratio = self.set100(tf.where(curr_cap == 0., 0., ratio))

        penalty = tf.math.abs(curr_cap - ratio) / 2.0 * 100.0
        print('{:.4f}' .format(tf.reduce_sum(penalty)))
        if tf.reduce_sum(penalty) < 30:
            penalty = 0.
        return ratio, penalty * -1.0

    def similar_with_cap(self, ratio):
        curr_cap = self.environment.get_cap()
        curr_cap = np.where(np.isnan(curr_cap), 0., curr_cap)
        curr_price = self.environment.get_price()
        curr_cap = self.set100(np.where(curr_price == 0., 0., curr_cap)).numpy()
        ratio = tf.where(curr_cap == 0., 0, ratio).numpy() * self.OVER_CAP + curr_cap

        return tf.cast(self.set100(ratio), tf.float32)

    def renewal_portfolio_ratio(self, transaction, buy_value_each=None, diff_stock_idx=None):
        if tf.reduce_sum(self.num_stocks) > 0:
            curr_price = self.environment.get_price()
            if diff_stock_idx is not None:
                curr_price = self.environment.get_price_last_portf()

            # 상장폐지 처리, 다음 종가로 갱신할 때 가격, 비율 정보 모두 없으면 상폐로 간주하고 종목 가치만큼 차감
            if not transaction:
                curr_cap = self.environment.get_cap()
                cap_nan_check = tf.cast(~tf.math.is_nan(curr_cap), tf.float32)
                self.num_stocks = tf.where((curr_price + cap_nan_check) == 0., 0., self.num_stocks)
                
            self.portfolio_value_each = tf.cast(self.num_stocks * curr_price, tf.float32)
            if transaction:
                # 매도 수수료는 balance에 반영 -> 매수만 반영
                self.portfolio_value_each -= tf.math.ceil(buy_value_each * self.TRADING_TAX[0])
            self.portfolio_ratio = self.set100(self.portfolio_value_each)
            self.portfolio_value = tf.cast(tf.reduce_sum(self.portfolio_value_each), tf.float32) + self.balance


    def decide_action(self, ratio):
        # hold 여부 결정
        if self.hold_criter > 0.:
            ratio = self.set100(np.where(abs(ratio - self.portfolio_ratio) < self.hold_criter, self.portfolio_ratio, ratio))

        action = ratio - self.portfolio_ratio
        return action, ratio

    def decide_trading_unit(self, ratio, diff_stocks_idx=None):
        curr_price = self.environment.get_price()
        if diff_stocks_idx is not None:
            curr_price = self.environment.get_price_last_portf()
        sell_trading_unit = tf.math.floor(tf.clip_by_value(self.portfolio_ratio - ratio, 0., 10.) *\
                                     self.portfolio_value_each / np.where(curr_price == 0., 1., curr_price))

        # 변경 종목 idx중 기존 종목 모두 매도
        if diff_stocks_idx is not None:
            if tf.is_tensor(sell_trading_unit):
                sell_trading_unit = tf.make_ndarray(tf.make_tensor_proto(sell_trading_unit))
            sell_trading_unit[diff_stocks_idx] = tf.gather(self.num_stocks, diff_stocks_idx)

        # 거래정지 상태인데 거래하는 경우 방지
        sell_trading_unit = tf.where(curr_price == 0., 0., sell_trading_unit)
        sell_trading_value = tf.math.floor(curr_price * sell_trading_unit * (1 - self.TRADING_TAX[1]))

        buy_trading_ratio = tf.clip_by_value(ratio - self.portfolio_ratio, 0., 10.)
        curr_price = self.environment.get_price()

        # 새로운 종목 모두 매수
        if diff_stocks_idx is not None:
            if tf.is_tensor(buy_trading_ratio):
                buy_trading_ratio = tf.make_ndarray(tf.make_tensor_proto(buy_trading_ratio))
            buy_trading_ratio[diff_stocks_idx] = tf.gather(ratio, diff_stocks_idx)

        # 거래정지 상태인데 거래하는 경우 방지
        buy_trading_unit = tf.math.floor(self.set100(tf.where(curr_price == 0., 0., buy_trading_ratio)) * \
                                    (tf.reduce_sum(sell_trading_value) + self.balance) / np.where(curr_price == 0., 1., curr_price))
        return buy_trading_unit, sell_trading_unit

    def get_reward(self):
        # ks200 대비 수익률로 보상 결정
        ks_now = self.environment.get_ks()
        ks_ret = (ks_now - self.base_ks) / self.base_ks
        self.profitloss = ((self.portfolio_value - self.initial_balance) / self.initial_balance) - ks_ret

        if self.profitloss > 0:
            self.win_cnt += 1
        # self.portfolio_ratio = tf.cast(self.portfolio_ratio, tf.float32)
        # 즉시 보상 - ks200 대비 아웃퍼폼, 기준 시점 대비 변화가 클수록 기여도 큰 것으로 적용
        self.immediate_reward = (self.profitloss - self.last_profitloss) * tf.abs(self.portfolio_ratio - self.last_portfolio_ratio) * 100000

        return self.immediate_reward

    def act(self, ratio, diff_stocks_idx=None):
        curr_price = self.environment.get_price()

        # 거래 수량, 금액 결정
        buy_unit, sell_unit = self.decide_trading_unit(ratio, diff_stocks_idx)
        buy_value_each = buy_unit * curr_price
        if diff_stocks_idx is None:
            sell_value_each = sell_unit * curr_price
        else:
            sell_value_each = sell_unit * self.environment.get_price_last_portf()
        self.num_stocks += buy_unit - sell_unit

        # 매도 수수료는 바로 반영, 매수 수수료는 장부가치에 반영
        buy_value = tf.cast(tf.reduce_sum(buy_value_each), dtype=tf.float32)
        sell_value = tf.cast(tf.reduce_sum(sell_value_each) * (1 - self.TRADING_TAX[1]), dtype=tf.float32)
        self.balance = sell_value + self.balance - buy_value

        self.last_portfolio_value = self.portfolio_value
        self.last_portfolio_ratio = self.portfolio_ratio
        self.last_profitloss = self.profitloss

        # 포트폴리오 가치 갱신, 거래세 반영
        self.renewal_portfolio_ratio(transaction=True, buy_value_each=buy_value_each)

        return


    def tf2np(self, tensor):
        return tf.make_ndarray(tf.make_tensor_proto(tensor))