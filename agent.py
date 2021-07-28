import numpy as np
import tensorflow as tf


class Agent:
    # 매매 수수료 및 세금
    TRADING_TAX = [0.001, 0.003]  # 거래세 매수, 매도

    # 시총 비중 대비 매수 매도 차이
    OVER_CAP = [0.001, 0.001]

    # 행동
    ACTION_BUY = 0  # 매수
    ACTION_SELL = 1  # 매도
    ACTION_HOLD = 2  # 홀딩

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
        self.base_portfolio_ratio = np.zeros((self.num_ticker,))

        # 포트폴리오 가치: balance + num_stocks * {현재 주식 가격}
        self.portfolio_value = 0 
        self.base_portfolio_value = 0  # 직전 학습 시점의 PV
        # 종목마다 포트폴리오 가치 (balance고려 안됨)
        self.portfolio_value_each = np.zeros((self.num_ticker,))

        self.num_buy = np.zeros((self.num_ticker,))  # 매수 횟수
        self.num_sell = np.zeros((self.num_ticker,))  # 매도 횟수
        self.num_hold = np.zeros((self.num_ticker,))  # 홀딩 횟수
        self.immediate_reward = 0  # 즉시 보상
        self.profitloss = 0  # 현재 손익

        self.base_ks = self.environment.get_ks_to_reset()  # 기준 시점의 ks200지수, 초기값은 첫 ks200지수
        self.win_cnt = 0

    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = np.zeros((self.num_ticker,))
        self.portfolio_ratio = np.zeros((self.num_ticker,))
        self.portfolio_value = self.initial_balance
        self.portfolio_value_each = np.zeros((self.num_ticker,))
        self.base_portfolio_ratio = np.zeros((self.num_ticker,))
        self.base_portfolio_value = self.initial_balance

        self.num_buy = np.zeros((self.num_ticker,))
        self.num_sell = np.zeros((self.num_ticker,))
        self.num_hold = np.zeros((self.num_ticker,))
        self.immediate_reward = 0

        self.base_ks = self.environment.get_ks_to_reset()
        self.win_cnt = 0

    def set_balance(self, balance):
        self.initial_balance = balance

    def set100(self, tensor):
        return tensor / tf.reduce_sum(tensor)

    def similar_with_cap(self, ratio):
        curr_cap = self.environment.get_cap()
        curr_price = self.environment.get_price()
        curr_cap = self.set100(tf.where(curr_price == 0., 0., curr_cap))
        ratio = self.set100(tf.clip_by_value(ratio, curr_cap - self.OVER_CAP[1], curr_cap + self.OVER_CAP[0]))
        return ratio

    def renewal_portfolio_ratio(self, transaction, buy_value_each=None):
        if tf.reduce_sum(self.num_stocks) > 0:
            curr_price = self.environment.get_price()
            self.portfolio_value_each = self.num_stocks * curr_price
            if transaction:
                # 매도 수수료는 balance에 반영
                self.portfolio_value_each -= buy_value_each * self.TRADING_TAX[0]
            self.portfolio_ratio = self.set100(self.portfolio_value_each)
            self.portfolio_value = tf.reduce_sum(self.portfolio_value_each) + self.balance

    def decide_action(self, ratio, curr_cap, epsilon):
        # 이전 비중보다 커지면 매수, 작아지면 매도로 행동 결정, 상한선 두기
        diff = tf.clip_by_value(abs(ratio - self.portfolio_ratio), 0, tf.reduce_mean(ratio) / 2)
        action = np.where(diff > 0, self.ACTION_BUY, self.ACTION_SELL)

        # 탐험 여부 결정
        exploration = [False] * self.num_ticker
        if np.random.rand() < 0.5:
            exploration = np.random.random((self.num_ticker,)) < epsilon
            random_action = np.random.random((self.num_ticker,)) < 0.5
            action = np.where(exploration == 1, random_action, action)

            # 매도 탐험의 하한선은 보유 비중 전체, 1은 명목상 존재
            ratio = np.clip(np.where((action == 1) & (exploration == 1), self.portfolio_ratio - diff, ratio), 0, 1)
            ratio = np.where((action == 0) & (exploration == 1), self.portfolio_ratio + diff, ratio)
            ratio = self.set100(ratio)

        # hold 여부 결정
        if self.hold_criter > 0.:
            action = np.where(abs(ratio - self.portfolio_ratio) < self.hold_criter, self.ACTION_HOLD, action)
            ratio = self.set100(np.where(abs(ratio - self.portfolio_ratio) < self.hold_criter, self.portfolio_ratio, ratio))

        ratio = self.similar_with_cap(ratio)

        # 횟수 갱신
        self.num_buy += np.where(action==self.ACTION_BUY, 1, 0)
        self.num_sell += np.where(action==self.ACTION_SELL, 1, 0)
        self.num_hold += np.where(action == self.ACTION_HOLD, 1, 0)

        return action, ratio, exploration

    def decide_trading_unit(self, ratio, curr_price):
        sell_trading_unit = tf.math.floor(tf.clip_by_value(self.portfolio_ratio - ratio, 0, 10) * \
                                     self.portfolio_value_each / np.where(curr_price == 0., 1., curr_price))
        # 거래정지 상태인데 거래하는 경우 방지
        sell_trading_unit = tf.where(curr_price == 0., 0., sell_trading_unit)
        sell_trading_value = curr_price * sell_trading_unit * (1 - self.TRADING_TAX[1])
        buy_trading_ratio = tf.clip_by_value(ratio - self.portfolio_ratio, 0, 10)
        # 거래정지 상태인데 거래하는 경우 방지
        buy_trading_unit = tf.math.floor(self.set100(tf.where(curr_price == 0., 0., buy_trading_ratio)) * \
                                    (tf.reduce_sum(sell_trading_value) + self.balance) / np.where(curr_price==0., 1., curr_price))

        return buy_trading_unit, sell_trading_unit

    def get_reward(self):
        # ks200 대비 수익률로 보상 결정
        ks_now = self.environment.get_ks()
        ks_ret = (ks_now - self.base_ks) / self.base_ks
        self.profitloss = ((self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value) - ks_ret

        if self.profitloss > 0:
            self.win_cnt += 1

        # 즉시 보상 - ks200 대비 아웃퍼폼, 기준 시점 대비 변화가 클수록 기여도 큰 것으로 적용
        self.immediate_reward = self.profitloss * tf.abs(self.portfolio_ratio - self.base_portfolio_ratio)

        # 지연 보상 - 익절, 손절 기준
        delayed_reward = 0
        if self.profitloss > self.delayed_reward_threshold or self.profitloss < -self.delayed_reward_threshold / 2:
            # 목표 수익률 달성하여 기준 포트폴리오 가치 갱신
            # 또는 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_ratio = self.portfolio_ratio
            self.base_portfolio_value = self.portfolio_value
            self.base_ks = ks_now
            delayed_reward = self.immediate_reward
        else:
            delayed_reward = np.zeros((self.num_ticker,), dtype='float')
        return self.immediate_reward, delayed_reward

    def act(self, ratio):
        curr_price = self.environment.get_price()
        ratio = self.similar_with_cap(ratio)

        # 거래 수량, 금액 결정
        buy_unit, sell_unit = self.decide_trading_unit(ratio, curr_price)
        buy_value_each = buy_unit * curr_price
        sell_value_each = sell_unit * curr_price
        self.num_stocks += buy_unit - sell_unit

        # 매도 수수료는 바로 반영, 매수 수수료는 장부가치에 반영
        buy_value = tf.reduce_sum(buy_value_each)
        sell_value = tf.reduce_sum(sell_value_each) * (1 - self.TRADING_TAX[1])
        self.balance = sell_value + self.balance - buy_value

        # 포트폴리오 가치 갱신, 거래세 반영
        self.renewal_portfolio_ratio(transaction=True, buy_value_each=buy_value_each)

    def tf2np(self, tensor):
        return tf.make_ndarray(tf.make_tensor_proto(tensor))