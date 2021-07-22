import numpy as np
import utils


class Agent:
    # 에이전트 상태가 구성하는 값 개수
    STATE_DIM = 2  # 주식 보유 비율, 포트폴리오 가치 비율

    # 매매 수수료 및 세금
    TRADING_CHARGE = 0.00015  # 거래 수수료 0.015%
    # TRADING_CHARGE = 0.00011  # 거래 수수료 0.011%
    # TRADING_CHARGE = 0  # 거래 수수료 미적용
    TRADING_TAX = 0.0025  # 거래세 0.25%
    # TRADING_TAX = 0  # 거래세 미적용

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
        # 포트폴리오 가치: balance + num_stocks * {현재 주식 가격}
        self.portfolio_value = 0 
        self.base_portfolio_value = 0  # 직전 학습 시점의 PV
        # 종목마다 포트폴리오 가치 (balance고려 안됨)
        self.portfolio_value_each = np.zeros((self.num_ticker,))
        self.base_portfolio_value_each = np.zeros((self.num_ticker,))

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
        self.base_portfolio_value = self.initial_balance
        self.portfolio_value_each = np.zeros((self.num_ticker,))
        self.base_portfolio_value_each = np.zeros((self.num_ticker,))

        self.num_buy = np.zeros((self.num_ticker,))
        self.num_sell = np.zeros((self.num_ticker,))
        self.num_hold = np.zeros((self.num_ticker,))
        self.immediate_reward = 0
        self.base_ks = self.environment.get_ks_to_reset()
        self.win_cnt = 0

    def set100(self, tensor):
        return tensor / tensor.sum()

    def set_balance(self, balance):
        self.initial_balance = balance

    def renewal_portfolio_ratio(self):
        if self.num_stocks.sum() > 0:
            curr_price = self.environment.get_price()
            self.portfolio_value_each = self.num_stocks * curr_price
            self.portfolio_ratio = self.set100(self.portfolio_value_each)
            self.portfolio_value = self.portfolio_value_each.sum() + self.balance

    def decide_action(self, pred_policy, epsilon):

        # 시총 가중으로 오늘 투자할 포트폴리오 비중 결정
        ratio = self.set100(pred_policy)

        # 이전 비중보다 커지면 매수, 작아지면 매도로 행동 결정
        action = np.where(ratio > self.portfolio_ratio, self.ACTION_BUY, self.ACTION_SELL)

        # 탐험 여부 결정
        exploration = [False] * self.num_ticker
        if np.random.rand() < epsilon:
            exploration = np.random.random((self.num_ticker,)) < epsilon
            random_action = [np.random.randint(0,2) for _ in range(self.num_ticker)]
            action = np.where(exploration==1, random_action, action)
            ratio = self.set100(np.where(exploration==1, ratio.mean()/2, ratio))

        # hold 여부 결정
        action = np.where(abs(ratio - self.portfolio_ratio) < self.hold_criter, self.ACTION_HOLD, action)
        ratio = self.set100(np.where(abs(ratio - self.portfolio_ratio) < self.hold_criter, self.portfolio_ratio, ratio))

        # 횟수 갱신
        self.num_buy += np.where(action==self.ACTION_BUY, 1, 0)
        self.num_sell += np.where(action==self.ACTION_SELL, 1, 0)
        self.num_hold += np.where(action == self.ACTION_HOLD, 1, 0)

        return action, ratio, exploration

    def decide_trading_unit(self, ratio, curr_price):
        sell_trading_unit = np.floor((self.portfolio_ratio - ratio).clip(0, 10) * self.portfolio_value_each / curr_price)
        
        sell_trading_value = curr_price * sell_trading_unit

        buy_trading_unit = np.floor((ratio - self.portfolio_ratio).clip(0, 10) * (sell_trading_value.sum() + self.balance) / curr_price)

        return buy_trading_unit, sell_trading_unit

    def act(self, ratio):
        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()
        
        # 새로운 가격으로 portfolio_value 갱신
        
        # 즉시 보상 초기화
        self.immediate_reward = 0

        buy_unit, sell_unit = self.decide_trading_unit(ratio, curr_price)
        self.num_stocks += buy_unit - sell_unit
        self.balance = (sell_unit * curr_price).sum() + self.balance - (buy_unit * curr_price).sum()

        # 포트폴리오 가치 갱신
        self.renewal_portfolio_ratio()

        # ks200 대비 수익률로 보상 결정
        ks_now = self.environment.get_ks()
        ks_ret = (ks_now - self.base_ks) / self.base_ks
        self.profitloss = ((self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value) - ks_ret

        if self.profitloss > 0:
            self.win_cnt += 1

        # 즉시 보상 - ks200 대비 아웃퍼폼 / 값이 너무 작아질까 우려되어 * 10
        self.immediate_reward = self.profitloss * self.portfolio_ratio

        # 지연 보상 - 익절, 손절 기준
        delayed_reward = 0
        if self.profitloss > self.delayed_reward_threshold or self.profitloss < -self.delayed_reward_threshold:
            # 목표 수익률 달성하여 기준 포트폴리오 가치 갱신
            # 또는 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
            self.base_ks = ks_now
            delayed_reward = self.immediate_reward
        else:
            delayed_reward = np.zeros((self.num_ticker,))

        return self.immediate_reward, delayed_reward
