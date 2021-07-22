import os
import logging
import abc
import collections
import threading
import time
import numpy as np
from utils import sigmoid
from environment import Environment
from agent import Agent
from networks import Network, DNN
from visualizer import Visualizer


class ReinforcementLearner:
    __metaclass__ = abc.ABCMeta
    lock = threading.Lock()

    def __init__(self, rl_method='rl', trainable=False,
                price_data=None, vol_data=None, ks_data=None, training_data=None,
                hold_criter=0., delayed_reward_threshold=.05,
                num_ticker=100, num_features=7, num_steps=1, lr=0.001,
                value_network=None, policy_network=None,
                output_path='', reuse_models=True):
        # 인자 확인
        assert num_steps > 0
        assert lr > 0
        # 강화학습 기법 설정
        self.rl_method = rl_method
        # 학습여부 설정
        self.trainable = trainable
        # 환경 설정
        self.price_data = price_data
        self.environment = Environment(price_data, vol_data, ks_data)

        # 에이전트 설정
        self.agent = Agent(self.environment, num_ticker=num_ticker, hold_criter=hold_criter,
                    delayed_reward_threshold=delayed_reward_threshold)
        # 학습 데이터
        self.training_data = training_data
        self.total_len = len(training_data.index)  # 전체 학습 기간의 날짜 수
        self.sample = None
        self.training_data_idx = -1

        self.num_ticker = num_ticker
        self.num_features = num_features
        # 신경망 설정
        self.num_steps = num_steps
        self.lr = lr
        self.value_network = value_network
        self.policy_network = policy_network
        self.reuse_models = reuse_models

        # 메모리
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_learning_idx = []
        # 에포크 관련 정보
        self.loss = 0.
        self.itr_cnt = 0
        self.batch_size = 0
        self.learning_cnt = 0
        # 로그 등 출력 경로
        self.output_path = output_path

    def init_value_network(self, activation='linear', loss='mse'):
        self.value_network = DNN(
            input_dim=self.num_features,
            output_dim=self.agent.NUM_ACTIONS * self.num_ticker,
            num_ticker=self.num_ticker,
            trainable=self.trainable,
            lr=self.lr, activation=activation, loss=loss)
        if self.reuse_models and \
            os.path.exists(self.value_network_path):
                self.value_network.load_model(
                    model_path=self.value_network_path)

    def init_policy_network(self, activation='sigmoid', loss='binary_crossentropy'):
        self.policy_network = DNN(
            input_dim=self.num_features,
            output_dim=self.num_ticker,
            num_ticker=self.num_ticker,
            trainable=self.trainable,
            lr=self.lr, activation=activation, loss=loss)
        if self.reuse_models and \
            os.path.exists(self.policy_network_path):
            self.policy_network.load_model(
                model_path=self.policy_network_path)

    def reset(self):
        self.sample = None
        self.training_data_idx = -1
        # 환경 초기화
        self.environment.reset()
        # 에이전트 초기화
        self.agent.reset()

        # 메모리 초기화
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_learning_idx = []
        # 에포크 관련 정보 초기화
        self.loss = 0.
        self.itr_cnt = 0
        self.batch_size = 0
        self.learning_cnt = 0

    def build_sample(self):
        self.environment.observe()
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            self.sample = self.training_data.iloc[
                self.training_data_idx].values.reshape((-1,7)).tolist()
            return self.sample
        return None

    @abc.abstractmethod
    def get_batch(self, batch_size, delayed_reward, discount_factor):
        pass

    def update_networks(self, 
            batch_size, delayed_reward, discount_factor):
        # 배치 학습 데이터 생성
        x, y_value, y_policy = self.get_batch(
            batch_size, delayed_reward, discount_factor)
        if len(x) > 0:
            loss = 0
            if y_value is not None:
                # 가치 신경망 갱신
                loss += self.value_network.train_on_batch(x, y_value)
            if y_policy is not None:
                # 정책 신경망 갱신
                loss += self.policy_network.train_on_batch(x, y_policy)
            return loss
        return None

    def fit(self, delayed_reward, discount_factor, full=False):
        batch_size = len(self.memory_reward) if full \
            else self.batch_size
        # 배치 학습 데이터 생성 및 신경망 갱신
        if batch_size > 0:
            _loss = self.update_networks(
                batch_size, delayed_reward, discount_factor)
            if _loss is not None:
                self.loss += abs(_loss)
                self.learning_cnt += 1
                self.memory_learning_idx.append(self.training_data_idx)
            self.batch_size = 0

    def run(
        self, num_epoches=100, balance=10000000,
        discount_factor=0.9, start_epsilon=0.5, learning=True):
        info = "RL:{rl} LR:{lr} " \
            "DF:{discount_factor} DRT:{delayed_reward_threshold}".format(rl=self.rl_method,
            lr=self.lr, discount_factor=discount_factor,
            delayed_reward_threshold=self.agent.delayed_reward_threshold
        )
        with self.lock:
            logging.info(info)

        # 시작 시간
        time_start = time.time()

        # 에이전트 초기 자본금 설정
        self.agent.set_balance(balance)

        # 학습에 대한 정보 초기화
        max_portfolio_value = 0
        epoch_win_cnt = 0

        # 학습 반복
        for epoch in range(num_epoches):
            time_start_epoch = time.time()

            # step 샘플을 만들기 위한 큐
            q_sample = collections.deque(maxlen=self.num_steps*self.num_ticker)
            
            # 환경, 에이전트, 신경망, 가시화, 메모리 초기화
            self.reset()

            # 학습을 진행할 수록 탐험 비율 감소
            if learning:
                epsilon = start_epsilon \
                    * (1. - float(epoch) / (num_epoches - 1))
            else:
                epsilon = start_epsilon

            while True:
                # 샘플 생성
                next_sample = self.build_sample()
                if next_sample is None:
                    break

                # num_steps만큼 샘플 저장
                # for t in range(0, self.num_ticker * self.num_features, self.num_features):
                #     q_sample.append(np.array(next_sample[t, t+self.num_features]))
                for s in np.array(next_sample).reshape((self.num_ticker, -1, self.num_features)):
                    q_sample.append(s)
                if len(q_sample) < self.num_steps * self.num_ticker:
                    continue

                # 가치, 정책 신경망 예측
                pred_value = None
                pred_policy = None
                if self.value_network is not None:
                    pred_value = self.value_network.predict(list(q_sample))
                if self.policy_network is not None:
                    pred_policy = self.policy_network.predict(list(q_sample))

                # 포트폴리오 가치를 오늘 가격 반영해서 갱신
                self.agent.renewal_portfolio_ratio()

                # 신경망 또는 탐험에 의한 행동 결정
                action, ratio, exploration = \
                    self.agent.decide_action(
                        pred_value, pred_policy, epsilon)

                # 결정한 행동을 수행하고 즉시 보상과 지연 보상 획득
                immediate_reward, delayed_reward = \
                    self.agent.act(ratio)

                # 행동 및 행동에 대한 결과를 기억
                self.memory_sample.append(list(q_sample))
                self.memory_action.append(action)
                self.memory_reward.append(immediate_reward)
                if self.value_network is not None:
                    self.memory_value.append(pred_value)
                if self.policy_network is not None:
                    self.memory_policy.append(pred_policy)
                self.memory_pv.append(self.agent.portfolio_value)
                self.memory_num_stocks.append(self.agent.num_stocks)

                # 반복에 대한 정보 갱신
                self.batch_size += 1
                self.itr_cnt += 1

                # 지연 보상 발생된 경우 미니 배치 학습
                if learning and (delayed_reward != 0):
                    self.fit(delayed_reward, discount_factor)

            # 에포크 종료 후 학습
            if learning:
                self.fit(
                    self.agent.profitloss, discount_factor, full=True)

            # 에포크 관련 정보 로그 기록
            num_epoches_digit = len(str(num_epoches))
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')
            time_end_epoch = time.time()
            elapsed_time_epoch = time_end_epoch - time_start_epoch
            if self.learning_cnt > 0:
                self.loss /= self.learning_cnt
            logging.info("[Epoch {}/{}] Epsilon:{:.4f} "
                "PV:{:,.0f} Win:{}/{} LC:{} Loss:{:.6f} ET:{:.4f}".format(
                    epoch_str, num_epoches, epsilon,
                    self.agent.portfolio_value, self.agent.win_cnt, self.total_len,
                    self.learning_cnt, self.loss, elapsed_time_epoch))

            # 학습 관련 정보 갱신
            max_portfolio_value = max(
                max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1

        # 종료 시간
        time_end = time.time()
        elapsed_time = time_end - time_start

        # 학습 관련 정보 로그 기록
        with self.lock:
            logging.info("Elapsed Time:{elapsed_time:.4f} "
                "Max PV:{max_pv:,.0f} #Win:{cnt_win}".format(
                elapsed_time=elapsed_time,
                max_pv=max_portfolio_value, cnt_win=epoch_win_cnt))

    def save_models(self):
        if self.value_network is not None and \
                self.value_network_path is not None:
            self.value_network.save_model(self.value_network_path)
        if self.policy_network is not None and \
                self.policy_network_path is not None:
            self.policy_network.save_model(self.policy_network_path)

class ActorCriticLearner(ReinforcementLearner):
    def __init__(self, *args, value_network_path=None, policy_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_network_path = value_network_path
        self.policy_network_path = policy_network_path
        if self.value_network is None:
            self.init_value_network()
        if self.policy_network is None:
            self.init_policy_network()

    def get_batch(self, batch_size, delayed_reward, discount_factor):
        memory = zip(
            reversed(self.memory_sample[-batch_size:]),
            reversed(self.memory_action[-batch_size:]),
            reversed(self.memory_value[-batch_size:]),
            reversed(self.memory_policy[-batch_size:]),
            reversed(self.memory_reward[-batch_size:]),
        )
        x = np.zeros((batch_size, self.num_steps, self.num_features))
        y_value = np.zeros((batch_size, self.agent.NUM_ACTIONS))
        y_policy = np.full((batch_size, self.agent.NUM_ACTIONS), .5)
        value_max_next = 0
        reward_next = self.memory_reward[-1]
        for i, (sample, action, value, policy, reward) \
            in enumerate(memory):
            x[i] = sample
            y_value[i] = value
            y_policy[i] = policy
            r = (delayed_reward + reward_next - reward * 2) * 100
            y_value[i, action] = r + discount_factor * value_max_next
            y_policy[i, action] = sigmoid(value[action])
            value_max_next = value.max()
            reward_next = reward
        return x, y_value, y_policy


class A2CLearner(ActorCriticLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def get_batch(self, batch_size, delayed_reward, discount_factor):
        memory = zip(
            reversed(self.memory_sample[-batch_size:]),
            reversed(self.memory_action[-batch_size:]),
            reversed(self.memory_value[-batch_size:]),
            reversed(self.memory_policy[-batch_size:]),
            reversed(self.memory_reward[-batch_size:]),
        )
        x = np.zeros((batch_size, self.num_steps, self.num_features))
        y_value = np.zeros((batch_size, self.agent.NUM_ACTIONS))
        y_policy = np.full((batch_size, self.agent.NUM_ACTIONS), .5)
        value_max_next = 0
        reward_next = self.memory_reward[-1]
        print(batch_size)
        for i, (sample, action, value, policy, reward) \
            in enumerate(memory):
            print(sample, action, value, policy, reward)
            x[i] = sample
            r = (delayed_reward + reward_next - reward * 2) * 100
            y_value[i, action] = r + discount_factor * value_max_next
            advantage = value[action] - value.mean()
            y_policy[i, action] = sigmoid(advantage)
            value_max_next = value.max()
            reward_next = reward
        return x, y_value, y_policy