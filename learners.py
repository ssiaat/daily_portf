import os
import logging
import abc

import threading
import time
import numpy as np
from utils import sigmoid
from environment import Environment
from agent import Agent
from networks import DNN

import tensorflow as tf

class ReinforcementLearner:
    __metaclass__ = abc.ABCMeta
    lock = threading.Lock()

    def __init__(self, trainable=False,
                price_data=None, cap_data=None, ks_data=None, training_data=None,
                hold_criter=0., delayed_reward_threshold=.05,
                num_ticker=100, num_features=7, lr=0.001,
                value_network=None, policy_network=None,
                value_network_path=None, policy_network_path=None,
                output_path='', reuse_models=True):
        # 인자 확인
        assert lr > 0
        # 강화학습 기법 설정
        # 학습여부 설정
        self.trainable = trainable
        # 환경 설정
        self.price_data = price_data
        self.environment = Environment(price_data, cap_data, ks_data)
        self.ks_ret = ((ks_data.iloc[-1] - ks_data.iloc[0]) / ks_data.iloc[0]).values[0]

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
        self.value_loss = 0.
        self.policy_loss = 0.
        self.itr_cnt = 0
        self.batch_size = 0
        self.learning_cnt = 0

        # 로그 등 출력 경로
        self.value_network_path = value_network_path
        self.policy_network_path = policy_network_path
        self.output_path = output_path

        # action 조정 ([0,1,0,0,1] => [0,3,4,6,9]
        self.modify_action_idx = np.array([i*2 for i in range(self.num_ticker)])

    def init_value_network(self, activation='sigmoid'):
        self.value_network = DNN(
            input_dim=self.num_features,
            output_dim=self.agent.NUM_ACTIONS * self.num_ticker,
            num_ticker=self.num_ticker,
            trainable=self.trainable,
            lr=self.lr, activation=activation)
        if self.reuse_models and \
            os.path.exists(self.value_network_path):
                self.value_network.load_model(model_path=self.value_network_path)

    def init_policy_network(self, activation='sigmoid'):
        self.policy_network = DNN(
            input_dim=self.num_features,
            output_dim=self.num_ticker,
            num_ticker=self.num_ticker,
            trainable=self.trainable,
            lr=self.lr, activation=activation)
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
        self.value_loss = 0.
        self.policy_loss = 0.
        self.itr_cnt = 0
        self.batch_size = 0
        self.learning_cnt = 0

    def build_sample(self):
        self.environment.observe()
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            self.sample = self.training_data.iloc[self.training_data_idx].values.reshape((-1,7))
            return self.sample
        return None

    @abc.abstractmethod
    def get_batch(self, batch_size, delayed_reward, discount_factor):
        pass

    def update_networks(self, batch_size, delayed_reward, discount_factor):
        # 배치 학습 데이터 생성
        x, y_value, y_policy = self.get_batch(batch_size, delayed_reward, discount_factor)
        if len(x) > 0:
            value_loss = self.value_network.learn(x, y_value)
            policy_loss = self.policy_network.learn(x, y_policy)
            # if value_loss == 0.:
            #     print(y_value)
            #     print(value_loss)
            #     print()
            return value_loss, policy_loss
        return None

    def fit(self, delayed_reward, discount_factor, full=False):
        batch_size = len(self.memory_reward) if full else self.batch_size
        # 배치 학습 데이터 생성 및 신경망 갱신
        if batch_size > 0:
            value_loss, policy_loss = self.update_networks(batch_size, delayed_reward, discount_factor)
            self.value_loss += value_loss
            self.policy_loss += policy_loss
            self.learning_cnt += 1
            self.memory_learning_idx.append(self.training_data_idx)
            self.batch_size = 0

    def run(self, num_epoches=100, balance=10000000, discount_factor=0.9, start_epsilon=0.5, learning=True):
        info = "RL:a2c LR:{lr} " \
            "DF:{discount_factor} DRT:{delayed_reward_threshold}".format(
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
        print('Start Learning')
        for epoch in range(num_epoches):
            time_start_epoch = time.time()

            # 환경, 에이전트, 신경망, 가시화, 메모리 초기화
            self.reset()

            # 학습을 진행할 수록 탐험 비율 감소
            if learning:
                epsilon = start_epsilon  * (1. - float(epoch) / (num_epoches - 1))
            else:
                epsilon = start_epsilon

            while True:
                # 샘플 생성
                next_sample = self.build_sample()
                if next_sample is None:
                    break
                next_sample = np.split(next_sample, self.num_ticker, axis=0)

                # 가치, 정책 신경망 예측
                pred_value = None
                pred_policy = None
                curr_cap = self.environment.get_cap()
                curr_cap_value = np.tile(curr_cap.reshape(-1, 1), 2).reshape(-1,)

                if self.value_network is not None:
                    pred_value = self.value_network.predict(next_sample) * curr_cap_value
                if self.policy_network is not None:
                    pred_policy = self.policy_network.predict(next_sample) * curr_cap

                # 포트폴리오 가치를 오늘 가격 반영해서 갱신
                self.agent.renewal_portfolio_ratio(transaction=False)

                # 신경망 또는 탐험에 의한 행동 결정
                action, ratio, exploration = self.agent.decide_action(pred_policy, epsilon)

                # 결정한 행동을 수행하고 즉시 보상과 지연 보상 획득
                immediate_reward, delayed_reward = self.agent.act(ratio)

                # 행동 및 행동에 대한 결과를 기억
                self.memory_sample.append(next_sample)
                self.memory_action.append(action + self.modify_action_idx)
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
                if learning and (tf.reduce_sum(delayed_reward) != 0):
                    self.fit(delayed_reward, discount_factor)

            # 에포크 종료 후 학습
            if learning:
                self.fit(self.agent.profitloss * self.agent.portfolio_ratio, discount_factor, full=True)

            # 에포크 관련 정보 로그 기록
            num_epoches_digit = len(str(num_epoches))
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')
            time_end_epoch = time.time()
            elapsed_time_epoch = time_end_epoch - time_start_epoch
            if self.learning_cnt > 0:
                self.value_loss /= self.learning_cnt
                self.policy_loss /= self.learning_cnt
            pv_ret = (self.agent.portfolio_value - self.agent.initial_balance) / self.agent.initial_balance
            logging.info("[Epoch {}/{}] Epsilon:{:.4f} PV:{:,.0f} PVReturn: {:.4f}/{:.4f}"
                " Win:{}/{} LC:{} ValueLoss:{:.6f} PolicyLoss:{:.6f} ET:{:.4f}".format(
                    epoch_str, num_epoches, epsilon, self.agent.portfolio_value,
                    pv_ret, self.ks_ret, self.agent.win_cnt, self.total_len,
                    self.learning_cnt, self.value_loss, self.policy_loss, elapsed_time_epoch))

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
        if self.value_network is not None and self.value_network_path is not None:
            self.value_network.save_model(self.value_network_path)
        if self.policy_network is not None and self.policy_network_path is not None:
            self.policy_network.save_model(self.policy_network_path)

class A2CLearner(ReinforcementLearner):
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
        x = np.zeros((batch_size, self.num_ticker, 1, self.num_features))
        y_value = np.zeros((batch_size, self.num_ticker * self.agent.NUM_ACTIONS))
        y_policy = np.zeros((batch_size, self.num_ticker))
        value_max_next = np.zeros((self.num_ticker,))
        reward_next = self.memory_reward[-1]
        for i, (sample, action, value, policy, reward) in enumerate(memory):
            x[i] = np.array(sample)
            r = (delayed_reward + reward_next - reward * 2) * 100
            y_value[i, action] = r + discount_factor * value_max_next
            advantage = tf.gather(value, action) - tf.reduce_mean(tf.reshape(value, (-1, 2)), axis=1)
            y_policy[i] = tf.nn.softmax(advantage)
            value_max_next = tf.reduce_max(tf.reshape(value, (-1, 2)), axis=1)
            reward_next = reward

        # input 형태로 변경
        return list(np.squeeze(x.swapaxes(0,1), axis=2)), y_value, y_policy
