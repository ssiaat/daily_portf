import os
import logging
import abc
import random
from collections import deque
import itertools
import threading
import time
import numpy as np
import pandas as pd

from environment import Environment
from agent import Agent
from networks import pi_network, q_network

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
class ReinforcementLearner:
    __metaclass__ = abc.ABCMeta
    lock = threading.Lock()

    def __init__(self, stock_codes_yearly=None, stock_codes=None, trainable=True, net='dnn', test=False,
                price_data=None, cap_data=None, index_data=None, index_ppc=None, training_data=None, num_steps=5,
                hold_criter=0., delayed_reward_threshold=.05, num_ticker=100, num_features=7, num_index=5, lr=0.001,
                value_network1_path=None, value_network2_path=None, target_value_network1_path=None, target_value_network2_path=None,
                policy_network_path=None, output_path='', reuse_models=True):
        # 인자 확인
        assert lr > 0

        # 학습여부 설정
        self.trainable = trainable
        self.test = test

        # 환경 설정
        self.environment = Environment(price_data, cap_data, index_data, index_ppc, training_data,
                                       stock_codes_yearly, num_ticker, num_steps, num_features)
        self.net = net

        # 추가 데이터 설정
        self.price_data = price_data
        self.ks_ret = ((index_data.ks200.iloc[-1] - index_data.ks200.iloc[0]) / index_data.ks200.iloc[0])

        # 에이전트 설정
        self.agent = Agent(self.environment, num_ticker=num_ticker, hold_criter=hold_criter, delayed_reward_threshold=delayed_reward_threshold)

        # 학습 데이터
        self.total_len = len(training_data.index) / len(stock_codes)  # 전체 학습 기간의 날짜 수
        self.num_ticker = num_ticker
        self.num_features = num_features
        self.num_index = num_index
        self.num_steps = num_steps
        
        # 신경망 설정
        self.lr = lr
        self.value_network = None
        self.target_value_network = None
        self.policy_network = None
        self.reuse_models = reuse_models
        self.output_dim = 1 if self.net == 'lstm' else self.num_ticker

        # 메모리
        self.memory_sample_idx = deque(maxlen=200)
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_pv = []
        self.memory_pr = []
        self.memory_num_stocks = []

        # 에포크 관련 정보
        self.value_loss = 0.
        self.policy_loss = 0.
        self.itr_cnt = 0
        self.learning_cnt = 0

        # 로그 등 출력 경로
        self.value_network1_path = value_network1_path
        self.value_network2_path = value_network2_path
        self.target_value_network1_path = target_value_network1_path
        self.target_value_network2_path = target_value_network2_path
        self.policy_network_path = policy_network_path
        self.output_path = output_path

        # action 조정 ([0,1,0,0,1] => [0,3,4,6,9]
        self.modify_action_idx = np.array([i*2 for i in range(self.num_ticker)])

        # test시 연간 변화 종목
        self.diff_stocks_idx = None

        # hyperparameters
        self.alpha = 0.2  # entropy 반영 비율
        self.discount_factor = 0.9  # 할인율
        self.deterministic = True if self.test else False
        self.polyak = 0.95

    def init_value_network(self, activation='linear'):
        self.value_network = q_network(net=self.net, lr=self.lr, input_dim=self.num_features, output_dim=self.output_dim,
                                        num_ticker=self.num_ticker, num_index=self.num_index, num_steps=self.num_steps,
                                        trainable=self.trainable, activation=activation, value_flag=True)
        if self.reuse_models and os.path.exists(self.value_network1_path):
            print('reuse')
            self.value_network.load_model(model_path=[self.value_network1_path, self.value_network2_path])


    def init_target_value_network(self, activation='linear'):
        self.target_value_network = q_network(net=self.net, lr=self.lr, input_dim=self.num_features, output_dim=self.output_dim,
                                            num_ticker=self.num_ticker, num_index=self.num_index, num_steps=self.num_steps,
                                            trainable=self.trainable, activation=activation, value_flag=True)
        self.target_value_network.network1.model.set_weights(self.value_network.network1.model.get_weights())
        self.target_value_network.network2.model.set_weights(self.value_network.network2.model.get_weights())
        if self.reuse_models and os.path.exists(self.value_network1_path):
            self.target_value_network.load_model(model_path=[self.target_value_network1_path, self.target_value_network2_path])


    def init_policy_network(self, activation='linear'):
        output_dim = 1 if self.net == 'lstm' else self.num_ticker
        self.policy_network = pi_network(net=self.net, lr=self.lr, input_dim=self.num_features, output_dim=self.output_dim,
                                         num_ticker=self.num_ticker, num_steps=self.num_steps, num_index=self.num_index,
                                         trainable=self.trainable, activation=activation, value_flag=False, alpha=self.alpha)
        if self.reuse_models and os.path.exists(self.policy_network_path):
            self.policy_network.load_model(model_path=self.policy_network_path)

    def reset(self):
        # 환경 초기화
        self.environment.reset()

        # 에이전트 초기화
        self.agent.reset()

        # 메모리 초기화
        self.memory_sample_idx = deque(maxlen=200)
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_pv = []
        self.memory_pr = []
        self.memory_num_stocks = []

        # 에포크 관련 정보 초기화
        self.value_loss = 0.
        self.policy_loss = 0.
        self.itr_cnt = 0
        self.learning_cnt = 0

    def build_sample(self):
        idx = self.environment.observe()
        sample = None
        if idx is not None:
            sample = self.environment.get_training_data(idx)
        return sample, idx

    @abc.abstractmethod
    def get_batch(self, batch_size, finished):
        pass

    @abc.abstractmethod
    def calculate_yvalue(self, r, next_s, d):
        pass

    def update_networks(self, batch_size, finished=False):
        s, a, r, next_s, d = self.get_batch(batch_size, finished)

        # q loss
        backup = self.calculate_yvalue(r, next_s, d)
        value_loss = self.value_network.learn(s.copy(), a, backup)
        policy_loss = self.policy_network.learn(s, self.value_network)

        return value_loss, policy_loss

    def update_target_networks(self):
        for i, (q, q_targ) in enumerate(zip(self.value_network.network1.model.trainable_variables, self.target_value_network.network1.model.trainable_variables)):
            q_targ = tf.math.multiply(q_targ, self.polyak)
            q_targ += (1 - self.polyak) * q
            self.target_value_network.network1.model.trainable_variables[i].assign(q_targ)
        for i, (q, q_targ) in enumerate(zip(self.value_network.network2.model.trainable_variables, self.target_value_network.network2.model.trainable_variables)):
            q_targ = tf.math.multiply(q_targ, self.polyak)
            q_targ += (1 - self.polyak) * q
            self.target_value_network.network2.model.trainable_variables[i].assign(q_targ)

    def fit(self,finished=False):
        batch_size = 5

        # 배치 학습 데이터 생성 및 신경망 갱신
        if batch_size > 0:
            value_loss, policy_loss = self.update_networks(batch_size, finished)
            self.value_loss += value_loss
            self.policy_loss += policy_loss
            self.learning_cnt += 1

        # target 신경망 갱신
        self.update_target_networks()

    def run(self, num_epoches=100, balance=10000000, start_epsilon=0.5):
        info = "RL:a2c LR:{lr}".format(lr=self.lr)
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
            if not self.test:
                epsilon = start_epsilon  * (1. - float(epoch) / (num_epoches - 1))
            else:
                epsilon = 0

            while True:

                # 샘플 생성, sample = [sample, sample of index]
                sample, idx = self.build_sample()
                if sample is None:
                    break
                next_sample = self.environment.transform_sample(sample[0])
                next_sample.append(np.array([sample[1]]))

                curr_cap = self.environment.get_cap()
                # 시총 가중으로 오늘 투자할 포트폴리오 비중 결정
                pi, logp_pi = self.policy_network.predict(next_sample, self.deterministic, learn=False)
                pi = self.agent.similar_with_cap(pi)

                # 포트폴리오 가치를 오늘 가격 반영해서 갱신
                self.agent.renewal_portfolio_ratio(transaction=False, diff_stock_idx=self.diff_stocks_idx)

                # 오늘 가격으로 변경된 portf_value로 어제 투자에 대한 보상 계산
                immediate_reward = self.agent.get_reward()
                
                # 신경망 또는 탐험에 의한 행동 결정
                ratio, exploration = self.agent.decide_action(pi, epsilon)
                action = ratio - self.agent.portfolio_ratio

                # 종목 변화가 있다면 해당 종목의 idx 저장, agent.act에서 반영
                self.agent.act(ratio, self.diff_stocks_idx)
                self.diff_stocks_idx = None

                # 행동 및 행동에 대한 결과를 기억
                self.memory_sample_idx.append(idx)
                self.memory_action.append(action)
                self.memory_reward.append(immediate_reward)
                self.memory_pv.append(self.agent.last_portfolio_value)  # last는 어제 투자한 portf를 오늘 종가로 평가한 것

                # 반복에 대한 정보 갱신
                self.itr_cnt += 1

                if self.itr_cnt % 10 == 0:
                    if not self.test and self.itr_cnt == 10:
                        _ = self.memory_sample_idx.popleft()
                    self.fit()
                    print('{:,} {:.4f}'.format(self.agent.portfolio_value, (self.environment.get_ks() - self.environment.ks_data.iloc[0]) / self.environment.ks_data.iloc[0]))

                # test는 연도별로 종목 갱신, 하루 끝나고 매일 체크
                if self.test:
                    self.diff_stocks_idx = self.environment.update_stock_codes()
                    if self.diff_stocks_idx:
                        print(f'change universe  {len(self.diff_stocks_idx)}')

            # 에포크 종료 후 학습
            for i in range(10):
                self.fit(finished=True)
            # print(f'differ between port and cap: {self.agent.portfolio_ratio - self.environment.get_cap()}')

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
            max_portfolio_value = max(max_portfolio_value, self.agent.portfolio_value)
            if (self.agent.portfolio_value - self.agent.initial_balance) / self.agent.initial_balance > (self.environment.get_ks() - self.agent.base_ks) / self.agent.base_ks:
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
        if self.test:
            value_output1_path = self.value_network1_path[:-3] + '_output1' + self.value_network1_path[-3:]
            value_output2_path = self.value_network2_path[:-3] + '_output2' + self.value_network2_path[-3:]
            policy_network_path = self.policy_network_path[:-3] + '_output1' + self.policy_network_path[-3:]
            pd.DataFrame(self.memory_pv, index=self.price_data.index, columns=['pv']).to_csv('models/test_result.csv')
        else:
            value_output1_path = self.value_network1_path
            value_output2_path = self.value_network2_path
            policy_network_path = self.policy_network_path
        if self.value_network is not None and self.value_network1_path is not None:
            self.value_network.save_model([value_output1_path, value_output2_path])
        if self.policy_network is not None and self.policy_network_path is not None:
            self.policy_network.save_model(policy_network_path)


class A2CLearner(ReinforcementLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_value_network()
        self.init_target_value_network()
        self.init_policy_network()

    def get_batch(self, batch_size, finished=False):
        idx_batch = random.sample(list(itertools.islice(self.memory_sample_idx, 0, len(self.memory_sample_idx)-1)), batch_size)
        # 행동에 대한 보상은 다음날 알 수 있음
        s = np.zeros((batch_size, self.num_ticker, 1, self.num_steps, self.num_features))
        s_index = np.zeros((batch_size, 1, 1, self.num_steps, self.num_index))
        next_s = np.zeros((batch_size, self.num_ticker, 1, self.num_steps, self.num_features))
        next_s_index = np.zeros((batch_size, 1, 1, self.num_steps, self.num_index))
        action = np.zeros((batch_size, self.num_steps, self.num_ticker))
        reward = np.zeros((batch_size, self.num_ticker))
        d = np.zeros((batch_size, 1))
        for i, idx in enumerate(idx_batch):
            sample = self.environment.get_training_data(idx)
            s[i] = self.environment.transform_sample(sample[0])
            s_index[i] = np.array([sample[1]])
            action[i] = np.array([self.memory_action[idx]])
            reward[i] = self.memory_reward[idx+1] * 100
            if idx == len(self.price_data) - 1:
                d[i] = 1

        # input 형태로 변경
        s = list(np.squeeze(s.swapaxes(0,1), axis=2))
        s.append(*np.squeeze(s_index.swapaxes(0,1), axis=2))
        next_s = list(np.squeeze(next_s.swapaxes(0, 1), axis=2))
        next_s.append(*np.squeeze(next_s_index.swapaxes(0, 1), axis=2))

        return s, action, reward, next_s, d

    def calculate_yvalue(self, r, next_s, d):
        a2, logp_a2 = self.policy_network.predict(next_s)
        q1_pi_targ, q2_pi_targ = self.value_network.predict(next_s, a2)
        q_pi_targ = tf.math.minimum(q1_pi_targ, q2_pi_targ)
        backup = r + self.discount_factor * (1 - d) * (q_pi_targ - self.alpha * logp_a2)
        return backup