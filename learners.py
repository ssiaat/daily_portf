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

    def __init__(self, trainable=True, net='dnn', test=False, hold_criter=0., num_ticker=100, num_features=7, num_index=5,
                price_data=None, cap_data=None, index_data=None, index_ppc=None, training_data=None, num_steps=5, lr=0.001,
                value_network1_path=None, value_network2_path=None, target_value_network1_path=None, clip=True,
                target_value_network2_path=None, policy_network_path=None, output_path='', reuse_models=True):
        # 인자 확인
        assert lr > 0

        # 학습여부 설정
        self.trainable = True
        self.test = test

        # 환경 설정
        self.environment = Environment(price_data, cap_data, index_data, index_ppc, training_data,
                                       num_ticker, num_steps, num_features)
        self.net = net

        # 추가 데이터 설정
        self.price_data = price_data
        self.ks_ret = ((index_data.ks200.iloc[-1] - index_data.ks200.iloc[0]) / index_data.ks200.iloc[0])

        # 에이전트 설정
        self.agent = Agent(self.environment, num_ticker=num_ticker, hold_criter=hold_criter)

        # 학습 데이터
        self.total_len = len(price_data)  # 전체 학습 기간의 날짜 수
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
        self.output_dim = self.num_ticker

        # 메모리
        self.memory_sample_idx = deque(maxlen=1000)
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_copy = []
        self.memory_pv = []
        self.memory_pr = pd.DataFrame(index=price_data.index, columns=price_data.columns)
        self.memory_ksret = []
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

        self.diff_stocks_idx = None

        # hyperparameters
        self.alpha = 0.2  # entropy 반영 비율
        self.discount_factor = 0.9  # 할인율
        # self.deterministic = True if self.test else False
        self.deterministic = True if test else False
        self.polyak = 0.98
        self.batch_size = 16
        self.max_sample_len = 200
        self.clip = clip

    def init_value_network(self, activation='linear'):
        self.value_network = q_network(net=self.net, lr=self.lr, input_dim=self.num_features, output_dim=self.output_dim,
                                        num_ticker=self.num_ticker, num_index=self.num_index, num_steps=self.num_steps,
                                        batch_size=self.batch_size, trainable=self.trainable, activation=activation, value_flag=True)
        if self.reuse_models and os.path.exists(self.value_network1_path):
            print('reuse')
            self.value_network.load_model(model_path=[self.value_network1_path, self.value_network2_path])

    def init_target_value_network(self, activation='linear'):
        self.target_value_network = q_network(net=self.net, lr=self.lr, input_dim=self.num_features, output_dim=self.output_dim,
                                            num_ticker=self.num_ticker, num_index=self.num_index, num_steps=self.num_steps,
                                            batch_size=self.batch_size, trainable=self.trainable, activation=activation, value_flag=True)
        self.target_value_network.network1.model.set_weights(self.value_network.network1.model.get_weights())
        self.target_value_network.network2.model.set_weights(self.value_network.network2.model.get_weights())
        if self.reuse_models and os.path.exists(self.value_network1_path):
            self.target_value_network.load_model(model_path=[self.target_value_network1_path, self.target_value_network2_path])

    def init_policy_network(self, activation='tanh'):
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
        _ = self.environment.observe()
        ratio = tf.cast(self.environment.get_cap(), tf.float32)
        self.agent.act(ratio)
        self.agent.balance += self.agent.initial_balance - self.agent.portfolio_value
        self.environment.reset()

        # 메모리 초기화
        self.memory_sample_idx = deque(maxlen=self.max_sample_len)
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_pv = []
        self.memory_pr = pd.DataFrame(index=self.price_data.index, columns=self.price_data.columns)
        self.memory_copy = []
        self.memory_ksret = []
        self.memory_num_stocks = []

        # 에포크 관련 정보 초기화
        self.value_loss = 0.
        self.policy_loss = 0.
        self.itr_cnt = 0
        self.learning_cnt = 0

    def build_sample(self):
        self.diff_stocks_idx, idx = self.environment.observe()
        sample = None
        if idx is not None:
            sample = self.environment.get_training_data(idx)
        return sample, idx

    @abc.abstractmethod
    def get_batch(self, finished):
        pass

    @abc.abstractmethod
    def calculate_yvalue(self, r, next_s, d):
        pass

    def update_networks(self, finished=False):
        s, a, r, next_s, d = self.get_batch(finished)
        # q loss
        backup = tf.stop_gradient(self.calculate_yvalue(r, next_s, d))
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
        # 배치 학습 데이터 생성 및 신경망 갱신
        value_loss, policy_loss = self.update_networks(finished)
        # print('{:.4f} {:.4f}' .format(policy_loss, value_loss))
        self.value_loss += value_loss
        self.policy_loss += policy_loss
        self.learning_cnt += 1

        # target 신경망 갱신
        self.update_target_networks()


    def run(self, num_epoches=100, balance=10000000):
        info = "RL:sac LR:{lr}".format(lr=self.lr)
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
            # 평균 복제율
            mean_copy = 0.

            # 환경, 에이전트, 신경망, 가시화, 메모리 초기화
            self.reset()
            while True:
                # 샘플 생성, sample = [sample, sample of index]
                sample, idx = self.build_sample()

                if sample is None:
                    break
                if self.diff_stocks_idx:
                    print(f'change universe  {len(self.diff_stocks_idx)}  {self.price_data.index[self.environment.idx + self.num_steps - 1]}')
                next_sample = self.environment.transform_sample(sample[0])
                next_sample.append(np.array([sample[1]]))
                next_sample.append(np.array(sample[2]))
                # next_sample.append(np.array([self.agent.portfolio_ratio]))

                # 시총 가중으로 오늘 투자할 포트폴리오 비중 결정
                pi, logp_pi = self.policy_network.predict(next_sample, self.deterministic, learn=False)

                # 포트폴리오 가치를 오늘 가격 반영해서 갱신
                self.agent.renewal_portfolio_ratio(transaction=False, diff_stock_idx=self.diff_stocks_idx)

                # 오늘 가격으로 변경된 portf_value로 어제 투자에 대한 보상 계산
                immediate_reward = self.agent.get_reward(self.diff_stocks_idx)
                ratio = self.agent.similar_with_cap(pi)

                # else:
                #     ratio, penalty = self.agent.penalty_diff_bm(pi)
                #     immediate_reward += penalty

                # action, ratio = self.agent.decide_action(pi, self.clip)

                # 종목 변화가 있다면 해당 종목의 idx 저장, agent.act에서 반영
                self.agent.act(ratio, self.diff_stocks_idx)

                self.diff_stocks_idx = None

                curr_cap = self.environment.get_cap()
                bm_copy = tf.reduce_sum(tf.math.abs(curr_cap - self.agent.portfolio_ratio)) / 2.0 * 100.0
                mean_copy += bm_copy

                # 행동 및 행동에 대한 결과를 기억
                self.memory_sample_idx.append(idx)
                self.memory_action.append(pi)
                self.memory_reward.append(immediate_reward)
                self.memory_pv.append(self.agent.last_portfolio_value.numpy())  # last는 어제 투자한 portf를 오늘 종가로 평가한 것
                self.memory_pr.iloc[self.environment.idx][self.environment.universe] = self.agent.last_portfolio_ratio
                self.memory_copy.append(bm_copy.numpy())
                self.memory_ksret.append((self.environment.get_ks() - self.environment.ks_data.iloc[0]) / self.environment.ks_data.iloc[0])

                # 반복에 대한 정보 갱신
                self.itr_cnt += 1
                if self.itr_cnt % 20 == 0:
                    if self.itr_cnt == 20:
                        _ = self.memory_sample_idx.popleft()
                    fit_iter = len(self.memory_sample_idx) // 50 + 1
                    for _ in range(fit_iter):
                        self.fit()
                    print('{:,} {:.4f} {:.4f} {:.4f}'.format(self.agent.portfolio_value, mean_copy / 20.0, (self.agent.portfolio_value - self.agent.initial_balance) / self.agent.initial_balance,
                                                      (self.environment.get_ks() - self.environment.ks_data.iloc[0]) / self.environment.ks_data.iloc[0]))
                    mean_copy = 0.

                if tf.math.is_nan(self.value_loss) or tf.math.is_nan(self.policy_loss):
                    print('loss is nan!')
                    return

            # 에포크 종료 후 학습
            for i in range(50):
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
            logging.info("[Epoch {}/{}] PV:{:,.0f} PVReturn: {:.4f}/{:.4f}"
                " Win:{}/{} LC:{} ValueLoss:{:.6f} PolicyLoss:{:.6f} ET:{:.4f}".format(
                    epoch_str, num_epoches, self.agent.portfolio_value,
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
            target_value_output1_path = self.target_value_network1_path[:-3] + '_output1' + self.target_value_network1_path[-3:]
            target_value_output2_path = self.target_value_network2_path[:-3] + '_output2' + self.target_value_network2_path[-3:]
            policy_network_path = self.policy_network_path[:-3] + '_output1' + self.policy_network_path[-3:]

            pd.DataFrame(np.array([self.memory_pv, self.memory_copy, self.memory_ksret]).T, index=self.price_data.index[self.num_steps-1:], columns=['pv', 'copy', 'ks200']).to_csv('models/test_result.csv')
            self.memory_pr.to_csv('models/portf_ratio.csv')
        else:
            value_output1_path = self.value_network1_path
            value_output2_path = self.value_network2_path
            target_value_output1_path = self.target_value_network1_path
            target_value_output2_path = self.target_value_network2_path
            policy_network_path = self.policy_network_path
        if self.value_network is not None and self.value_network1_path is not None:
            self.value_network.save_model([value_output1_path, value_output2_path])
            self.target_value_network.save_model([target_value_output1_path, target_value_output2_path])
        if self.policy_network is not None and self.policy_network_path is not None:
            self.policy_network.save_model(policy_network_path)


class SACLearner(ReinforcementLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_value_network()
        self.init_target_value_network()
        self.init_policy_network()

    def get_batch(self, finished=False):
        idx_batch = random.sample(list(itertools.islice(self.memory_sample_idx, 0, len(self.memory_sample_idx)-1)), self.batch_size)

        # 행동에 대한 보상은 다음날 알 수 있음
        s = np.zeros((self.batch_size, self.num_ticker, 1, self.num_steps, self.num_features))
        s_index = np.zeros((self.batch_size, 1, self.num_steps, self.num_index))
        ks_portf_state = np.zeros((self.batch_size, self.num_ticker))
        portf_state = np.zeros((self.batch_size, self.num_ticker))

        next_s = np.zeros((self.batch_size, self.num_ticker, 1, self.num_steps, self.num_features))
        next_s_index = np.zeros((self.batch_size,  1, self.num_steps, self.num_index))
        ks_portf_next_state = np.zeros((self.batch_size, self.num_ticker))
        portf_next_state = np.zeros((self.batch_size, self.num_ticker))

        action = np.zeros((self.batch_size, self.num_ticker))
        reward = np.zeros((self.batch_size, self.num_ticker))
        d = np.zeros((self.batch_size, 1))

        for i, idx in enumerate(idx_batch):
            sample = self.environment.get_training_data(idx)
            s[i] = self.environment.transform_sample(sample[0])
            s_index[i] = np.array([sample[1]])
            ks_portf_state[i] = np.array(sample[2])
            # portf_state[i] = np.array(self.memory_pr.iloc[idx].dropna().values)

            sample = self.environment.get_training_data(idx + 1)
            next_s[i] = self.environment.transform_sample(sample[0])
            next_s_index[i] = np.array([sample[1]])
            ks_portf_next_state[i] = np.array(sample[2])
            # portf_next_state[i] = np.array(self.memory_pr.iloc[idx + 1].dropna().values)

            action[i] = np.array(self.memory_action[idx])
            reward[i] = self.memory_reward[idx+1]
            if idx == len(self.price_data) - 1:
                d[i] = 1

        # input 형태로 변경
        s = list(np.squeeze(s.swapaxes(0,1), axis=2))
        s.append(np.squeeze(s_index, axis=1))
        s.append(ks_portf_state)
        # s.append(portf_state)

        next_s = list(np.squeeze(next_s.swapaxes(0, 1), axis=2))
        next_s.append(np.squeeze(next_s_index, axis=1))
        next_s.append(ks_portf_next_state)
        # next_s.append(portf_next_state)

        return s, action, reward, next_s, d

    def calculate_yvalue(self, r, next_s, d):
        a2, logp_a2 = self.policy_network.predict(next_s)
        q1_pi_targ, q2_pi_targ = self.target_value_network.predict(next_s.copy(), a2)
        q_pi_targ = tf.math.minimum(q1_pi_targ, q2_pi_targ)
        backup = r + self.discount_factor * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        return backup