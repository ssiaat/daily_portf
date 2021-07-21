import os
import sys
import logging
import argparse
import json

import numpy as np
import pandas as pd

import settings
import utils
import data_manager
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_name = 'dnn'
value_name = None
policy_name = None
start_date = '20000201'
end_date = '20001229'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ver', choices=['v1', 'v2', 'v3', 'v4'], default='v3')
    parser.add_argument('--rl_method',
                        choices=['ac', 'a2c'], default='a2c')
    parser.add_argument('--num_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--discount_factor', type=float, default=0.9)
    parser.add_argument('--start_epsilon', type=float, default=0.3)
    parser.add_argument('--balance', type=int, default=10000000)
    parser.add_argument('--num_epoches', type=int, default=30)
    parser.add_argument('--hold_criter', type=float, default=0.)
    parser.add_argument('--delayed_reward_threshold',
                        type=float, default=0.05)
    parser.add_argument('--output_name', default=utils.get_time_str())
    parser.add_argument('--value_network_name', default=value_name)
    parser.add_argument('--policy_network_name', default=policy_name)
    parser.add_argument('--reuse_models', action='store_true')
    parser.add_argument('--learning', action='store_true')
    parser.add_argument('--start_date', default=start_date)
    parser.add_argument('--end_date', default=end_date)
    args = parser.parse_args()

    # Keras Backend 설정
    os.environ['KERAS_BACKEND'] = 'tensorflow'

    # 출력 경로 설정
    output_path = os.path.join(settings.BASE_DIR, 'output/{}_{}'.format(args.output_name, args.rl_method))
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # 파라미터 기록
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        f.write(json.dumps(vars(args)))

    # 로그 기록 설정
    file_handler = logging.FileHandler(filename=os.path.join(
        output_path, "{}.log".format(args.output_name)), encoding='utf-8')
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s", handlers=[file_handler, stream_handler], level=logging.DEBUG)

    if args.learning == True:
        print('-----------------this running is for training')
    else:
        print('-----------------this running is for testing')

    # 로그, Keras Backend 설정을 먼저하고 RLTrader 모듈들을 이후에 임포트해야 함
    from learners import ActorCriticLearner, A2CLearner

    # 모델 경로 준비
    value_network_path = ''
    policy_network_path = ''
    if args.value_network_name is not None:
        value_network_path = os.path.join(settings.BASE_DIR,
                                          'models/{}.h5'.format(args.value_network_name))
    else:
        value_network_path = os.path.join(
            output_path, '{}_{}_value.h5'.format(
                args.rl_method, args.output_name))
    if args.policy_network_name is not None:
        policy_network_path = os.path.join(settings.BASE_DIR,
                                           'models/{}.h5'.format(args.policy_network_name))
    else:
        policy_network_path = os.path.join(
            output_path, '{}_{}_policy.h5'.format(
                args.rl_method, args.output_name))

    #-----------------------
    # 학습은 삼성전자부터 마지막 ticker, 마지막 ticker부터 삼성전자 순서로 진행함
    #-----------------------
    stock_codes = np.array(pd.read_sql('show tables', data_manager.conn).values).reshape(-1,)[:2]
    price_data, vol_data, training_data = data_manager.make_data(stock_codes, args.start_date, args.end_date)

    # 공통 파라미터 설정
    common_params = {'rl_method': args.rl_method, 'trainable': args.learning, 'num_features': int(len(training_data.columns) / len(stock_codes)),
                     'delayed_reward_threshold': args.delayed_reward_threshold, 'num_ticker': len(stock_codes), 'hold_criter': args.hold_criter,
                     'num_steps': args.num_steps, 'lr': args.lr, 'output_path': output_path, 'reuse_models': args.reuse_models}

    # 강화학습 시작
    learner = None
    # Not defined Error 때문에

    common_params.update({'price_data': price_data, 'vol_data': vol_data,
                          'training_data': training_data})
    if args.rl_method == 'ac':
        learner = ActorCriticLearner(**{**common_params,
                                        'value_network_path': value_network_path,
                                        'policy_network_path': policy_network_path})
    elif args.rl_method == 'a2c':
        learner = A2CLearner(**{**common_params,
                                'value_network_path': value_network_path,
                                'policy_network_path': policy_network_path})
    if learner is not None:
        learner.run(balance=args.balance,
                    num_epoches=args.num_epoches,
                    discount_factor=args.discount_factor,
                    start_epsilon=args.start_epsilon,
                    learning=args.learning)
        learner.save_models()