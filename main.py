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

model_name = 'lstm'
value_name = None
policy_name = None
start_date = '20000201'
end_date = '20151230'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--stock_code', nargs='+')
    parser.add_argument('--ver', choices=['v1', 'v2', 'v3', 'v4'], default='v3')
    parser.add_argument('--rl_method',
                        choices=['dqn', 'pg', 'ac', 'a2c', 'a3c'], default='a2c')
    parser.add_argument('--net',
                        choices=['dnn', 'lstm', 'cnn'], default=model_name)
    parser.add_argument('--num_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--discount_factor', type=float, default=0.9)
    parser.add_argument('--start_epsilon', type=float, default=0.3)
    parser.add_argument('--balance', type=int, default=10000000)
    parser.add_argument('--num_epoches', type=int, default=100)
    parser.add_argument('--delayed_reward_threshold',
                        type=float, default=0.05)
    parser.add_argument('--backend',
                        choices=['tensorflow', 'plaidml'], default='tensorflow')
    parser.add_argument('--output_name', default=utils.get_time_str())
    # f'{model_name}_value_15'   f'{model_name}_policy_15'
    parser.add_argument('--value_network_name', default=value_name)
    parser.add_argument('--policy_network_name', default=policy_name)
    parser.add_argument('--reuse_models', action='store_true')
    parser.add_argument('--learning', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--start_date', default=start_date)
    parser.add_argument('--end_date', default=end_date)
    args = parser.parse_args()

    # Keras Backend 설정
    if args.backend == 'tensorflow':
        os.environ['KERAS_BACKEND'] = 'tensorflow'
    elif args.backend == 'plaidml':
        os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

    # 출력 경로 설정
    output_path = os.path.join(settings.BASE_DIR,
                               'output/{}_{}_{}'.format(args.output_name, args.rl_method, args.net))
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
    logging.basicConfig(format="%(message)s",
                        handlers=[file_handler, stream_handler], level=logging.DEBUG)

    if args.learning == True:
        print('-----------------this running is for training')
    else:
        print('-----------------this running is for testing')

    # 로그, Keras Backend 설정을 먼저하고 RLTrader 모듈들을 이후에 임포트해야 함
    from agent import Agent
    from learners import DQNLearner, PolicyGradientLearner, \
        ActorCriticLearner, A2CLearner, A3CLearner

    # 모델 경로 준비
    value_network_path = ''
    policy_network_path = ''
    if args.value_network_name is not None:
        value_network_path = os.path.join(settings.BASE_DIR,
                                          'models/{}.h5'.format(args.value_network_name))
    else:
        value_network_path = os.path.join(
            output_path, '{}_{}_value_{}.h5'.format(
                args.rl_method, args.net, args.output_name))
    if args.policy_network_name is not None:
        policy_network_path = os.path.join(settings.BASE_DIR,
                                           'models/{}.h5'.format(args.policy_network_name))
    else:
        policy_network_path = os.path.join(
            output_path, '{}_{}_policy_{}.h5'.format(
                args.rl_method, args.net, args.output_name))

    common_params = {}
    list_stock_code = []
    list_chart_data = []
    list_training_data = []
    list_min_trading_unit = []
    list_max_trading_unit = []

    #-----------------------
    # 학습은 삼성전자부터 마지막 ticker, 마지막 ticker부터 삼성전자 순서로 진행함
    #-----------------------
    stock_codes = np.array(pd.read_sql('show tables', data_manager.conn).values).reshape(-1,)
    args.stock_code = np.concatenate([stock_codes, np.flip(stock_codes)])

    for stock_code in args.stock_code:
        # 차트 데이터, 학습 데이터 준비
        # chart_data, training_data = data_manager.load_data(
        #     os.path.join(settings.BASE_DIR,
        #     'data/{}/{}.csv'.format(args.ver, stock_code)),
        #     args.start_date, args.end_date, ver=args.ver)
        chart_data, training_data = data_manager.load_data_sql(
            stock_code, args.start_date, args.end_date, ver=args.ver)  ## 인자 1 ## 인자 2, 3

        # 최소/최대 투자 단위 설정
        try:
            min_trading_unit = max(int(100000 / chart_data.iloc[-1]['price_mod']), 1)
            max_trading_unit = max(int(1000000 / chart_data.iloc[-1]['price_mod']), 1)
        except:
            None

        # 공통 파라미터 설정
        common_params = {'rl_method': args.rl_method, 'trainable': args.learning,
                         'delayed_reward_threshold': args.delayed_reward_threshold,
                         'net': args.net, 'num_steps': args.num_steps, 'lr': args.lr,
                         'output_path': output_path, 'reuse_models': args.reuse_models,
                         'visualize':args.visualize}

        # 강화학습 시작
        learner = None
        if args.rl_method != 'a3c':
            common_params.update({'stock_code': stock_code,
                                  'chart_data': chart_data,
                                  'training_data': training_data,
                                  'min_trading_unit': min_trading_unit,
                                  'max_trading_unit': max_trading_unit})
            if args.rl_method == 'dqn':
                learner = DQNLearner(**{**common_params,
                                        'value_network_path': value_network_path})
            elif args.rl_method == 'pg':
                learner = PolicyGradientLearner(**{**common_params,
                                                   'policy_network_path': policy_network_path})
            elif args.rl_method == 'ac':
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
        else:
            list_stock_code.append(stock_code)
            list_chart_data.append(chart_data)
            list_training_data.append(training_data)
            list_min_trading_unit.append(min_trading_unit)
            list_max_trading_unit.append(max_trading_unit)

    if args.rl_method == 'a3c':
        learner = A3CLearner(**{
            **common_params,
            'list_stock_code': list_stock_code,
            'list_chart_data': list_chart_data,
            'list_training_data': list_training_data,
            'list_min_trading_unit': list_min_trading_unit,
            'list_max_trading_unit': list_max_trading_unit,
            'value_network_path': value_network_path,
            'policy_network_path': policy_network_path})

        learner.run(balance=args.balance, num_epoches=args.num_epoches,
                    discount_factor=args.discount_factor,
                    start_epsilon=args.start_epsilon,
                    learning=args.learning)
        learner.save_models()