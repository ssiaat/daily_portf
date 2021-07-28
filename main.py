import os
import sys
import logging
import argparse
import json

import settings
import utils
from data_manager import *
import warnings

warnings.filterwarnings('ignore')
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

num_stocks = 200
num_steps = 10
value_name = None
policy_name = None
start_date = 20000201
# 학습이 이뤄지는 마지막 시점
criterion_year = 2015
# test가 이뤄지는 마지막 시점
end_year = 2015

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_stocks', type=int, default=num_stocks)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--net', choices=['dnn', 'lstm'], default='dnn')
    parser.add_argument('--discount_factor', type=float, default=0.9)
    parser.add_argument('--start_epsilon', type=float, default=0.3)
    parser.add_argument('--balance', type=int, default=1e9)
    parser.add_argument('--num_epoches', type=int, default=30)
    parser.add_argument('--hold_criter', type=float, default=0.)
    parser.add_argument('--delayed_reward_threshold', type=float, default=0.02)
    parser.add_argument('--output_name', default=utils.get_time_str())
    parser.add_argument('--value_network_name', default=value_name)
    parser.add_argument('--policy_network_name', default=policy_name)
    parser.add_argument('--reuse_models', action='store_true')
    parser.add_argument('--learning', action='store_true')
    parser.add_argument('--stationary', action='store_true')
    args = parser.parse_args()

    # Keras Backend 설정
    os.environ['KERAS_BACKEND'] = 'tensorflow'

    # 출력 경로 설정
    output_path = os.path.join(settings.BASE_DIR, 'output/{}'.format(args.output_name))
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

    # learning이면 연도별로 종목 리밸런싱 불필요
    # =>criterion과 end를 둔 이유는 사이 기간에 연도별 리밸런싱을 하기 위함
    if args.learning == True:
        print('This running is for training')
        end_year = criterion_year
    else:
        print('This running is for testing')
        args.num_epoches = 1

    if args.net == 'dnn':
        num_steps = 1

    # 로그, Keras Backend 설정을 먼저하고 RLTrader 모듈들을 이후에 임포트해야 함
    from learners import A2CLearner

    # 모델 경로 준비
    value_network_path = ''
    policy_network_path = ''
    if args.value_network_name is not None:
        value_network_path = os.path.join(settings.BASE_DIR,
                                          'models/{}.h5'.format(args.value_network_name))
    else:
        value_network_path = os.path.join(
            output_path, '{}_value.h5'.format(args.output_name))
    if args.policy_network_name is not None:
        policy_network_path = os.path.join(settings.BASE_DIR,
                                           'models/{}.h5'.format(args.policy_network_name))
    else:
        policy_network_path = os.path.join(
            output_path, '{}_policy.h5'.format(args.output_name))

    # 포트폴리오 구성 ticker정하고 데이터 불러옴
    # 리밸런싱이 있는 기간에 리밸런싱을 할 날짜 계산
    rebalance_date = set_rebalance_date(criterion_year, end_year)

    # 리밸런싱 날짜마다 종목구하고 전체 종목 universe 계산
    stock_codes_yearly, stock_codes = get_stock_codes(args.num_stocks, rebalance_date)
    print(f'yearly: {args.num_stocks} total: {len(stock_codes)} stocks in universe')
    price_data, cap_data, ks_data, training_data = make_data(stock_codes, start_date, rebalance_date[-1], args.stationary)

    # 공통 파라미터 설정
    common_params = {'stock_codes_yearly': stock_codes_yearly, 'stock_codes': stock_codes, 'num_features': len(training_data.columns), 'net':args.net,
                     'delayed_reward_threshold': args.delayed_reward_threshold, 'num_ticker': args.num_stocks, 'hold_criter': args.hold_criter,
                     'num_steps':num_steps, 'lr': args.lr,  'reuse_models': args.reuse_models, 'trainable': args.learning,
                     'price_data': price_data, 'cap_data': cap_data, 'ks_data' : ks_data, 'training_data': training_data,
                     'output_path': output_path, 'value_network_path': value_network_path, 'policy_network_path': policy_network_path}

    learner = A2CLearner(**{**common_params})
    if learner is not None:
        learner.run(balance=args.balance, num_epoches=args.num_epoches, discount_factor=args.discount_factor, start_epsilon=args.start_epsilon, learning=args.learning)
        learner.save_models()