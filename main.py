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

# Hyperparameters
num_stocks = 10  # universe에 존재하는 종목수
num_steps = 5     # lstm 모델에서 input의 기간(날짜 수)
start_year = 2000 # 시작 연도
end_year = 2005   # 종료 연도

lr = 0.001
net = 'dnn'
discount_factor = 0.9
start_epsilon = 0.3
balance = 1e9     # 초기 자본금
num_epoches = 30
hold_criter = 0.  # 포트폴리오 변동 줄이기 위해 hold_criter이하면 보유
delayed_reward_threshold = 0.02  # 학습이 이뤄지는 기준 수익률(이상, 이하면 학습 진행)

value_network1_name = None
value_network2_name = None
policy_network_name = None
output_name = utils.get_time_str()
reuse_models = False

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--stationary', action='store_true')
    args = parser.parse_args()

    # Keras Backend 설정
    os.environ['KERAS_BACKEND'] = 'tensorflow'

    # 출력 경로 설정
    output_path = os.path.join(settings.BASE_DIR, 'output/{}'.format(output_name))
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # 파라미터 기록
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        f.write(json.dumps(vars(args)))

    # 로그 기록 설정
    file_handler = logging.FileHandler(filename=os.path.join(output_path, "{}.log".format(output_name)), encoding='utf-8')
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s", handlers=[file_handler, stream_handler], level=logging.DEBUG)


    if args.test == True:
        print('This running is for testing')
        num_epoches = 1
        if value_network1_name is None:
            value_network1_name = net + '_v1'
            value_network2_name = net + 'v2'
            policy_network_name = net + '_policy'
        reuse_models = True
    else:
        print('This running is for training')
    start_date = indexes.index[indexes.index.year == start_year][0]

    if net == 'dnn':
        num_steps = 1

    # 로그, Keras Backend 설정을 먼저하고 RLTrader 모듈들을 이후에 임포트해야 함
    from learners import A2CLearner

    # 모델 경로 준비
    value_network_path = ''
    policy_network_path = ''
    if value_network1_name is not None:
        value_network1_path = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(value_network1_name))
        value_network2_path = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(value_network2_name))
        policy_network_path = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(policy_network_name))
    else:
        value_network1_path = os.path.join(output_path, '{}_v1.h5'.format(output_name))
        value_network2_path = os.path.join(output_path, '{}_v2.h5'.format(output_name))
        policy_network_path = os.path.join(output_path, '{}_policy.h5'.format(output_name))

    # 포트폴리오 구성 ticker정하고 데이터 불러옴
    # 리밸런싱이 있는 기간에 리밸런싱을 할 날짜 계산
    rebalance_date = set_rebalance_date(start_year-1, end_year)
    if not args.test:
        rebalance_date = [rebalance_date[-1]]
    else:
        rebalance_date = rebalance_date[:-1]

    # 리밸런싱 날짜마다 종목구하고 전체 종목 universe 계산
    stock_codes_yearly, stock_codes = get_stock_codes(num_stocks, rebalance_date)
    print(f'yearly: {num_stocks} total: {len(stock_codes)} stocks in universe')
    price_data, cap_data, index_data, index_ppc, training_data = make_data(stock_codes, start_date, rebalance_date[-1], args.stationary, args.test)

    # 공통 파라미터 설정
    common_params = {'stock_codes_yearly': stock_codes_yearly, 'stock_codes': stock_codes, 'num_features': len(training_data.columns), 'num_index':len(index_ppc.columns), 'net':net,
                     'delayed_reward_threshold': delayed_reward_threshold, 'num_ticker': num_stocks, 'hold_criter': hold_criter, 'num_steps':num_steps, 'lr': lr, 'test': args.test,
                     'price_data': price_data, 'cap_data': cap_data, 'index_data' : index_data, 'index_ppc':index_ppc, 'training_data': training_data, 'reuse_models': reuse_models,
                     'output_path': output_path, 'value_network1_path': value_network1_path, 'value_network2_path':value_network2_path, 'policy_network_path': policy_network_path}

    learner = A2CLearner(**{**common_params})
    if learner is not None:
        learner.run(balance=balance, num_epoches=num_epoches, discount_factor=discount_factor, start_epsilon=start_epsilon)
        learner.save_models()