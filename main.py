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
num_stocks = 200  # universe에 존재하는 종목수
num_steps = 5     # lstm 모델에서 input의 기간(날짜 수)
start_year = 2004 # 시작 연도
end_year = 2015   # 종료 연도

lr = 1e-3
net = 'dnn'
discount_factor = 0.9
balance = 1e10     # 초기 자본금
num_epoches = 2
hold_criter = 0.  # 포트폴리오 변동 줄이기 위해 hold_criter이하면 보유

value_network1_name = None
value_network2_name = None
target_value_network1_name = None
target_value_network2_name = None
policy_network_name = None
output_name = utils.get_time_str()
reuse_models = False

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--stationary', action='store_true')
    parser.add_argument('--clip', action='store_true')
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

    is_clip = '_clip' if args.clip else ''

    if args.test == True:
        print('This running is for testing')
        num_epoches = 1
        if value_network1_name is None:
            value_network1_name = net + is_clip + '_v1'
            value_network2_name = net + is_clip + '_v2'
            target_value_network1_name = net + is_clip + '_tv1'
            target_value_network2_name = net + is_clip + '_tv2'
            policy_network_name = net + is_clip + '_p'

        reuse_models = True
    else:
        print('This running is for training')
    start_date = indexes.index[indexes.index.year == start_year][0]
    end_date = indexes.index[indexes.index.year == end_year][-1]

    if net == 'dnn':
        num_steps = 1

    # 로그, Keras Backend 설정을 먼저하고 RLTrader 모듈들을 이후에 임포트해야 함
    from learners import A2CLearner

    # 모델 경로 준비
    if value_network1_name is not None:
        value_network1_path = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(value_network1_name))
        value_network2_path = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(value_network2_name))
        target_value_network1_path = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(target_value_network1_name))
        target_value_network2_path = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(target_value_network2_name))
        policy_network_path = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(policy_network_name))
    else:
        value_network1_path = os.path.join(output_path, '{}_v1.h5'.format(output_name))
        value_network2_path = os.path.join(output_path, '{}_v2.h5'.format(output_name))
        target_value_network1_path = os.path.join(output_path, '{}_tv1.h5'.format(output_name))
        target_value_network2_path = os.path.join(output_path, '{}_tv2.h5'.format(output_name))
        policy_network_path = os.path.join(output_path, '{}_p.h5'.format(output_name))

    price_data, index_data, index_ppc, training_data = make_data(start_date, end_date, args.stationary, args.test)
    capital = capital.loc[price_data.index]
    print(price_data)
    print(training_data)
    exit()
    # 공통 파라미터 설정
    common_params = {'num_features': len(training_data.columns), 'num_index' : len(index_ppc.columns), 'net':net, 'clip' : args.clip,
                     'num_ticker': num_stocks, 'hold_criter': hold_criter, 'num_steps':num_steps, 'lr': lr, 'test': args.test, 'reuse_models': reuse_models,
                     'price_data': price_data, 'cap_data':capital, 'index_data' : index_data, 'index_ppc':index_ppc, 'training_data': training_data,
                     'output_path': output_path, 'value_network1_path': value_network1_path, 'value_network2_path':value_network2_path,
                     'target_value_network1_path': target_value_network1_path, 'target_value_network2_path': target_value_network2_path, 'policy_network_path': policy_network_path}

    learner = A2CLearner(**{**common_params})
    if learner is not None:
        learner.run(balance=balance, num_epoches=num_epoches)
        learner.save_models()