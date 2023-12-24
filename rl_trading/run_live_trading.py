import os
import time
import argparse
import logging
import configparser

import numpy as np

from ray.rllib.algorithms import Algorithm

from rl_trading.data.api import BinanceAPI
from rl_trading.data.provider import LiveDataProvider
from rl_trading.simulation.env import LiveTradingEnv, StateConfig
from rl_trading.utils import setup_logging, load_model
from rl_trading.constants import ROOT_DIR

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data-file-path',
                    default='/home/fassty/Devel/school/diploma_thesis/code/data/binance_BTC_USDT_vwap.h5',
                    type=str, help='Path to the data file.')
parser.add_argument('-m', '--model-file-path',
                    default='/home/fassty/Devel/school/diploma_thesis/code/exp_results/production/discrete_env_15ind_normalized/R2D2_0_2023-11-19_17-54-27/checkpoint_000100',
                    type=str, help='Path to the model file.')
# Logging configuration
parser.add_argument('--log-level', default=logging.INFO, type=int, help='Log level.')
parser.add_argument('--log-to_stderr', default=True, action='store_true', help='Log to stderr.')


def run_live_trading(model: Algorithm, env: LiveTradingEnv):
    lstm_cell_size = model.config.model['lstm_cell_size']
    state = [np.zeros([lstm_cell_size], np.float32) for _ in range(2)]
    prev_action = 0
    prev_reward = 0

    while True:
        obs, _ = env.get_current_state()
        action, state, _ = model.compute_single_action(
            observation=obs,
            state=state,
            prev_action=prev_action,
            prev_reward=prev_reward,
            explore=False,
            full_fetch=True
        )
        print(action)
        time.sleep(1)


def main(args: argparse.Namespace):
    setup_logging(log_level=args.log_level, log_to_stderr=args.log_to_stderr)

    api_config = configparser.ConfigParser()
    api_config.read(os.path.join(ROOT_DIR, 'config.ini'))
    api_key = api_config['binance']['api_key']
    api_secret = api_config['binance']['api_secret']

    model = load_model(args.model_file_path)
    api = BinanceAPI(api_key, api_secret)
    env = LiveTradingEnv(
        api=api,
        data_file_path=args.data_file_path,
        state_config=StateConfig(**model.config.env_config['state_config']),
        trade_frequency='1min',
    )

    run_live_trading(model, env)


if __name__ == '__main__':
    main(parser.parse_args([] if "__file__" not in globals() else None))
