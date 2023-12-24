import time

import ray
import schedule
import numpy as np
import pandas as pd

from rl_trading.data.api import BinanceAPI
from rl_trading.enums import Action
from rl_trading.simulation.env import LiveTradingEnv, StateConfig, BacktestingEnv
from rl_trading.utils import load_model


class TradingAgent:
    def __init__(self, api_key, secret_key, data_source, model_path):
        self.api = BinanceAPI(api_key, secret_key)
        ray.shutdown()
        self.model = load_model(model_path)
        state_config = StateConfig(**self.model.config.env_config['state_config'])
        self.env = LiveTradingEnv(self.api, data_source, state_config)

        self.rnn_state = [np.zeros([256], np.float32) for _ in range(2)]
        self.prev_action = 0
        self.prev_reward = 0
        print('Agent initialized')

    def step(self):
        observation, last_price = self.env.get_current_state()
        self.prev_reward = self.env.get_reward(last_price)
        balance = observation[-2]
        position = observation[-1]

        action, rnn_state_next, _ = self.model.compute_single_action(
            observation=observation,
            state=self.rnn_state,
            prev_action=self.prev_action,
            prev_reward=self.prev_reward
        )

        print(f'Selected action: {action}')

        if action == Action.BUY and balance > 0:
            ...
        elif action == Action.SELL and position > 0:
            ...

        self.rnn_state = rnn_state_next
        self.prev_action = action

    def run(self):
        print('Running...')
        self.env.data_provider.update_data(self.api)
        schedule.every().minute.at(":00").do(self.step)

        while True:
            schedule.run_pending()
            time.sleep(1)


class BacktestingAgent:
    def __init__(self, rl_pred_path: str):
        self.rl_preds = pd.read_csv(rl_pred_path, index_col=0)
        self.rl_preds.index = pd.to_datetime(self.rl_preds.index)

    def get_action(self, timestamp: pd.Timestamp) -> int:
        return self.rl_preds.loc[timestamp].item()
