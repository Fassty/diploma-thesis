from abc import abstractmethod, ABC
from typing import Optional, Union, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from gymnasium.core import RenderFrame, ObsType, ActType
from tqdm.auto import tqdm
from rl_trading.enums import Action
import logging
from rl_trading.strategy.strategy import AbstractStrategy
import gymnasium as gym
from gymnasium import spaces
import talib

logger = logging.getLogger('root')


class SimulationEnvBase(gym.Env, ABC):
    def __init__(self, prices: np.ndarray, initial_balance: int, *args, **kwargs):
        self.prices = prices
        self.initial_balance = initial_balance

    @abstractmethod
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        pass

    @abstractmethod
    def reset(self, seed: Optional[int] = None, **kwargs):
        pass

    @abstractmethod
    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass


class TradingEnvActionDiscrete(SimulationEnvBase):
    def __init__(self, prices: np.ndarray, initial_balance: int, fee: float = 0.0):
        super().__init__(prices, initial_balance)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,))
        self.action_space = spaces.Discrete(3)
        self.fee = fee
        self.i, self.balance, self.position, self.stats = [None] * 4
        self.metrics = {
            'sma': None,
            'ema': None,
            'macd': None,
            'rsi': None,
        }
        self.done = False
        self.reset()

    def step(self, action: int) -> Tuple[ObsType, float, bool, bool, dict]:
        if self.done:
            logger.log(msg='Step called on finished env!', level=logging.WARNING)
            return self.prices[0], 0, True, False, {}
        current_price = self.prices[self.i]

        if action == Action.BUY:
            buy_amount = 1 * self.balance
            self.position += (buy_amount / current_price) * (1 - self.fee)
            self.balance = 0
        elif action == Action.SELL:
            sell_amount = 1 * self.position
            self.balance += sell_amount * current_price * (1 - self.fee)
            self.position = 0

        reward = self.calculate_reward(current_price)
        window_size = 100
        self.calculate_metrics(window_size)

        self.i += 1
        if self.i == len(self.prices):
            self.done = True

        state = (
            self.balance,
            self.prices[self.i % len(self.prices)],
            self.position,
            self.metrics['sma'],
            self.metrics['ema']
        )

        return state, reward, self.done, False, self.metrics

    def reset(self, seed: Optional[int] = None, **kwargs):
        self.i = 0
        self.balance = self.initial_balance
        self.position = 0
        self.stats = {}
        self.done = False
        self.metrics = {
            'sma': 0,
            'ema': 0,
            'macd': 0,
            'rsi': 0,
        }
        state = (
            self.balance,
            self.prices[self.i],
            self.position,
            self.metrics['sma'],
            self.metrics['ema']
        )
        return state, self.metrics

    def calculate_reward(self, current_price: float):
        # Penalize the model for not trading
        # if self.quote_balance == 0:
        #     return -0.001
        if self.i == 0:
            return 0

        # TODO: this may not be ideal as the negative returns may outweigh the large positive return if the model were
        #       to long for longer
        rpc = (self.prices[self.i] - self.prices[self.i - 1]) / self.prices[self.i - 1]
        position_value = self.position * current_price
        return rpc * position_value

    def calculate_metrics(self, window_size: int):
        self.metrics['sma'] = np.mean(self.prices[self.i - min(self.i, window_size):self.i + 1])

        if self.metrics['ema'] == 0:
            self.metrics['ema'] = self.metrics['sma']
        else:
            k = 2 / (window_size + 1.0)
            self.metrics['ema'] = k * self.prices[self.i] + self.metrics['ema'] * (1 - k)

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass









class StockExchangeEnv(gym.Env):
    def __init__(
            self,
            price_data: np.ndarray,
            volume_data: np.ndarray,
            initial_cash: int,
            max_steps: int = 720,
            trading_fee: float = 0.0
    ):
        super().__init__()

        self.price_data = price_data
        self.volume_data = volume_data
        self.initial_cash = initial_cash
        self.trading_fee = trading_fee
        self.max_steps = max_steps

        self.balance_history = []
        self.action_history = []
        self.reward_history = []
        self.net_worth_changes = []

        self.start_step = 0
        self.current_step = 0
        self.cash_balance = self.initial_cash
        self.asset_holdings = 0

        # Define action space: 0 - hold, 1 - buy, 2 - sell
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            rng = np.random.default_rng(seed=seed)
            self.current_step = rng.integers(0, len(self.price_data) - self.max_steps)
            self.start_step = self.current_step
        else:
            self.current_step = np.random.randint(len(self.price_data) - self.max_steps)
            self.start_step = self.current_step
        self.cash_balance = self.initial_cash
        self.asset_holdings = 0
        return self._get_observation(), {}

    def step(self, action):
        assert self.action_space.contains(action)

        current_price = self.price_data[self.current_step]
        current_volume = self.volume_data[self.current_step]

        self.balance_history.append(self.cash_balance + self.asset_holdings * current_price)

        if action == 1:  # Buy
            amount_to_buy = (1 * self.cash_balance) / current_price
            cost = amount_to_buy * current_price
            self.cash_balance -= cost
            self.asset_holdings += amount_to_buy * (1 - self.trading_fee)
            self.action_history.append(1)
        elif action == 2:  # Sell
            amount_to_sell = self.asset_holdings
            revenue = amount_to_sell * current_price * (1 - self.trading_fee)
            self.cash_balance += revenue
            self.asset_holdings = 0
            self.action_history.append(-1)
        else:
            self.action_history.append(0)

        self.current_step += 1
        done = self.current_step == self.start_step + self.max_steps
        reward = self._get_reward()
        self.reward_history.append(reward)

        return self._get_observation(), reward, done, False, {}

    def _get_observation(self):
        current_price = self.price_data[self.current_step]
        current_volume = self.volume_data[self.current_step]

        short_mavg = talib.SMA(self.price_data[self.start_step:self.current_step + 1], timeperiod=5)[-1] if self.current_step - self.start_step >= 4 else current_price
        long_mavg = talib.SMA(self.price_data[self.start_step:self.current_step + 1], timeperiod=20)[-1] if self.current_step - self.start_step >= 19 else current_price

        if self.current_step - self.start_step >= 26:
            macd, macd_signal, _ = talib.MACD(self.price_data[self.start_step:self.current_step + 1], fastperiod=12, slowperiod=26, signalperiod=9)
            macd_diff = macd[-1] - macd_signal[-1]
            if np.isnan(macd_diff):
                macd_diff = 0
        else:
            macd_diff = 0

        rsi = talib.RSI(self.price_data[self.start_step:self.current_step + 1], timeperiod=14)[-1] if self.current_step - self.start_step >= 14 else 50

        ema = talib.EMA(self.price_data[self.start_step:self.current_step + 1], timeperiod=12)[-1] if self.current_step - self.start_step >= 11 else current_price

        observation = np.array([current_price, current_volume, short_mavg, long_mavg, macd_diff, rsi, ema, self.cash_balance, self.asset_holdings])
        return observation

    def _get_reward(self):
        current_net_worth_change = self.cash_balance + self.asset_holdings * self.price_data[self.current_step] - self.initial_cash
        if len(self.net_worth_changes) > 0:
            previous_net_worth_change = self.net_worth_changes[-1]
        else:
            previous_net_worth_change = 0
        self.net_worth_changes.append(current_net_worth_change)  # Add the net worth change to the list
        return current_net_worth_change - previous_net_worth_change

    def render(self, mode='human'):
        if self.current_step == self.start_step:
            self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) = \
                plt.subplots(4, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [3, 3, 3, 1]}, sharex=True)
            self.ax1.set_title('Stock Price')
            self.ax1.set_xlabel('Step')
            self.ax1.set_ylabel('Price')

            self.ax2.set_title('Portfolio Value')
            self.ax2.set_xlabel('Step')
            self.ax2.set_ylabel('Amount')

            self.ax3.set_title('Rewards')
            self.ax3.set_xlabel('Step')
            self.ax3.set_ylabel('Sharpe Ratio')

            self.ax4.set_title('Action History')
            self.ax4.set_xlabel('Step')
            self.ax4.set_ylabel('Action')
        else:
            x_data = np.arange(self.current_step - self.start_step)
            y_data = self.price_data[self.start_step:self.current_step]
            self.ax1.plot(x_data, y_data, color='C0')
            self.ax2.plot(x_data, self.balance_history[:self.current_step - self.start_step], color='C1')
            self.ax3.plot(x_data, self.reward_history[:self.current_step - self.start_step], color='C1')
            self.ax4.plot(x_data, self.action_history[:self.current_step - self.start_step], color='C1')
            plt.pause(0.01)
