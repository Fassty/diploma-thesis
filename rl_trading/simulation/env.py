from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple, Dict, Callable
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from rl_trading.enums import Action
import logging
import gymnasium as gym
from gymnasium import spaces
import talib
import time


TechnicalIndicator = Callable
IndicatorParams = Dict[str, int]


logger = logging.getLogger('root')


@dataclass
class SimulationConfig:
    granularity: str = '1m'
    max_steps: int = 1440       # default 1 day (24 * 60 minutes)
    initial_cash: int = 10_000   # default 10,000$


@dataclass
class ExchangeConfig:
    maker_fee: float = 0.0
    taker_fee: float = 0.0
    slippage: float = 0.0


def MACD(data, fastperiod=12, slowperiod=26, signalperiod=9):
    macd, _, _ = talib.MACD(data, fastperiod, slowperiod, signalperiod)
    return macd


def MACD_DIFF(data, fastperiod=12, slowperiod=26, signalperiod=9):
    macd, macd_signal, _ = talib.MACD(data, fastperiod, slowperiod, signalperiod)
    return macd - macd_signal


indicator_padding: Dict[TechnicalIndicator, Callable[[Dict], int]] = {
    talib.SMA: lambda config: config['timeperiod'],
    talib.EMA: lambda config: config['timeperiod'],
    talib.RSI: lambda config: config['timeperiod'],
    MACD: lambda config: (config['slowperiod'] - 1) + (config['signalperiod'] - 1),
    MACD_DIFF: lambda config: (config['slowperiod'] - 1) + (config['signalperiod'] - 1),
}


@dataclass
class StateConfig:
    market_state: List[str] = field(default_factory=lambda: ['price', 'amount'])
    technical_indicators: Dict[TechnicalIndicator, IndicatorParams] = field(default_factory=lambda: {
        talib.SMA: dict(timeperiod=20),
        talib.EMA: dict(timeperiod=12),
        MACD_DIFF: dict(fastperiod=12, slowperiod=26, signalperiod=9),
        talib.RSI: dict(timeperiod=14),
    })
    account_state: List[str] = field(default_factory=lambda: ['cash_balance', 'asset_holdings'])

    def __len__(self) -> int:
        return len(self.market_state) + len(self.technical_indicators) + len(self.account_state)


class StockExchangeEnv(gym.Env):
    def __init__(
            self,
            market_data: pd.DataFrame,
            sim_config: SimulationConfig = SimulationConfig(),
            exchange_config: ExchangeConfig = ExchangeConfig(),
            state_config: StateConfig = StateConfig()
    ):
        super().__init__()

        self.initial_cash = sim_config.initial_cash
        self.max_steps = sim_config.max_steps
        if sim_config.granularity != '1m':
            agg_mapping = {'price': 'last', 'amount': 'sum'}
            market_data = market_data.groupby(pd.Grouper(freq=sim_config.granularity)).agg(agg_mapping).copy()

        self.market_data = market_data
        self.price_data = market_data['price'].to_numpy()
        self.volume_data = market_data['amount'].to_numpy()

        self.trading_fee = exchange_config.maker_fee
        self.slippage = exchange_config.slippage

        self.balance_history = []
        self.action_history = []
        self.reward_history = []
        self.net_worth_changes = []

        self.padding = max(
            [indicator_padding[ind](indicator_cfg) for ind, indicator_cfg in state_config.technical_indicators.items()],
            default=0
        )
        self.start_idx = self.padding
        self.i = 0
        self.cash_balance = self.initial_cash
        self.asset_holdings = 0

        self.state_config = state_config
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.state_config),), dtype=np.float32)

        # Define action space: 0 - hold, 1 - buy, 2 - sell
        self.action_space = spaces.Discrete(3)

        self.reset()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = self.np_random
        # -1 to prevent IndexError when accessing the next price
        self.start_idx = rng.integers(self.padding, len(self.price_data) - self.max_steps - 1)
        self.i = 0
        self.cash_balance = self.initial_cash
        self.asset_holdings = 0

        self.balance_history = []
        self.action_history = []
        self.reward_history = []
        self.net_worth_changes = []
        return self._get_observation(), {}

    @property
    def current_idx(self):
        return self.start_idx + self.i

    def step(self, action: int):
        assert self.action_space.contains(action)

        current_price = self.price_data[self.current_idx]
        next_price = self.price_data[self.current_idx + 1]
        slippage = current_price * self.slippage * np.sign(next_price - current_price)
        execution_price = current_price + slippage

        self.balance_history.append(self.cash_balance + self.asset_holdings * current_price)

        if action == Action.BUY:
            amount_to_buy = (1 * self.cash_balance) / execution_price
            cost = amount_to_buy * execution_price
            self.cash_balance -= cost
            self.asset_holdings += amount_to_buy * (1 - self.trading_fee)
            self.action_history.append(1)
        elif action == Action.SELL:
            amount_to_sell = self.asset_holdings
            revenue = amount_to_sell * execution_price * (1 - self.trading_fee)
            self.cash_balance += revenue
            self.asset_holdings = 0
            self.action_history.append(-1)
        else:
            self.action_history.append(0)

        self.i += 1
        done = self.i == self.max_steps
        reward = self._get_reward()
        self.reward_history.append(reward)

        return self._get_observation(), reward, done, False, {}

    def _get_observation(self):
        # Market state
        market_state = [self.market_data[key].iloc[self.current_idx] for key in self.state_config.market_state]

        # Technical indicators
        technical_indicators = [
            # Compute the technical indicators for the current trading window and get the value for current state
            ind_func(self.price_data[self.start_idx - self.padding: self.current_idx + 1], **ind_params)[-1]
            for ind_func, ind_params in self.state_config.technical_indicators.items()
        ]
        if any(np.isnan(technical_indicators)):
            raise ValueError(f'NaN value encountered when computing technical indicators: {technical_indicators}')

        # Account state
        account_state = [getattr(self, attr_) for attr_ in self.state_config.account_state]

        observation = np.array(market_state + technical_indicators + account_state)
        return observation

    def _get_reward(self):
        current_net_worth_change = self.cash_balance + self.asset_holdings * self.price_data[self.current_idx] - self.initial_cash
        if len(self.net_worth_changes) > 0:
            previous_net_worth_change = self.net_worth_changes[-1]
        else:
            previous_net_worth_change = 0
        self.net_worth_changes.append(current_net_worth_change)  # Add the net worth change to the list
        reward = current_net_worth_change - previous_net_worth_change
        return reward

    def render(self, mode='human'):
        if self.i == 0:
            plt.ion()
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

            self.ax1_line, = self.ax1.plot([], [], color='C0')
            self.ax2_line, = self.ax2.plot([], [], color='C1')
            self.ax3_line, = self.ax3.plot([], [], color='C1')
            self.ax4_line, = self.ax4.plot([], [], color='C1')
            self.fig.tight_layout()
        else:
            x_data = np.arange(self.i)
            y_data = self.price_data[self.start_idx:self.current_idx]
            self.ax1_line.set_data(x_data, y_data)
            self.ax2_line.set_data(x_data, self.balance_history[:self.i])
            self.ax3_line.set_data(x_data, self.reward_history[:self.i])
            self.ax4_line.set_data(x_data, self.action_history[:self.i])
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.relim()
                ax.autoscale_view(True, True, True)
            plt.pause(0.01)
