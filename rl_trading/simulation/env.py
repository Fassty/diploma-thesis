import os
import random
from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple, Dict, Callable
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.registration import EnvSpec

from rl_trading.data.api import BinanceAPI
from rl_trading.data.indicators import *
from rl_trading.data.provider import MarketDataProvider, LiveDataProvider
from rl_trading.enums import Action
import logging
import gymnasium as gym
from gymnasium import spaces

from rl_trading.utils import granularity_convert


logger = logging.getLogger('root')


@dataclass
class SimulationConfig:
    granularity: str = '1min'
    max_steps: int = 1440        # default 1 day (24 * 60 minutes)
    initial_cash: int = 10_000   # default 10,000$
    reward_type: str = 'absolute'  # default 'absolute'
    reward_scale: float = 1.0


@dataclass
class ExchangeConfig:
    maker_fee: float = 0.0
    taker_fee: float = 0.0
    slippage: float = 0.0


@dataclass
class StateConfig:
    market_state: List[str] = field(default_factory=lambda: ['price', 'volume'])
    technical_indicators: List[Tuple[TechnicalIndicator, IndicatorParams, str]] = field(default_factory=lambda: [
        (SMA, dict(timeperiod=20), '1min'),
        (EMA, dict(timeperiod=12), '1min'),
        (MACD_DIFF, dict(fastperiod=12, slowperiod=26, signalperiod=9), '1min'),
        (RSI, dict(timeperiod=14), '1min')
    ])
    account_state: List[str] = field(default_factory=lambda: ['cash_balance', 'asset_holdings'])

    def __len__(self) -> int:
        return len(self.market_state) + len(self.technical_indicators) + len(self.account_state)


class StockExchangeEnv0(gym.Env):
    def __init__(
        self,
        data_file_path: str = '/home/fassty/Devel/school/diploma_thesis/code/data/binance_BTC_USDT.h5',
        sim_config: Optional[dict] = None,
        exchange_config: Optional[dict] = None,
        state_config: Optional[dict] = None,
        # For debugging purposes, limit the sampled indices to set values
        _n_days: Optional[int] = None,
        seed: Optional[int] = None
    ):
        sim_config = SimulationConfig(**sim_config) if sim_config is not None else SimulationConfig()
        exchange_config = ExchangeConfig(**exchange_config) if exchange_config is not None else ExchangeConfig()
        state_config = StateConfig(**state_config) if state_config is not None else StateConfig()

        super().__init__()
        # Setup simulation
        # ================
        self.initial_cash = sim_config.initial_cash
        self.max_steps = sim_config.max_steps
        self.sim_granularity = sim_config.granularity
        self.reward_type = sim_config.reward_type
        self.reward_scale = sim_config.reward_scale

        # Load market data
        # ================
        data_provider = MarketDataProvider(data_file_path)
        self.market_data = data_provider.get_market_data(self.sim_granularity, state_config.technical_indicators)
        self.price_data = {gran: data['price'].to_numpy() for gran, data in self.market_data.items()}

        # Setup exchange params
        # ================
        self.trading_fee = exchange_config.maker_fee
        self.slippage = exchange_config.slippage

        # Setup environment
        # ================
        self.state_config = state_config
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.state_config),), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        # Set default values
        # ================
        self.padding = self._calculate_padding(sim_config, state_config)
        self.start_idx = self.padding
        self.i = 0
        self.cash_balance = self.initial_cash
        self.asset_holdings = 0
        self.balance_history = []
        self.action_history = []
        self.reward_history = []
        self.net_worth_changes = []

        if _n_days:
            rng = np.random.default_rng(seed=seed)
            self._idxs_range = rng.integers(self.padding, len(self.price_data[self.sim_granularity]) - self.max_steps * _n_days - 1, _n_days)
        else:
            self._idxs_range = None

        self.reset()

    def _calculate_padding(self, sim_config, state_config):
        max_padding = 0
        for ind, ind_cfg, gran in state_config.technical_indicators:
            padding = indicator_padding[ind.__name__](ind_cfg)
            padding *= granularity_convert(gran, sim_config.granularity)
            max_padding = max(max_padding, padding)
        return int(max_padding)

    def reset(self, *, seed=None, options=None):
        if self._idxs_range is not None:
            self.start_idx = random.choice(self._idxs_range)
        else:
            if seed is not None:
                rng = np.random.default_rng(seed=seed)
            else:
                rng = self.np_random
            # -1 to prevent IndexError when accessing the next price
            self.start_idx = rng.integers(self.padding, len(self.price_data[self.sim_granularity]) - self.max_steps)
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

        current_price = self.price_data[self.sim_granularity][self.current_idx]
        next_price = self.price_data[self.sim_granularity][self.current_idx + 1]
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
        market_state = [self.market_data[self.sim_granularity][key].iloc[self.current_idx] / self.price_data[self.sim_granularity][self.current_idx]
                        for key in self.state_config.market_state]

        # Technical indicators
        # Compute the technical indicators for the current trading window and get the value for current state
        technical_indicators = []
        for ind_func, ind_params, gran in self.state_config.technical_indicators:
            if gran != self.sim_granularity:
                start_ts: pd.Timestamp = self.market_data[self.sim_granularity].index[self.start_idx]
                current_ts: pd.Timestamp = self.market_data[self.sim_granularity].index[self.current_idx]
                start_idx = self.market_data[gran].index.get_loc(
                    start_ts.floor(freq=gran)
                )
                current_idx = self.market_data[gran].index.get_loc(
                    current_ts.floor(freq=gran)
                )
                ind_padding = indicator_padding[ind_func.__name__](ind_params)
                ti = ind_func(self.price_data[gran][start_idx - ind_padding: current_idx + 1], **ind_params)
            else:
                ind_padding = indicator_padding[ind_func.__name__](ind_params)
                ti = ind_func(
                    self.price_data[gran][self.start_idx - ind_padding: self.current_idx + 1],
                    **ind_params
                )
            technical_indicators.append(ti)

        if any(np.isnan(technical_indicators)):
            raise ValueError(f'NaN value encountered when computing technical indicators: {technical_indicators}')

        # Account state
        account_state = [getattr(self, attr_) for attr_ in self.state_config.account_state]
        account_state[0] /= self.initial_cash

        observation = np.array(market_state + technical_indicators + account_state)
        return observation

    def _get_reward(self):
        current_net_worth = self.cash_balance + self.asset_holdings * self.price_data[self.sim_granularity][self.current_idx]
        if len(self.net_worth_history) > 0:
            previous_net_worth = self.net_worth_history[-1]
        else:
            previous_net_worth = self.initial_cash
        self.net_worth_history.append(current_net_worth)  # Add the net worth change to the list
        if self.reward_type == 'absolute':
            reward = current_net_worth - previous_net_worth
        elif self.reward_type == 'relative':
            reward = (current_net_worth - previous_net_worth) / previous_net_worth
        else:
            raise ValueError(f'Unknown reward type: {self.reward_type}')
        return reward * self.reward_scale

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
            y_data = self.price_data[self.sim_granularity][self.start_idx:self.current_idx]
            self.ax1_line.set_data(x_data, y_data)
            self.ax2_line.set_data(x_data, self.balance_history[:self.i])
            self.ax3_line.set_data(x_data, self.reward_history[:self.i])
            self.ax4_line.set_data(x_data, self.action_history[:self.i])
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.relim()
                ax.autoscale_view(True, True, True)
            plt.pause(0.01)


class StockExchangeEnv1(StockExchangeEnv0):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action)
        action = action[0]

        current_price = self.price_data[self.sim_granularity][self.current_idx]
        next_price = self.price_data[self.sim_granularity][self.current_idx + 1]
        slippage = current_price * self.slippage * np.sign(next_price - current_price)
        execution_price = current_price + slippage

        self.balance_history.append(self.cash_balance + self.asset_holdings * current_price)

        if action > 0:
            amount_to_buy = (action * self.cash_balance) / execution_price
            cost = amount_to_buy * execution_price
            self.cash_balance -= cost
            self.asset_holdings += amount_to_buy * (1 - self.trading_fee)
        elif action < 0:
            amount_to_sell = self.asset_holdings * np.abs(action)
            revenue = amount_to_sell * execution_price * (1 - self.trading_fee)
            self.cash_balance += revenue
            self.asset_holdings -= amount_to_sell
        self.action_history.append(action)

        self.i += 1
        done = self.i == self.max_steps
        reward = self._get_reward()
        self.reward_history.append(reward)

        return self._get_observation(), reward, done, False, {}


class LiveTradingEnv:
    def __init__(
        self,
        api: BinanceAPI,
        data_file_path: str,
        state_config: StateConfig,
        trade_frequency: str = '1min',
    ):
        self.api = api
        self.padding = self._calculate_padding(trade_frequency, state_config)
        self.data_provider = LiveDataProvider(data_file_path, self.padding)
        self.data_provider.update_data(self.api)
        self.state_config = state_config
        self.trade_frequency = trade_frequency

        self.initial_cash, self.asset_holdings = self._get_balance_and_position()
        self.cash_balance = self.initial_cash

        self.previous_net_worth_change = 0

    def _get_balance_and_position(self):
        account_info = self.api.get_account_info()

        balances_df = pd.DataFrame(account_info['balances'])
        usdt_balance = float(balances_df[balances_df['asset'] == 'USDT']['free'].item())
        btc_balance = float(balances_df[balances_df['asset'] == 'BTC']['free'].item())

        return usdt_balance, btc_balance

    def _calculate_padding(self, granularity, state_config):
        max_padding = 0
        for ind, ind_cfg, gran in state_config.technical_indicators:
            padding = indicator_padding[ind.__name__](ind_cfg)
            padding *= granularity_convert(gran, granularity)
            max_padding = max(max_padding, padding)
        return int(max_padding)

    def get_current_state(self):
        self.data_provider.update_data(self.api)

        market_data = self.data_provider.get_market_data(self.trade_frequency, self.state_config.technical_indicators)
        price_data = {gran: np.array(data.iloc[-self.padding - 1:]['price']) for gran, data in market_data.items()}

        # Market state
        market_state = [market_data[self.trade_frequency][key].iloc[-1] / price_data[self.trade_frequency][-1]
                        for key in self.state_config.market_state]

        # Technical indicators
        # Compute the technical indicators for the current trading window and get the value for current state
        technical_indicators = []
        for ind_func, ind_params, gran in self.state_config.technical_indicators:
            ind_padding = indicator_padding[ind_func.__name__](ind_params)
            ti = ind_func(price_data[gran][-ind_padding - 1:], **ind_params)
            technical_indicators.append(ti)

        if any(np.isnan(technical_indicators)):
            raise ValueError(f'NaN value encountered when computing technical indicators: {technical_indicators}')

        # Account state
        self.cash_balance, self.asset_holdings = self._get_balance_and_position()
        account_state = [getattr(self, attr_) for attr_ in self.state_config.account_state]
        account_state[0] /= self.initial_cash

        observation = np.array(market_state + technical_indicators + account_state)
        return observation, price_data[self.trade_frequency][-1]

    def get_reward(self, last_price: float):
        current_net_worth_change = self.cash_balance + self.asset_holdings * last_price - self.initial_cash
        reward = current_net_worth_change - self.previous_net_worth_change

        self.previous_net_worth_change = current_net_worth_change

        return reward
