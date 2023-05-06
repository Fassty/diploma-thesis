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













def simulation(prices: np.ndarray, initial_balance: int, strategy: AbstractStrategy, fee: float = 0.0):
    base_balance = initial_balance
    quote_balance = 0
    action = 0  # 0|1|-1 -> HOLD|BUY|SELL
    current_price = prices[0]
    max_drop = hwm = initial_balance
    trajectory = []

    for price in tqdm(prices):
        current_price = price

        # BUY if sufficient funds
        if action == 1 and base_balance > 0:
            quote_balance = base_balance / current_price * (1 - fee)
            base_balance = 0
        # SELL if you have something to sell
        elif action == -1 and quote_balance != 0:
            base_balance = quote_balance * current_price * (1 - fee)
            quote_balance = 0

        # Retrieve the action from the strategy given the current price
        #
        # The action is always performed in the next step.
        # Which means that it will be performed with the price
        # in the next minute that can is different from the
        # price used for deciding the action.
        action = strategy.get_action(current_price)

        # Keep track of maximum drawdown, high watermark and trajectory
        # (the minimum and maximum of the balance/position)
        max_drop = min(base_balance + quote_balance * current_price, max_drop)
        hwm = max(base_balance + quote_balance * current_price, hwm)
        trajectory.append(base_balance + quote_balance * current_price)

    # If after the end of the simulation I have some open position,
    # close it for the current price
    if quote_balance > 0:
        base_balance = quote_balance * current_price * (1 - fee)
    return base_balance, max_drop, hwm, np.array(trajectory)


def plot_sim_results(prices: np.ndarray, trajectory: np.ndarray, log_scale=False, twin_axes=False):
    if not twin_axes:
        plt.figure(figsize=(8, 6))
        plt.plot(prices, label='Market price')
        plt.plot(trajectory, label='Wallet balance')
        plt.xlabel('Timestamp [s]')
        plt.ylabel('Value [$]')
        if log_scale:
            plt.yscale('log')
        plt.legend()
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(prices, label='Market price', color='C0')
        ax.set_xlabel('Timestamp [s]')
        ax.set_ylabel('Market price [$]')
        ax.legend()

        ax2 = ax.twinx()
        ax2.plot(trajectory, label='Wallet balance', color='C1')
        ax2.set_ylabel('Wallet balance [$]')
        ax2.legend()

    plt.title('Simulation results')
    plt.show()
