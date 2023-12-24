import random

import backtrader as bt
import numpy as np
import tqdm
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA

from rl_trading.enums import Action
from rl_trading.strategy.agent import BacktestingAgent


class BaseStrategy(bt.Strategy):
    def __init__(self):
        self.data_log = []
        self.final_value = None

    def next(self):
        self.data_log.append({
            'datetime': self.datas[0].datetime.datetime(0),
            'balance': self.broker.get_cash(),
            'position': self.getposition().size,
            'value': self.broker.getvalue()
        })

    def prenext(self):
        self.data_log.append({
            'datetime': self.datas[0].datetime.datetime(0),
            'balance': self.broker.get_cash(),
            'position': self.getposition().size,
            'value': self.broker.getvalue()
        })

    def stop(self):
        self.final_value = self.broker.getvalue()
        print(f'Final RQ: {self.final_value}')


class BuyAndHoldStrategy(BaseStrategy):
    def next(self):
        super().next()
        if len(self) == 1:
            self.buy()


class MovingAverageCrossoverStrategy(BaseStrategy):
    params = (
        ('short_term', 10),
        ('long_term', 30),
    )

    def __init__(self):
        super().__init__()
        self.sma_short = bt.indicators.SimpleMovingAverage(self.data.close,
                                                          period=self.params.short_term)
        self.sma_long = bt.indicators.SimpleMovingAverage(self.data.close,
                                                         period=self.params.long_term)
        self.crossover = bt.indicators.CrossOver(self.sma_short, self.sma_long)

    def next(self):
        super().next()
        if not self.position:
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            self.close()


class MeanReversionStrategy(BaseStrategy):
    params = (
        ('period', 15),
        ('devfactor', 1.5),  # May tweak based on volatility
    )

    def __init__(self):
        super().__init__()
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.period)
        self.stddev = bt.indicators.StandardDeviation(self.data.close, period=self.params.period)

        self.upper_band = self.sma + self.stddev * self.params.devfactor
        self.lower_band = self.sma - self.stddev * self.params.devfactor

    def next(self):
        super().next()
        if not self.position:
            if self.data.close[0] < self.lower_band[0]:
                self.buy()
        elif self.data.close[0] > self.upper_band[0]:
            self.close()


class MomentumStrategy(BaseStrategy):
    params = (
        ('momentum_period', 20),  # Period for momentum calculation
    )

    def __init__(self):
        super().__init__()
        # Momentum indicator: difference between current close and close 'momentum_period' bars ago
        self.momentum = self.data.close - self.data.close(-self.params.momentum_period)

    def next(self):
        super().next()
        if not self.position:
            if self.momentum[0] > 0:
                self.buy()
        elif self.momentum[0] < 0:
            self.close()


class RandomStrategy(BaseStrategy):
    plotinfo = dict(subplot=True, plotname='Random')

    def __init__(self):
        super().__init__()
        # Nothing to initialize for a random strategy
        self.plotinfo.plotname = 'Random'

    def next(self):
        super().next()
        action = random.choice(['buy', 'sell', 'hold'])

        if action == 'buy' and not self.position:
            self.buy()
        elif action == 'sell' and self.position:
            self.close()
        # 'hold' implies doing nothing


class BreakoutStrategy(BaseStrategy):
    params = (('lookback', 20), ('enter_perc', 0.01), ('exit_perc', 0.01),)

    def __init__(self):
        super().__init__()
        self.order = None
        self.price_high = bt.ind.Highest(self.data.high(-1), period=self.p.lookback)
        self.price_low = bt.ind.Lowest(self.data.low(-1), period=self.p.lookback)

    def next(self):
        super().next()
        if self.order:
            return  # Pending order execution

        if not self.position:
            if self.data.close > self.price_high * (1 + self.p.enter_perc):
                self.order = self.buy()
        else:
            if self.data.close < self.price_low * (1 - self.p.exit_perc):
                self.order = self.close()


class PairsTradingStrategy(BaseStrategy):
    params = (('lookback', 30), ('zscore_low', -2.0), ('zscore_high', 2.0),)

    def __init__(self):
        super().__init__()
        # Assuming self.datas[0] and self.datas[1] are the paired securities
        self.spread = self.datas[0].close - self.datas[1].close
        self.zscore = (self.spread - bt.ind.SMA(self.spread, period=self.p.lookback)) / bt.ind.StandardDeviation(self.spread, period=self.p.lookback)

    def next(self):
        super().next()
        if self.zscore < self.p.zscore_low:
            self.buy(data=self.datas[0])
            self.sell(data=self.datas[1])
        elif self.zscore > self.p.zscore_high:
            self.close(data=self.datas[0])
            self.close(data=self.datas[1])


class BollingerBandsStrategy(BaseStrategy):
    params = (('period', 20), ('devfactor', 2),)

    def __init__(self):
        super().__init__()
        self.boll = bt.indicators.BollingerBands(period=self.p.period, devfactor=self.p.devfactor)

    def next(self):
        super().next()
        if not self.position:
            if self.data.close < self.boll.lines.bot:
                self.buy()
        elif self.data.close > self.boll.lines.top:
            self.close()


class VolatilityAdjustedDMAC(BaseStrategy):
    params = (
        ('fast1', 5),
        ('slow1', 25),
        ('fast2', 15),
        ('slow2', 49),
        ('volatility_period', 14),
        ('volatility_threshold', 0.0005)  # Adjust based on the BTC/USDT characteristics
    )

    def __init__(self):
        super().__init__()
        # Moving averages
        self.sma_fast1 = bt.indicators.SMA(self.data.close, period=self.p.fast1)
        self.sma_slow1 = bt.indicators.SMA(self.data.close, period=self.p.slow1)
        self.sma_fast2 = bt.indicators.SMA(self.data.close, period=self.p.fast2)
        self.sma_slow2 = bt.indicators.SMA(self.data.close, period=self.p.slow2)

        # Crossover signals
        self.crossover1 = bt.ind.CrossOver(self.sma_fast1, self.sma_slow1)
        self.crossover2 = bt.ind.CrossOver(self.sma_fast2, self.sma_slow2)

        # Volatility indicator
        self.atr = bt.indicators.ATR(self.data, period=self.p.volatility_period)

    def next(self):
        super().next()
        # Check if volatility is above the threshold
        if self.atr[0] > self.p.volatility_threshold:
            # Buy signal
            if not self.position and self.crossover1 > 0 and self.sma_fast2 > self.sma_slow2:
                self.buy()

            # Sell signal
            elif self.position and self.crossover1 < 0 and self.sma_fast2 < self.sma_slow2:
                self.close()


class ARIMAStrategy(BaseStrategy):
    params = (
        ('order', (0, 0, 1)),  # ARIMA order
        ('lookback', 30),  # Lookback period for ARIMA model
    )

    def __init__(self):
        super().__init__()
        self.data_close = self.datas[0].close  # Use close prices
        self.order = None  # To keep track of pending orders

    def next(self):
        super().next()

        if len(self) <= self.p.lookback:
            return

        if self.order:  # Check if an order is pending, if so, do nothing
            # Get the past 'lookback' close prices
            return

        prices = np.array([self.data_close[i] for i in range(-self.p.lookback, 0)])
        transformed_prices = np.diff(prices)

        try:
            # Fit the ARIMA model
            model = ARIMA(transformed_prices, order=self.p.order)
            model_fit = model.fit()

            # Forecast the next price
            forecast = model_fit.forecast()[0]
        except Exception as e:
            print(f"Error fitting ARIMA model: {e}")
            return

        # Buy if the forecast is higher than zero
        if not self.position and forecast > 20:
            self.order = self.buy()
        # Sell if the forecast is less than zero
        elif self.position and forecast < -20:
            self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            self.order = None


class RLStrategy(BaseStrategy):
    params = (
        ('pred_path', '/home/fassty/Devel/school/diploma_thesis/code/data/rl_preds.csv'),
    )

    def __init__(self):
        super().__init__()
        self.agent = BacktestingAgent(self.params.pred_path)
        self.order = None  # To keep track of pending orders

    def next(self):
        super().next()
        current_ts = pd.Timestamp(self.datas[0].datetime.datetime(0))

        # Get action from the RL agent
        action = self.agent.get_action(current_ts)

        # Execute the action
        if action == Action.BUY and not self.position:
            self.order = self.buy()
        elif action == Action.SELL and self.position:
            self.order = self.close()

    def notify_order(self, order):
        # Reset the order attribute if the order is completed or canceled
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None
