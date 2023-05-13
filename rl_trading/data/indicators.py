import numpy as np
import pandas as pd
import talib
from typing import Callable, Dict
import numba


TechnicalIndicator = Callable
IndicatorParams = Dict[str, int]


def SMA(data: np.ndarray, timeperiod: int = 30):
    return [talib.SMA(data, timeperiod=timeperiod)[-1]]


def EMA(data: np.ndarray, timeperiod: int = 30):
    return [talib.EMA(data, timeperiod=timeperiod)[-1]]


def RSI(data: np.ndarray, timeperiod: int = 14):
    return [talib.RSI(data, timeperiod=timeperiod)[-1]]


def MACD(data: np.ndarray, fastperiod=12, slowperiod=26, signalperiod=9):
    macd, _, _ = talib.MACD(data, fastperiod, slowperiod, signalperiod)
    return [macd[-1]]


def MACD_DIFF(data: np.ndarray, fastperiod=12, slowperiod=26, signalperiod=9):
    macd, macd_signal, _ = talib.MACD(data, fastperiod, slowperiod, signalperiod)
    return [(macd - macd_signal)[-1]]


def BBANDS(data: np.ndarray, timeperiod: int = 5, nbdevup: int = 2, nbdevdn: int = 2):
    lower_bands, _, upper_bands = talib.BBANDS(data, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn)
    return [lower_bands[-1], upper_bands[-1]]


# TODO: Add Stochastic Oscillator - requires minutely OHLC that I need to get somewhere
indicator_padding: Dict[TechnicalIndicator, Callable[[Dict], int]] = {
    SMA.__name__: lambda config: config['timeperiod'],
    EMA.__name__: lambda config: config['timeperiod'],
    RSI.__name__: lambda config: config['timeperiod'],
    MACD.__name__: lambda config: (config['slowperiod'] - 1) + (config['signalperiod'] - 1),
    MACD_DIFF.__name__: lambda config: (config['slowperiod'] - 1) + (config['signalperiod'] - 1),
    BBANDS.__name__: lambda config: config['timeperiod'],
}
