import time
import datetime
from typing import List
from threading import Thread
import logging

import pandas as pd
import numpy as np

from rl_trading.singleton import Singleton
from rl_trading.data.api import BinanceAPI
from rl_trading.utils import get_current_minute_ts

logger = logging.getLogger(__name__)


def vwap(df):
    df['vwap'] = (df['price'] * df['volume']).cumsum() / df['volume'].cumsum()
    return df


def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby(df.index.date).apply(vwap).droplevel(0).ffill().bfill()
    return df


def typical_price(price: pd.Series):
    return (price.max() + price.min() + price[-1]) / 3


class AbstractProvider:
    AGG_MAPPING = {
        'price': 'last', 'volume': 'sum', 'vwap': 'last'
    }

    def get_market_data(self, granularity: str, technical_indicators: List):
        raise NotImplementedError()


# TODO: If in future I'll be using more markets refactor to AttrSingleton
class MarketDataProvider(AbstractProvider, metaclass=Singleton):
    AGG_MAPPING = {
        'price': 'last', 'volume': 'sum', 'vwap': 'last'
    }

    def __init__(self, data_file_path: str):
        self.exchange_data: pd.DataFrame = pd.read_hdf(data_file_path)
        self.exchange_data.sort_index(inplace=True)
        self.exchange_data = self.exchange_data[~self.exchange_data.index.duplicated(keep='first')]
        self.exchange_data = self.exchange_data.reindex(np.arange(self.exchange_data.index[0], self.exchange_data.index[-1] + 1, 60))
        self.exchange_data['price'] = self.exchange_data['price'].ffill()
        self.exchange_data['amount'] = self.exchange_data['amount'].fillna(value=0)
        self.exchange_data.index = pd.to_datetime(self.exchange_data.index * 1e9)
        self.exchange_data.columns = ['volume', 'price']
        # Calculate VWAP
        self.exchange_data = calculate_vwap(self.exchange_data)
        self.data_cache = {}

    def update_data(self, price: float, volume: float):
        last_timestamp = self.exchange_data.index[-1]
        next_timestamp = last_timestamp + pd.Timedelta(minutes=1)

        next_minute_df = pd.DataFrame(data={'price': price, 'volume': volume, 'vwap': np.nan},
                                      index=[next_timestamp])
        self.exchange_data = pd.concat([self.exchange_data, next_minute_df])
        daily_mask = self.exchange_data.index.date == next_timestamp.date()
        self.exchange_data[daily_mask] = calculate_vwap(self.exchange_data[daily_mask])

        if last_timestamp.day != next_timestamp.day:
            if '1d' in self.data_cache:
                next_day_df = self.exchange_data[
                              next_timestamp - pd.Timedelta(days=1) + pd.Timedelta(minutes=1):next_timestamp]
                price = next_day_df['price'].iloc[-1]
                volume = next_day_df['volume'].sum()
                vwap = next_day_df['vwap'].iloc[-1]
                self.data_cache['1d'].loc[next_timestamp] = {'price': price, 'volume': volume, 'vwap': vwap}
        if last_timestamp.hour != next_timestamp.hour:
            if '1h' in self.data_cache:
                next_hour_df = self.exchange_data[
                              next_timestamp - pd.Timedelta(hours=1) + pd.Timedelta(minutes=1):next_timestamp]
                price = next_hour_df['price'].iloc[-1]
                volume = next_hour_df['volume'].sum()
                vwap = next_hour_df['vwap'].iloc[-1]
                self.data_cache['1h'].loc[next_timestamp] = {'price': price, 'volume': volume, 'vwap': vwap}
        self.data_cache['1min'] = self.exchange_data

    def get_market_data(self, granularity: str, technical_indicators: List):
        market_data = {}
        required_granularities = set([gran for _, _, gran in technical_indicators] + [granularity])
        for gran in required_granularities:
            if gran in self.data_cache:
                market_data[gran] = self.data_cache[gran]
            elif gran != '1min':
                self.data_cache[gran] = self.exchange_data.groupby(pd.Grouper(freq=gran, closed='right', label='right')).agg(self.AGG_MAPPING)
                market_data[gran] = self.data_cache[gran]
            else:
                market_data[gran] = self.exchange_data
        return market_data


class LiveDataProvider(AbstractProvider):
    def __init__(self, data_file_path: str, padding: int, symbol: str = 'BTCUSDT'):
        self.historical_data: pd.DataFrame = pd.read_hdf(data_file_path)
        if 'vwap' not in self.historical_data.columns:
            self.preprocess_data()
        self.symbol = symbol

        self._logger = logging.getLogger(f'{self.__class__.__name__}')

        self.store_thread = None

    def preprocess_data(self):
        self._logger.info(f'Preprocessing historical data')
        self.historical_data.sort_index(inplace=True)
        self.historical_data = self.historical_data[~self.historical_data.index.duplicated(keep='first')]
        self.historical_data = self.historical_data.reindex(
            np.arange(self.historical_data.index[0], self.historical_data.index[-1] + 1, 60))
        self.historical_data['price'] = self.historical_data['price'].ffill()
        self.historical_data['amount'] = self.historical_data['amount'].fillna(value=0)
        self.historical_data.index = self.historical_data.to_datetime(self.historical_data.index * 1e9)
        self.historical_data.columns = ['volume', 'price']
        self.historical_data = calculate_vwap(self.historical_data)

    def get_last_available_timestamp(self):
        return int(self.historical_data.index[-1].timestamp())

    def _update_data_with_new_trades(self, trades, timestamp: int):
        price = [float(t['p']) for t in trades][-1]
        volume = sum([float(t['q']) for t in trades])

        next_minute_df = pd.DataFrame(data={'price': price, 'volume': volume, 'vwap': np.nan},
                                      index=[pd.to_datetime(timestamp * 1e6)])
        self.historical_data = pd.concat([self.historical_data, next_minute_df])
        return price, volume

    def _store_data(self):
        self._logger.info(f'Storing new data')
        self.store_thread = Thread(
            target=self.historical_data.to_hdf,
            args=('/home/fassty/Devel/school/diploma_thesis/code/data/binance_BTC_USDT_vwap.h5',
                  'data')
        )
        self.store_thread.start()

    def update_data(self, api: BinanceAPI):
        if self.store_thread is not None:
            self.store_thread.join()
            self.store_thread = None

        last_timestamp = self.get_last_available_timestamp()
        last_timestamp_ms = last_timestamp * 1000
        next_download_timestamp_ms = last_timestamp_ms + 60 * 1000
        current_timestamp_ms = get_current_minute_ts()

        self._logger.info(
            f'Updating data from {pd.to_datetime(last_timestamp_ms * 1e6)} '
            f'to {pd.to_datetime(current_timestamp_ms * 1e6)}'
        )

        counter = 0
        while next_download_timestamp_ms <= current_timestamp_ms:
            counter += 1
            self._logger.info(
                f'Updating data from {pd.to_datetime(last_timestamp_ms * 1e6)} '
                f'to {pd.to_datetime(next_download_timestamp_ms * 1e6)}'
            )
            new_trades = api.get_aggregated_trades(self.symbol, start_time=last_timestamp_ms + 1, end_time=next_download_timestamp_ms, limit=1000)
            all_trades = new_trades

            retry_count = 0
            while len(new_trades) == 1000:
                retry_count += 1
                last_trade_id = new_trades[-1]['a']
                self._logger.debug(
                    f'Last request failed to retrieve all trades, retry count: {retry_count}, '
                    f'fetching trades from id: {last_trade_id}'
                )
                new_trades = api.get_aggregated_trades(self.symbol, from_id=last_trade_id + 1, limit=1000)
                new_trades = [trade for trade in new_trades if int(trade['T']) < next_download_timestamp_ms]
                all_trades = all_trades + new_trades
            else:
                self._logger.debug(f'Fetched {len(all_trades)} trades')

            self._update_data_with_new_trades(all_trades, next_download_timestamp_ms)
            last_timestamp_ms = next_download_timestamp_ms
            next_download_timestamp_ms += 60 * 1000

            if counter % 60 == 0:
                self.historical_data = calculate_vwap(self.historical_data)
                self._store_data()

        if counter > 0:
            self.historical_data = calculate_vwap(self.historical_data)
            self._store_data()

    def get_market_data(self, granularity: str, technical_indicators: List):
        market_data = {}
        required_granularities = set([gran for _, _, gran in technical_indicators] + [granularity])
        for gran in required_granularities:
            if gran != '1min':
                market_data[gran] = self.historical_data.groupby(pd.Grouper(freq=gran, closed='right', label='right')).agg(self.AGG_MAPPING)
            else:
                market_data[gran] = self.historical_data
        return market_data
