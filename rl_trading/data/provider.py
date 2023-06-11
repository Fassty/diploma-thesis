import time
import datetime
from typing import List
from threading import Thread
import logging

import pandas as pd
import numpy as np

from rl_trading.singleton import Singleton
from rl_trading.data.api import BinanceAPI

logger = logging.getLogger('root')


def vwap(df):
    df['vwap'] = (df['price'] * df['volume']).cumsum() / df['volume'].cumsum()
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
        exchange_data: pd.DataFrame = pd.read_hdf(data_file_path)
        exchange_data.sort_index(inplace=True)
        exchange_data = exchange_data[~exchange_data.index.duplicated(keep='first')]
        exchange_data = exchange_data.reindex(np.arange(exchange_data.index[0], exchange_data.index[-1] + 1, 60))
        exchange_data['price'] = exchange_data['price'].ffill()
        exchange_data['amount'] = exchange_data['amount'].fillna(value=0)
        exchange_data.index = pd.to_datetime(exchange_data.index * 1e9)
        exchange_data.columns = ['volume', 'price']
        # Calculate VWAP
        exchange_data = exchange_data.groupby(exchange_data.index.date).apply(vwap).droplevel(0).ffill().bfill()
        self.market_data = exchange_data
        self.data_cache = {}

    def get_market_data(self, granularity: str, technical_indicators: List):
        market_data = {}
        required_granularities = set([gran for _, _, gran in technical_indicators] + [granularity])
        for gran in required_granularities:
            if gran in self.data_cache:
                market_data[gran] = self.data_cache[gran]
            elif gran != '1min':
                self.data_cache[gran] = self.market_data.groupby(pd.Grouper(freq=gran)).agg(self.AGG_MAPPING)
                market_data[gran] = self.data_cache[gran]
            else:
                market_data[gran] = self.market_data
        return market_data


class LiveDataProvider(AbstractProvider):
    def __init__(self, data_file_path: str, padding: int, symbol: str = 'BTCUSDT'):
        historical_data: pd.DataFrame = pd.read_hdf(data_file_path)
        historical_data.sort_index(inplace=True)
        historical_data = historical_data[~historical_data.index.duplicated(keep='first')]
        historical_data = historical_data.reindex(np.arange(historical_data.index[0], historical_data.index[-1] + 1, 60))
        historical_data['price'] = historical_data['price'].ffill()
        historical_data['amount'] = historical_data['amount'].fillna(value=0)
        historical_data.index = pd.to_datetime(historical_data.index * 1e9)
        historical_data.columns = ['volume', 'price']
        self.historical_data = historical_data
        self.calculate_vwap()
        self.symbol = symbol

        self.store_thread = None

    def calculate_vwap(self):
        self.historical_data = (
            self.historical_data
            .groupby(self.historical_data.index.date)
            .apply(vwap)
            .droplevel(0)
            .ffill()
            .bfill()
        )

    def get_last_available_timestamp(self):
        return int(self.historical_data.index[-1].timestamp())

    def get_current_minute_ts(self):
        current_ts = datetime.datetime.now()
        rounded = current_ts - datetime.timedelta(seconds=current_ts.second, microseconds=current_ts.microsecond)
        return int(rounded.timestamp()) * 1000

    def _update_data_with_new_trades(self, trades, timestamp: int):
        price = [float(t['p']) for t in trades][-1]
        volume = sum([float(t['q']) for t in trades])

        next_minute_df = pd.DataFrame(data={'price': price, 'volume': volume, 'vwap': np.nan},
                                      index=[pd.to_datetime(timestamp * 1e6)])
        self.historical_data = pd.concat([self.historical_data, next_minute_df])
        return price, volume

    def _store_data(self):
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
        current_timestamp_ms = self.get_current_minute_ts()

        #TODO: replace all prints here by logger.debug
        print(f'Updating data from {pd.to_datetime(last_timestamp_ms * 1e6)} to {pd.to_datetime(current_timestamp_ms * 1e6)}')

        counter = 0
        while next_download_timestamp_ms <= current_timestamp_ms:
            counter += 1
            print(
                f'Updating data from {pd.to_datetime(last_timestamp_ms * 1e6)} to {pd.to_datetime(next_download_timestamp_ms * 1e6)}')
            new_trades = api.get_aggregated_trades(self.symbol, start_time=last_timestamp_ms + 1, end_time=next_download_timestamp_ms, limit=1000)
            all_trades = new_trades

            retry_count = 0
            while len(new_trades) == 1000:
                retry_count += 1
                last_trade_id = new_trades[-1]['a']
                print(f'Last request failed to retrieve all trades, retry count: {retry_count}, fetching trades from id: {last_trade_id}')
                new_trades = api.get_aggregated_trades(self.symbol, from_id=last_trade_id + 1, limit=1000)
                new_trades = [trade for trade in new_trades if int(trade['T']) < next_download_timestamp_ms]
                all_trades = all_trades + new_trades
            else:
                print(f'Fetched {len(all_trades)}')

            self._update_data_with_new_trades(all_trades, next_download_timestamp_ms)
            last_timestamp_ms = next_download_timestamp_ms
            next_download_timestamp_ms += 60 * 1000

            if counter % 60 == 0:
                self.calculate_vwap()
                self._store_data()

        self.calculate_vwap()
        self._store_data()

    def get_market_data(self, granularity: str, technical_indicators: List):
        market_data = {}
        required_granularities = set([gran for _, _, gran in technical_indicators] + [granularity])
        for gran in required_granularities:
            if gran != '1min':
                market_data[gran] = self.historical_data.groupby(pd.Grouper(freq=gran)).agg(self.AGG_MAPPING)
            else:
                market_data[gran] = self.historical_data
        return market_data
