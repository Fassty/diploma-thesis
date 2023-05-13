from typing import List

import pandas as pd
import numpy as np

from rl_trading.singleton import Singleton


def vwap(df):
    df['vwap'] = (df['price'] * df['volume']).cumsum() / df['volume'].cumsum()
    return df


def typical_price(price: pd.Series):
    return (price.max() + price.min() + price[-1]) / 3


# TODO: If in future I'll be using more markets refactor to AttrSingleton
class MarketDataProvider(metaclass=Singleton):
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
