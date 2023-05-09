import pandas as pd
import numpy as np

from rl_trading.singleton import Singleton


# TODO: If in future I'll be using more markets refactor to AttrSingleton
class MarketDataProvider(metaclass=Singleton):
    def __init__(self, data_file_path: str):
        exchange_data: pd.DataFrame = pd.read_hdf(data_file_path)
        exchange_data.sort_index(inplace=True)
        exchange_data = exchange_data[~exchange_data.index.duplicated(keep='first')]
        exchange_data = exchange_data.reindex(np.arange(exchange_data.index[0], exchange_data.index[-1] + 1, 60))
        exchange_data['price'] = exchange_data['price'].ffill()
        exchange_data['amount'] = exchange_data['amount'].fillna(value=0)
        exchange_data.index = pd.to_datetime(exchange_data.index * 1e9)
        self.market_data = exchange_data

    def get_market_data(self):
        return self.market_data

    def get_price_data(self):
        return self.market_data['price']

    def get_volume_data(self):
        return self.market_data['amount']
