import datetime
import logging
from typing import Dict, Any, Union, Optional, List

import requests
import json
import time
import hmac
import hashlib
import shortuuid
from urllib.parse import urlencode

from rl_trading.enums import OrderType, Side
from rl_trading.utils import RateLimited


class BinanceAPI:
    def __init__(self, api_key: str, api_secret: str):
        self.base_url = 'https://api.binance.com'
        self.api_key = api_key
        self.api_secret = api_secret

        self._instance_uuid = shortuuid.uuid()
        self._logger = logging.getLogger(f'{self.__class__.__name__}:uuid={self._instance_uuid}')

    def _get_signature(self, params: Dict[str, Any]):
        ordered_params = urlencode(sorted(params.items()))
        signature = hmac.new(self.api_secret.encode('utf-8'), ordered_params.encode('utf-8'), hashlib.sha256).hexdigest()
        return signature

    @RateLimited(max_calls_per_minute=60)
    def get_server_time(self):
        path = '/api/v3/time'
        url = self.base_url + path

        response = requests.get(url)

        if response.status_code == 200:
            return response.json()['serverTime']
        else:
            return None

    @RateLimited(max_calls_per_minute=60)
    def get_account_info(self):
        path = '/api/v3/account'
        url = self.base_url + path

        params = {}
        params['timestamp'] = int(time.time() * 1000)
        params['signature'] = self._get_signature(params)

        response = requests.get(url, params=params, headers={'X-MBX-APIKEY': self.api_key})

        if response.status_code == 200:
            return response.json()
        else:
            return None

    @RateLimited(max_calls_per_minute=60)
    def get_exchange_info(self, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        path = '/api/v3/exchangeInfo'
        url = self.base_url + path

        params = {}
        if symbol:
            params['symbol'] = symbol

        response = requests.get(url, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            return None

    @RateLimited(max_calls_per_minute=60)
    def get_latest_price(self, symbol: str) -> Optional[float]:
        path = '/api/v3/ticker/price'
        url = self.base_url + path
        params = {'symbol': symbol}

        response = requests.get(url, params=params)

        if response.status_code == 200:
            return float(response.json()['price'])
        else:
            return None

    @RateLimited(max_calls_per_minute=1000)
    def get_aggregated_trades(
            self, symbol: str, from_id: Optional[int] = None, start_time: Optional[int] = None,
            end_time: Optional[int] = None, limit: int = 500
    ) -> Optional[List[Dict[str, Any]]]:
        path = '/api/v3/aggTrades'
        url = self.base_url + path

        params = {'symbol': symbol, 'limit': limit}
        if from_id:
            params['fromId'] = from_id
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        response = requests.get(url, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            return None

    def get_latest_trades(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        now = datetime.datetime.utcnow()

        # Round down to the nearest whole minute by subtracting the current number of seconds
        minute_start_time = now - datetime.timedelta(seconds=now.second, microseconds=now.microsecond)

        # Convert back to a timestamp (in seconds)
        minute_start_ts = int(minute_start_time.timestamp())
        previous_minute_start_ts = int((minute_start_time - datetime.timedelta(minutes=1)).timestamp())
        current_timestamp_ms = minute_start_ts * 1000
        previous_timestamp_ms = previous_minute_start_ts * 1000

        latest_trades = self.get_aggregated_trades(symbol, start_time=previous_timestamp_ms, end_time=current_timestamp_ms, limit=1000)
        return latest_trades

    @RateLimited(max_calls_per_minute=50)
    def place_order(self, symbol: str, side: Union[str, Side], order_type: Union[str, OrderType], **order_kwargs):
        path = '/api/v3/order'
        url = self.base_url + path

        timestamp = str(int(time.time() * 1000))

        # Common mandatory params
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'timestamp': timestamp,
        }
        # Other mandatory params depending on the order type
        if order_type == OrderType.LIMIT:
            mandatory_keys = {'timeInForce', 'quantity', 'price'}
            if mandatory_keys <= set(order_kwargs.keys()):
                for k, v in order_kwargs.items():
                    params[k] = v
            else:
                raise ValueError(f'Missing mandatory keys: {mandatory_keys - set(order_kwargs.keys())} '
                                 f'for order type {order_type}')
        elif order_type == OrderType.MARKET:
            # Can specify either quantity or quoteOrderQty, but not both
            # quantity is always in the base currency and quoteOrderQty in the quote currency
            #    -> so for BTCUSDT quantity is the amount of BTC and quoteOrderQty is the amount of BTC equivalent
            #    to the amount of USDT I want to buy
            if 'quantity' in order_kwargs and 'quoteOrderQty' not in order_kwargs:
                params['quantity'] = order_kwargs['quantity']
            elif 'quoteOrderQty' in order_kwargs and 'quantity' not in order_kwargs:
                params['quoteOrderQty'] = order_kwargs['quoteOrderQty']
            else:
                raise ValueError(f'Exactly one of: quantity or quoteOrderQty must be specified '
                                 f'for order type {order_type}, got {order_kwargs.keys()}')
        elif order_type == OrderType.STOP_LOSS:
            raise NotImplementedError(f'Order type {order_type} not supported.')
        elif order_type == OrderType.TAKE_PROFIT:
            raise NotImplementedError(f'Order type {order_type} not supported.')
        else:
            raise NotImplementedError(f'Unknown order type: {order_type}')

        params = {k: params[k] for k in sorted(params)}
        params['signature'] = self._get_signature(params)

        headers = {
            'X-MBX-APIKEY': self.api_key
        }

        response = requests.post(url, data=params, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            print(f'Failed to place order: {response.text}')
            return None

    @RateLimited(max_calls_per_minute=50)
    def cancel_order(self, symbol: str, order_id: int):
        path = '/api/v3/order'
        url = self.base_url + path

        timestamp = str(int(time.time() * 1000))

        params = {
            'symbol': symbol,
            'orderId': order_id,
            'timestamp': timestamp,
        }

        params = {k: params[k] for k in sorted(params)}
        params['signature'] = self._get_signature(params)

        headers = {
            'X-MBX-APIKEY': self.api_key
        }

        response = requests.delete(url, data=params, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            print(f'Failed to cancel order: {response.text}')
            return None

    @RateLimited(max_calls_per_minute=50)
    def fetch_open_orders(self, symbol: str):
        path = '/api/v3/openOrders'
        url = self.base_url + path

        params = {
          'symbol': symbol,
        }

        params = {k: params[k] for k in sorted(params)}
        params['signature'] = self._get_signature(params)

        headers = {
            'X-MBX-APIKEY': self.api_key
        }

        response = requests.get(url, params=params, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            return None


class BinanceAPIMock(BinanceAPI):
    def get_account_info(self):
        ...

    def place_order(self, symbol: str, side: Union[str, Side], order_type: Union[str, OrderType], **order_kwargs):
        ...

    def cancel_order(self, symbol: str, order_id: int):
        ...

    def fetch_open_orders(self, symbol: str):
        ...
