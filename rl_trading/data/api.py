import datetime
import requests
import json
import time
import hmac
import hashlib
from urllib.parse import urlencode

from rl_trading.utils import RateLimited


class BinanceAPI:
    def __init__(self, api_key, api_secret):
        self.base_url = 'https://api.binance.com'
        self.api_key = api_key
        self.api_secret = api_secret

    def _get_signature(self, params):
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
    def get_latest_price(self, symbol):
        path = '/api/v3/ticker/price'
        url = self.base_url + path
        params = {'symbol': symbol}

        response = requests.get(url, params=params)

        if response.status_code == 200:
            return response.json()['price']
        else:
            return None

    @RateLimited(max_calls_per_minute=1000)
    def get_aggregated_trades(self, symbol, from_id=None, start_time=None, end_time=None, limit=500):
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

    def get_latest_trades(self, symbol):
        now = datetime.datetime.utcnow()

        # Round down to the nearest whole minute by subtracting the current number of seconds
        rounded = now - datetime.timedelta(seconds=now.second, microseconds=now.microsecond)

        # Convert back to a timestamp (in seconds)
        current_timestamp = int(rounded.timestamp())
        previous_timestamp = int((rounded - datetime.timedelta(minutes=1)).timestamp())
        current_timestamp_ms = current_timestamp * 1000
        previous_timestamp_ms = previous_timestamp * 1000

        latest_trades = self.get_aggregated_trades(symbol, start_time=previous_timestamp_ms, end_time=current_timestamp_ms, limit=1000)
        return latest_trades

    @RateLimited(max_calls_per_minute=50)
    def create_order(self, symbol, side, type, quantity):
        path = '/api/v3/order'
        url = self.base_url + path

        timestamp = str(int(time.time() * 1000))

        query_string = f'symbol={symbol}&side={side}&type={type}&quantity={quantity}&timestamp={timestamp}'
        signature = hmac.new(bytes(self.api_secret, 'latin-1'), msg=bytes(query_string, 'latin-1'), digestmod=hashlib.sha256).hexdigest()

        headers = {
            'X-MBX-APIKEY': self.api_key
        }

        params = {
            'symbol': symbol,
            'side': side,
            'type': type,
            'quantity': quantity,
            'timestamp': timestamp,
            'signature': signature
        }

        response = requests.post(url, headers=headers, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            return None
