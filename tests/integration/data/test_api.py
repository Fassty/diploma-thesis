import os
from typing import Optional

import pytest
import configparser

from rl_trading.data.api import BinanceAPI
from rl_trading.constants import ROOT_DIR


@pytest.fixture(scope='session')
def api_config() -> configparser.SectionProxy:
    config = configparser.ConfigParser()
    config.read(os.path.join(ROOT_DIR, 'config.ini'))

    return config['test']


@pytest.fixture(scope='session')
def api_key(api_config: configparser.SectionProxy) -> str:
    return api_config['api_key']


@pytest.fixture(scope='session')
def api_secret(api_config: configparser.SectionProxy) -> str:
    return api_config['api_secret']


@pytest.fixture(scope='session')
def api(api_key: str, api_secret: str) -> BinanceAPI:
    return BinanceAPI(api_key, api_secret)


def test_get_server_time(api):
    server_time = api.get_server_time()

    assert server_time is not None
    assert isinstance(server_time, int)


def test_get_account_info(api):
    account_info = api.get_account_info()

    assert account_info is not None
    assert 'balances' in account_info


def test_get_exchange_info(api):
    exchange_info = api.get_exchange_info(symbol='BTCUSDT')

    assert exchange_info is not None


def test_get_latest_price(api):
    price = api.get_latest_price('BTCUSDT')

    assert price is not None
    assert isinstance(price, float)
    assert price > 0


def test_get_aggregated_trades(api):
    trades = api.get_aggregated_trades('BTCUSDT', limit=100)

    assert trades is not None
    assert isinstance(trades, list)
    assert len(trades) > 0


def test_get_latest_trades(api):
    trades = api.get_latest_trades('BTCUSDT')

    assert trades is not None
    assert isinstance(trades, list)


@pytest.mark.parametrize('order_type, params, expected_response', [
    ('LIMIT_BUY',
     {'symbol': 'BTCUSDT', 'side': 'BUY', 'order_type': 'LIMIT', 'timeInForce': 'GTC', 'quantity': '0.001', 'price': '10000'},
     {}),
    # ('MARKET_BUY',
    #  {'symbol': 'BTCUSDT', 'side': 'BUY', 'order_type': 'MARKET'},
    #  {}),
    # ('LIMIT_SELL',
    #  {'symbol': 'BTCUSDT', 'side': 'SELL', 'order_type': 'LIMIT'},
    #  {}),
    # ('MARKET_SELL',
    #  {'symbol': 'BTCUSDT', 'side': 'SELL', 'order_type': 'MARKET'},
    #  {}),
])
def test_place_order(api, order_type, params, expected_response):
    order_id: Optional[int] = None

    try:
        order_response = api.place_order(**params)
        order_id = order_response['orderId']

        assert order_response['status'] == 'NEW'
    finally:
        if order_id is not None:
            cancel_response = api.cancel_order(params['symbol'], order_id)
            assert cancel_response['status'] == 'CANCELED'



