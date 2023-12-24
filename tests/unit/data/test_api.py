import pytest
from unittest.mock import patch
from rl_trading.data.api import BinanceAPI
from rl_trading.enums import Side, OrderType, TimeInForce


@pytest.fixture
def binance_api():
    return BinanceAPI('test', 'test')


@pytest.mark.parametrize('symbol, expected_trades', [
    ('BTCUSD', [{'id': 1, 'price': '50000', 'qty': '1', 'symbol': 'BTCUSD'}]),
    ('ETHUSD', [{'id': 2, 'price': '3000', 'qty': '2', 'symbol': 'ETHUSD'}])
])
def test_get_latest_trades(binance_api, symbol, expected_trades):
    with patch('rl_trading.data.api.requests.get') as mock_get:
        mock_get.return_value.json.return_value = expected_trades
        mock_get.return_value.status_code = 200
        result = binance_api.get_latest_trades(symbol)
        mock_get.assert_called_once()
        assert result == expected_trades


@pytest.mark.parametrize('order_type, params, expected_response', [
    ('LIMIT_BUY',
     {'symbol': 'BTCUSDT', 'side': 'BUY', 'order_type': 'LIMIT', 'timeInForce': 'GTC', 'quantity': '1', 'price': '1000'},
     {'orderId': 100, 'status': 'NEW'}),
    ('MARKET_BUY',
     {'symbol': 'BTCUSDT', 'side': 'BUY', 'order_type': 'MARKET', 'timeInForce': 'GTC', 'quantity': '1', 'price': '1000'},
     {'orderId': 100, 'status': 'NEW'}),
])
def test_place_order_valid(binance_api, order_type, params, expected_response):
    with patch('rl_trading.data.api.requests.post') as mock_post:
        mock_post.return_value.json.return_value = expected_response
        mock_post.return_value.status_code = 200
        result = binance_api.place_order(**params)
        mock_post.assert_called_once()
        assert result == expected_response
