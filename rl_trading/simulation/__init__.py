from ray.tune.registry import register_env
from .env import StockExchangeEnv

register_env('StockExchangeEnv-v0', lambda config: StockExchangeEnv(**config))
