from gymnasium.wrappers import TimeLimit
from gymnasium.envs import register, make
from ray.tune.registry import register_env
from .env import StockExchangeEnv0, StockExchangeEnv1, SimulationConfig

register('StockExchangeEnv-v0', StockExchangeEnv0)
register('StockExchangeEnv-v1', StockExchangeEnv1)

register_env('StockExchangeEnv-v0', lambda config: StockExchangeEnv0(**config))
register_env('StockExchangeEnv-v1', lambda config: StockExchangeEnv1(**config))
