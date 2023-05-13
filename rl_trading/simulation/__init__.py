from gymnasium.wrappers import TimeLimit
from gymnasium.envs import register, make
from ray.tune.registry import register_env
from .env import StockExchangeEnv0, SimulationConfig

register('StockExchangeEnv-v0', StockExchangeEnv0)

register_env('StockExchangeEnv-v0', lambda config: StockExchangeEnv0(**config))
