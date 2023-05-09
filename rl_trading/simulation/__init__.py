from gymnasium.wrappers import TimeLimit
from gymnasium.envs import register, make
from ray.tune.registry import register_env
from .env import StockExchangeEnv0, StockExchangeEnv1, SimulationConfig

register('StockExchangeEnv-v0', StockExchangeEnv0)
register('StockExchangeEnv-v1', StockExchangeEnv1)

register_env('StockExchangeEnv-v0', lambda config: TimeLimit(
    StockExchangeEnv0(**config),
    max_episode_steps=SimulationConfig.max_steps if 'sim_config' not in config else config.sim_config.max_steps
))

register_env('StockExchangeEnv-v1', lambda config: StockExchangeEnv1(**config))
