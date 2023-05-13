import argparse
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig

from rl_trading.data.indicators import *
from rl_trading.simulation.env import *


parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", default='DQN_5step', type=str, help="Experiment name.")
parser.add_argument("--save_dir", default='../exp_results/discrete_env_21ind', type=str, help="Where to store the experiments.")
parser.add_argument("--algo", default='DQN', type=str, help='Which algorithm to use for the experiment.')
parser.add_argument("--iterations", default=1000, type=int, help="Number of training iterations to run.")
parser.add_argument("--num_samples", default=1, type=int, help="Number of times to run each experiment.")


def main(args: argparse.Namespace):
    config = (
        DQNConfig()
        .rollouts(num_rollout_workers=1, num_envs_per_worker=8)
        .training(n_step=5)
        .resources(num_gpus=1)
        .environment(env='StockExchangeEnv-v0', env_config={'state_config': {
            'market_state': ['price', 'vwap'],
            'technical_indicators': [
                (EMA, dict(timeperiod=5), '1min'),
                (EMA, dict(timeperiod=13), '1min'),
                (RSI, dict(timeperiod=7), '1min'),
                (BBANDS, dict(timeperiod=10), '1min'),
                (EMA, dict(timeperiod=20), '1h'),
                (EMA, dict(timeperiod=50), '1h'),
                (RSI, dict(timeperiod=14), '1h'),
                (BBANDS, dict(timeperiod=20), '1h'),
                (MACD_DIFF, dict(fastperiod=12, slowperiod=26, signalperiod=9), '1h'),
                (EMA, dict(timeperiod=50), '1d'),
                (EMA, dict(timeperiod=200), '1d'),
                (RSI, dict(timeperiod=14), '1d'),
                (BBANDS, dict(timeperiod=20), '1d'),
                (MACD_DIFF, dict(fastperiod=12, slowperiod=26, signalperiod=9), '1d'),
            ],
        }})
        .reporting(min_sample_timesteps_per_iteration=16 * 1440)
        .evaluation(evaluation_interval=10, evaluation_duration=20)
    )

    tuner = tune.Tuner(
        args.algo,
        run_config=air.RunConfig(
            name=args.exp_name,
            local_dir=args.save_dir,
            stop={'training_iteration': args.iterations},
            checkpoint_config=air.CheckpointConfig(
                checkpoint_at_end=True
            )
        ),
        tune_config=tune.TuneConfig(
            num_samples=args.num_samples
        ),
        param_space=config
    )
    tuner.fit()


if __name__ == '__main__':
    main(parser.parse_args([] if "__file__" not in globals() else None))
