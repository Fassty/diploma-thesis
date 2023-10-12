import os
os.environ['RAY_DEDUP_LOGS'] = '0'
import argparse
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.apex_dqn import ApexDQNConfig
from ray.rllib.algorithms.r2d2 import R2D2Config
from ray.rllib.algorithms.a3c import A3CConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.appo import APPOConfig

from rl_trading.data.indicators import *
from rl_trading.simulation.env import *


parser = argparse.ArgumentParser()
parser.add_argument("--exp_group", default='normalized_reward', type=str, help="Experiment group.")
parser.add_argument("--exp_name", default='R2D2', type=str, help="Experiment name.")
parser.add_argument("--run_name", default='R2D2_10M', type=str, help="Run name.")
parser.add_argument("--save_dir", default='/home/fassty/Devel/school/diploma_thesis/code/exp_results/',
                    type=str, help="Where to store the experiments.")
parser.add_argument("--algo", default='R2D2', type=str, help='Which algorithm to use for the experiment.')
parser.add_argument("--iterations", default=100, type=int, help="Number of training iterations to run.")
parser.add_argument("--episode_length", default=1000, type=int, help="Length of the episode.")
parser.add_argument("--num_workers", default=5, type=int, help="Number of parallel workers.")
parser.add_argument("--num_envs", default=20, type=int, help="Number of vector environments per worker.")
parser.add_argument("--num_samples", default=5, type=int, help="Number of times to run each experiment.")
# Options for running benchmark experiments
parser.add_argument("--n_days", default=None, type=int, help="Number of days to use for training.")
parser.add_argument('--seed', default=None, type=int, help="Random seed to be used for sampling the days.")


def main(args: argparse.Namespace):
    config = (
        R2D2Config()
        .rollouts(num_rollout_workers=args.num_workers, num_envs_per_worker=args.num_envs)
        .resources(num_gpus=1, num_cpus_per_worker=1)
        .training(model={'use_lstm': True, 'max_seq_len': 80, 'lstm_use_prev_action': True, 'lstm_use_prev_reward': True},
                  replay_buffer_config={'replay_burn_in': 40},
                  zero_init_states=True)
        .environment(env='StockExchangeEnv-v0',
                     env_config={
                         'sim_config': {'max_steps': args.episode_length},
                         'state_config': {
                             'market_state': ['vwap'],
                             'technical_indicators': [
                                (RPC, {}, '1min'),
                                (EMA, dict(timeperiod=5, normalize=True), '1min'),
                                (EMA, dict(timeperiod=13, normalize=True), '1min'),
                                (RSI, dict(timeperiod=7, normalize=True), '1min'),
                                (BBANDS, dict(timeperiod=10), '1min'),
                                (EMA, dict(timeperiod=20, normalize=True), '1h'),
                                (EMA, dict(timeperiod=50, normalize=True), '1h'),
                                (RSI, dict(timeperiod=14, normalize=True), '1h'),
                                (BBANDS, dict(timeperiod=20), '1h'),
                                (MACD_DIFF, dict(fastperiod=12, slowperiod=26, signalperiod=9, normalize=True), '1h'),
                                (EMA, dict(timeperiod=50, normalize=True), '1d'),
                                (EMA, dict(timeperiod=200, normalize=True), '1d'),
                                (RSI, dict(timeperiod=14, normalize=True), '1d'),
                                (BBANDS, dict(timeperiod=20), '1d'),
                                (MACD_DIFF, dict(fastperiod=12, slowperiod=26, signalperiod=9, normalize=True), '1d'),
                             ]},
                         'exchange_config': {'maker_fee': 1e-3},
                     })
        .reporting(min_sample_timesteps_per_iteration=args.num_workers * args.num_envs * args.episode_length,
                   metrics_num_episodes_for_smoothing=args.num_workers * args.num_envs * args.episode_length)
        .evaluation(evaluation_interval=1, evaluation_duration=10, evaluation_config={'explore': False})
    )

    tuner = tune.Tuner(
        args.algo,
        run_config=air.RunConfig(
            name=args.exp_name.format(args.seed) if args.seed is not None else args.exp_name,
            local_dir=args.save_dir,
            stop={'training_iteration': args.iterations},
            checkpoint_config=air.CheckpointConfig(
                checkpoint_at_end=True,
                checkpoint_frequency=1000
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
