import importlib
import os

import rl_trading.simulation
from rl_trading.utils import custom_trial_name_creator, str_to_obj

os.environ['RAY_DEDUP_LOGS'] = '0'
import argparse
import ray
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.apex_dqn import ApexDQNConfig, ApexDQN
from ray.rllib.algorithms.r2d2 import R2D2Config, R2D2
from ray.rllib.algorithms.a3c import A3CConfig
from ray.rllib.algorithms.sac import SACConfig, SAC
from ray.rllib.algorithms.appo import APPOConfig

from rl_trading.data.indicators import *
from rl_trading.simulation.env import *


parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", default='/home/fassty/Devel/school/diploma_thesis/code/exp_results/',
                    type=str, help="Where to store the experiments.")
parser.add_argument("--exp_group", default='short_episodes_future', type=str, help="Experiment group.")
parser.add_argument("--exp_name", default='discrete_env_15ind_normalized', type=str, help="Experiment name.")
parser.add_argument("--algo", default='ApexDQN', type=str_to_obj, help='Which algorithm to use for the experiment.')
parser.add_argument("--iterations", default=200, type=int, help="Number of training iterations to run.")
parser.add_argument("--episode_length", default=1000, type=int, help="Length of the episode.")
parser.add_argument("--num_workers", default=1, type=int, help="Number of parallel workers.")
parser.add_argument("--num_envs", default=50, type=int, help="Number of vector environments per worker.")
parser.add_argument("--num_samples", default=1, type=int, help="Number of times to run each experiment.")
# Options for running benchmark experiments
parser.add_argument("--n_days", default=None, type=int, help="Number of days to use for training.")
parser.add_argument('--seed', default=None, type=int, help="Random seed to be used for sampling the days.")

ray.get_gpu_ids()


def r2d2_config(args) -> R2D2Config:
    config = (
        R2D2Config()
        .rollouts(num_rollout_workers=args.num_workers, num_envs_per_worker=args.num_envs)
        .resources(num_gpus=1, num_cpus_per_worker=1)
        .training(
            model={'use_lstm': True, 'max_seq_len': 80, 'lstm_use_prev_action': True, 'lstm_use_prev_reward': True},
            replay_buffer_config={'replay_burn_in': 40},
            zero_init_states=True)
        .exploration(exploration_config={
            'initial_epsilon': 1.0,
            'final_epsilon': 0.01,
            'epsilon_timesteps': 7_500_000})
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
                         'exchange_config': {'maker_fee': 1e-4},
                         'eval_period': 1440  # * 24 * 60
                     })
        .reporting(min_sample_timesteps_per_iteration=args.num_workers * args.num_envs * args.episode_length,
                   metrics_num_episodes_for_smoothing=args.num_workers * args.num_envs,
                   min_time_s_per_iteration=None)
        .evaluation(evaluation_interval=1,
                    evaluation_duration=2,
                    evaluation_config={
                        'explore': False,
                        'num_envs_per_worker': 1,
                        'num_rollout_workers': 1,
                        'env_config': {
                            'stage': 'eval'
                        }},
                    evaluation_num_workers=0,
                    evaluation_sample_timeout_s=3600)
    )

    return config


def sac_config(args) -> SACConfig:
    config = (
        SACConfig()
        .rollouts(num_rollout_workers=args.num_workers, num_envs_per_worker=args.num_envs)
        .resources(num_gpus=1, num_cpus_per_worker=1)
        .training(initial_alpha=0.2,
                  replay_buffer_config={'prioritized_replay': True},
                  gamma=0.999,
                  num_steps_sampled_before_learning_starts=10000,
                  target_network_update_freq=0,
                  optimization_config={
                      "actor_learning_rate": 1e-4,
                      "critic_learning_rate": 5e-4,
                      "entropy_learning_rate": 3e-4,
                  })
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
                         'exchange_config': {'maker_fee': 1e-4},
                         'eval_period': 1440  # * 24 * 60
                     })
        .reporting(min_sample_timesteps_per_iteration=args.num_workers * args.num_envs * args.episode_length,
                   metrics_num_episodes_for_smoothing=args.num_workers * args.num_envs,
                   min_time_s_per_iteration=None)
        .evaluation(evaluation_interval=1,
                    evaluation_duration=2,
                    evaluation_config={
                        'explore': False,
                        'num_envs_per_worker': 1,
                        'num_rollout_workers': 1,
                        'env_config': {
                            'stage': 'eval'
                        }},
                    evaluation_num_workers=1,
                    evaluation_sample_timeout_s=3600)
    )

    return config


def ppo_config(args) -> PPOConfig:
    config = (
        PPOConfig()
        .rollouts(num_rollout_workers=args.num_workers, num_envs_per_worker=args.num_envs)
        .resources(num_gpus=0.25, num_cpus_per_worker=0.5)
        .training(train_batch_size=10000,
                  sgd_minibatch_size=1000,
                  lambda_=0.95,
                  clip_param=0.2,
                  shuffle_sequences=False,
                  vf_clip_param=20)
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
                         'eval_period': 1440  # * 24 * 60
                     })
        .reporting(min_sample_timesteps_per_iteration=args.num_workers * args.num_envs * args.episode_length,
                   metrics_num_episodes_for_smoothing=args.num_workers * args.num_envs,
                   min_time_s_per_iteration=None)
        .evaluation(evaluation_interval=1,
                    evaluation_duration=2,
                    evaluation_config={
                        'explore': True,
                        'num_envs_per_worker': 1,
                        'num_rollout_workers': 1,
                        'env_config': {
                            'stage': 'eval'
                        }},
                    evaluation_num_workers=1,
                    evaluation_sample_timeout_s=3600)
    )

    return config


def dqn_config(args) -> ApexDQNConfig:
    config = (
        ApexDQNConfig()
        .rollouts(num_rollout_workers=args.num_workers, num_envs_per_worker=args.num_envs)
        .resources(num_gpus=0.25, num_cpus_per_worker=0.5)
        .training(n_step=1)
        .exploration(exploration_config={
            'initial_epsilon': 1.0,
            'final_epsilon': 0.005,
            'epsilon_timesteps': 7_500_000})
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
                         'eval_period': 1440  # * 24 * 60
                     })
        .reporting(min_sample_timesteps_per_iteration=args.num_workers * args.num_envs * args.episode_length,
                   metrics_num_episodes_for_smoothing=args.num_workers * args.num_envs,
                   min_time_s_per_iteration=None)
        .evaluation(evaluation_interval=1,
                    evaluation_duration=2,
                    evaluation_config={
                        'explore': True,
                        'num_envs_per_worker': 1,
                        'num_rollout_workers': 1,
                        'env_config': {
                            'stage': 'eval'
                        }},
                    evaluation_num_workers=1,
                    evaluation_sample_timeout_s=3600)
    )

    return config


def main(args: argparse.Namespace):
    save_dir = os.path.join(args.save_dir, args.exp_group)

    if issubclass(args.algo, R2D2):
        config = r2d2_config(args)
    elif issubclass(args.algo, SAC):
        config = sac_config(args)
    elif issubclass(args.algo, PPO):
        config = ppo_config(args)
    elif issubclass(args.algo, ApexDQN):
        config = dqn_config(args)

    tuner = tune.Tuner(
        args.algo,
        run_config=air.RunConfig(
            name=args.exp_name,
            local_dir=save_dir,
            stop={'training_iteration': args.iterations},
            checkpoint_config=air.CheckpointConfig(
                checkpoint_at_end=True,
                checkpoint_frequency=1,
                num_to_keep=1,
                checkpoint_score_attribute='episode_reward_mean',
                checkpoint_score_order='max'
            ),
            verbose=1
        ),
        tune_config=tune.TuneConfig(
            num_samples=args.num_samples,
            trial_name_creator=custom_trial_name_creator
        ),
        param_space=config
    )
    tuner.fit()


if __name__ == '__main__':
    main(parser.parse_args([] if "__file__" not in globals() else None))
