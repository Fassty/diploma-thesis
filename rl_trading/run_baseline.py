import os

from rl_trading.utils import custom_trial_name_creator, str_to_obj

os.environ['RAY_DEDUP_LOGS'] = '0'
import argparse
import ray
from ray import tune, air

from rl_trading.data.indicators import *
from rl_trading.simulation.env import *


parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", default='/home/fassty/Devel/school/diploma_thesis/code/exp_results/',
                    type=str, help="Where to store the experiments.")
parser.add_argument("--exp_group", default='production', type=str, help="Experiment group.")
parser.add_argument("--exp_name", default='discrete_env_15ind_normalized', type=str, help="Experiment name.")
parser.add_argument("--algo", default='SAC', type=str_to_obj, help='Which algorithm to use for the experiment.')
parser.add_argument("--iterations", default=100, type=int, help="Number of training iterations to run.")
parser.add_argument("--episode_length", default=1000, type=int, help="Length of the episode.")
parser.add_argument("--num_workers", default=4, type=int, help="Number of parallel workers.")
parser.add_argument("--num_envs", default=25, type=int, help="Number of vector environments per worker.")
parser.add_argument("--num_samples", default=1, type=int, help="Number of times to run each experiment.")
# Options for running benchmark experiments
parser.add_argument("--n_days", default=None, type=int, help="Number of days to use for training.")
parser.add_argument('--seed', default=None, type=int, help="Random seed to be used for sampling the days.")


def main(args: argparse.Namespace):
    save_dir = os.path.join(args.save_dir, args.exp_group)

    config = (
        args.algo.get_default_config()
        .rollouts(num_rollout_workers=args.num_workers, num_envs_per_worker=args.num_envs, batch_mode='complete_episodes')
        .resources(num_gpus=0.2, num_cpus_per_worker=1)
        .training(initial_alpha=tune.grid_search([0.1, 0.2, 0.5, 1.0]),
                  replay_buffer_config={'prioritized_replay': tune.grid_search([True, False])},
                  gamma=0.999,
                  train_batch_size=4000,
                  num_steps_sampled_before_learning_starts=10000,
                  target_network_update_freq=tune.grid_search([0, 1000, 5000]),)
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
                   metrics_num_episodes_for_smoothing=args.num_workers * args.num_envs,
                   min_time_s_per_iteration=None)
        .evaluation(evaluation_interval=1, evaluation_duration=10, evaluation_config={'explore': False})
    )

    tuner = tune.Tuner(
        args.algo,
        run_config=air.RunConfig(
            name=args.exp_name,
            local_dir=save_dir,
            stop={'training_iteration': args.iterations},
            checkpoint_config=air.CheckpointConfig(
                checkpoint_at_end=True,
                checkpoint_frequency=1000
            )
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
