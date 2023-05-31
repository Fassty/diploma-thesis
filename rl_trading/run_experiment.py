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
parser.add_argument("--exp_name", default='SAC_network_size', type=str, help="Experiment name.")
parser.add_argument("--save_dir", default='/home/fassty/Devel/school/diploma_thesis/code/exp_results/continuous_env/SAC', type=str, help="Where to store the experiments.")
parser.add_argument("--algo", default='SAC', type=str, help='Which algorithm to use for the experiment.')
parser.add_argument("--iterations", default=100, type=int, help="Number of training iterations to run.")
parser.add_argument("--num_samples", default=1, type=int, help="Number of times to run each experiment.")
parser.add_argument("--n_days", default=None, type=int, help="Number of days to use for training.")


def main(args: argparse.Namespace):
    config = (
        SACConfig()
        .rollouts(num_rollout_workers=2, num_envs_per_worker=8)
        .resources(num_gpus=0.2, num_cpus_per_worker=0.5)
        .training(
            initial_alpha=0.1,
            q_model_config={
                'fcnet_hiddens': tune.grid_search([[256, 256, 256], [1024, 1024], [1024, 1024, 1024]])},
            policy_model_config={
                'fcnet_hiddens': tune.grid_search([[256, 256], [256, 256, 256], [1024, 1024], [1024, 1024, 1024]])}
        )
        .environment(env='StockExchangeEnv-v1',
                     env_config={
                         'state_config': {
                             'market_state': ['vwap'],
                             'technical_indicators': [
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
                         #'_idxs_range': np.linspace(288000, 2792281 - 1440 - 1, args.n_days, dtype=int).tolist()
                     })
        .reporting(min_sample_timesteps_per_iteration=2 * 8 * 1440)
        .evaluation(evaluation_interval=10, evaluation_duration=20)
    )

    tuner = tune.Tuner(
        args.algo,
        run_config=air.RunConfig(
            name=args.exp_name.format(args.n_days) if args.n_days is not None else args.exp_name,
            local_dir=args.save_dir,
            stop={'training_iteration': args.iterations},
            checkpoint_config=air.CheckpointConfig(
                checkpoint_at_end=True,
                checkpoint_frequency=50
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
