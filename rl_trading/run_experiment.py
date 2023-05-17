import argparse
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.apex_dqn import ApexDQNConfig
from ray.rllib.algorithms.r2d2 import R2D2Config

from rl_trading.data.indicators import *
from rl_trading.simulation.env import *


parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", default='R2D2_stored', type=str, help="Experiment name.")
parser.add_argument("--save_dir", default='../exp_results/discrete_env_21ind_normalized_fee1e-3', type=str, help="Where to store the experiments.")
parser.add_argument("--algo", default='R2D2', type=str, help='Which algorithm to use for the experiment.')
parser.add_argument("--iterations", default=1000, type=int, help="Number of training iterations to run.")
parser.add_argument("--num_samples", default=4, type=int, help="Number of times to run each experiment.")


def main(args: argparse.Namespace):
    config = (
        R2D2Config()
        .rollouts(num_rollout_workers=5, num_envs_per_worker=8)
        .training(model={'use_lstm': True,
                         'lstm_use_prev_action': True,
                         'lstm_use_prev_reward': True,
                         'max_seq_len': 80},
                  replay_buffer_config={'replay_burn_in': 40, 'capacity': 100_000},
                  zero_init_states=False,
                  target_network_update_freq=24_000,
                  train_batch_size=512
                  )
        .exploration(exploration_config={'epsilon_timesteps': 15_000_000})
        .resources(num_gpus=1)
        .environment(env='StockExchangeEnv-v0',
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
                         'exchange_config': {'maker_fee': 1e-3}

                     })
        .reporting(min_sample_timesteps_per_iteration=24000)
        .evaluation(evaluation_interval=25, evaluation_duration=40)
    )

    tuner = tune.Tuner(
        args.algo,
        run_config=air.RunConfig(
            name=args.exp_name,
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
