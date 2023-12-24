import logging
import sys
import datetime

import pandas as pd
import time
from functools import wraps
from ray.rllib.utils.checkpoints import get_checkpoint_info
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.algorithms.dqn import DQNConfig, DQN
from ray.rllib.algorithms.apex_dqn import ApexDQNConfig, ApexDQN
from ray.rllib.algorithms.r2d2 import R2D2Config, R2D2
from ray.rllib.algorithms.a3c import A3CConfig, A3C
from ray.rllib.algorithms.sac import SACConfig, SAC
from ray.rllib.algorithms.appo import APPOConfig, APPO
from ray.rllib.algorithms.algorithm import Algorithm


def granularity_convert(source_granularity: str, target_granularity: str = '1min'):
    return pd.to_timedelta(source_granularity).total_seconds() // pd.to_timedelta(target_granularity).total_seconds()


def load_model(model_path):
    checkpoint_info = get_checkpoint_info(model_path)
    checkpoint_state = Algorithm._checkpoint_info_to_algorithm_state(checkpoint_info)
    if issubclass(checkpoint_state['algorithm_class'], R2D2):
        checkpoint_state['config']['replay_buffer_config']['replay_sequence_length'] = -1
    return Algorithm.from_state(checkpoint_state)


def get_current_minute_ts():
    current_ts = datetime.datetime.now()
    rounded = current_ts - datetime.timedelta(seconds=current_ts.second, microseconds=current_ts.microsecond)
    return int(rounded.timestamp()) * 1000


def custom_trial_name_creator(trial: 'Trial'):
    return f'{trial.trainable_name}'


def str_to_obj(string_name):
    """Convert string to an object of the same name, if it exists."""
    return globals().get(string_name)


class RateLimited:
    def __init__(self, max_calls_per_minute):
        self.max_calls_per_minute = max_calls_per_minute
        self.calls = []

        self._logger = logging.getLogger(self.__class__.__name__)

    def _cleanup(self):
        """Remove timestamps older than 1 minute."""
        one_minute_ago = time.time() - 60
        while self.calls and self.calls[0] < one_minute_ago:
            self.calls.pop(0)

    def _wait_if_needed(self):
        """If rate limit has been reached, wait until we can make a new request."""
        self._cleanup()
        if len(self.calls) >= self.max_calls_per_minute:
            time_to_wait = 60 - (time.time() - self.calls[0])
            self._logger.info(f'Rate limit reached, waiting {time_to_wait} seconds')
            if time_to_wait > 0:
                time.sleep(time_to_wait)
            self.calls.pop(0)  # remove this timestamp, we've waited long enough

    def __call__(self, function):
        """This makes the class instance callable."""
        @wraps(function)
        def wrapped(*args, **kwargs):
            """This function is called instead of the original function."""
            self._wait_if_needed()
            self.calls.append(time.time())
            return function(*args, **kwargs)
        return wrapped


def setup_logging(log_level: int = logging.INFO, log_to_stderr: bool = False):
    # Create a handler that writes log messages to a file
    file_handler = logging.FileHandler('my_app.log')
    file_handler.setLevel(log_level)

    # Create a formatter that includes the module name in the log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    handlers = [file_handler]

    # Console Handler for logging to stderr
    if log_to_stderr:
        console_handler = logging.StreamHandler(stream=sys.stderr)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)

    # Configure the root logger to use this handler
    logging.basicConfig(level=log_level, handlers=handlers)
