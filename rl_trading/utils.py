import pandas as pd
import time
from functools import wraps
from ray.rllib.utils.checkpoints import get_checkpoint_info
from ray.rllib.algorithms.r2d2 import R2D2
from ray.rllib.algorithms.algorithm import Algorithm


def granularity_convert(source_granularity: str, target_granularity: str = '1min'):
    return pd.to_timedelta(source_granularity).total_seconds() // pd.to_timedelta(target_granularity).total_seconds()


def load_model(model_path):
    checkpoint_info = get_checkpoint_info(model_path)
    checkpoint_state = Algorithm._checkpoint_info_to_algorithm_state(checkpoint_info)
    if issubclass(checkpoint_state['algorithm_class'], R2D2):
        checkpoint_state['config']['replay_buffer_config']['replay_sequence_length'] = -1
    return Algorithm.from_state(checkpoint_state)


class RateLimited:
    def __init__(self, max_calls_per_minute):
        self.max_calls_per_minute = max_calls_per_minute
        self.calls = []

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
