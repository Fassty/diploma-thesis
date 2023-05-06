from abc import ABC, abstractmethod
import numpy as np

from rl_trading.enums import Action


class AbstractStrategy(ABC):
    @abstractmethod
    def get_action(self, current_price):
        pass


class AbstractPriceEstimator(ABC):
    def __init__(self, relative: bool):
        self.relative = relative

    @abstractmethod
    def register(self, current_price: float):
        pass

    @abstractmethod
    def predict_next(self) -> float:
        pass


class RandomStrategy(AbstractStrategy):
    """Randomly pick either BUY or SELL"""

    def get_action(self, current_price):
        return 1 if np.random.uniform() < 0.5 else -1


class BaselineStrategy(AbstractStrategy):
    """
    Simple strategy that tries to BUY if the current price is smaller than the last price
    and tries to SELL otherwise
    """

    def __init__(self, threshold: float = 0.0):
        self.last_price = 0
        self.threshold = threshold

    def get_action(self, current_price: float):
        if self.last_price == 0:
            relative_change = 0
        else:
            relative_change = (current_price - self.last_price) / self.last_price

        action = 0
        if abs(relative_change) > self.threshold:
            if self.last_price > current_price:
                action = 1
            elif self.last_price < current_price:
                action = 2
        self.last_price = current_price
        return action


class ReactiveStrategy(AbstractStrategy):
    """Uses an estimator to predict the next price and buys/sells if its higher/lower than the current price."""

    def __init__(self, price_estimator, threshold=None):
        self.price_estimator = price_estimator
        self.threshold = threshold

    def get_action(self, current_price):
        self.price_estimator.register(current_price)
        pred = self.price_estimator.predict_next()

        if self.threshold is None or abs(pred) >= self.threshold:
            if self.price_estimator.relative:
                return Action.BUY if pred > 0 else Action.SELL
            else:
                return Action.BUY if pred > current_price else Action.SELL


class SlidingWindowEstimator(AbstractPriceEstimator):
    """Uses given model to predict the next price if there is enough historic data, otherwise just predicts that
    the next price will equal the current price."""

    def __init__(self, window_length: int, model, relative: bool):
        super().__init__(relative)
        self.window_length = window_length
        self.model = model

        self.history = []

    def register(self, current_price: float):
        self.history.append(current_price)
        if len(self.history) > self.window_length:
            self.history.pop(0)

    def predict_next(self) -> float:
        if len(self.history) < self.window_length:
            return self.history[-1]
        return self.model(np.asarray([self.history]))
