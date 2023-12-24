from enum import Enum, IntEnum
from typing import Any


class StringEnum(str, Enum):
    def __eq__(self, other: Any):
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)


class Action(IntEnum):
    HOLD = 0
    BUY = 1
    SELL = 2


class OrderType(StringEnum):
    MARKET = 'MARKET'
    LIMIT = 'LIMIT'
    STOP_LOSS = 'STOP_LOSS'
    TAKE_PROFIT = 'TAKE_PROFIT'


class Side(StringEnum):
    BUY = 'BUY'
    SELL = 'SELL'


class TimeInForce(StringEnum):
    GoodTillCanceled = 'GTC'
    ImmediateOrCancel = 'IOC'
    FillOrKill = 'FOK'
    PostOnly = 'GTX'
