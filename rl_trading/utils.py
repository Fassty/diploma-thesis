import pandas as pd


def granularity_convert(source_granularity: str, target_granularity: str = '1min'):
    return pd.to_timedelta(source_granularity).total_seconds() // pd.to_timedelta(target_granularity).total_seconds()
