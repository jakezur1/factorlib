import inspect
import numpy as np
import pandas as pd
from datetime import datetime

__all__ = ['_delocalize_datetime',
           '_shift_by_time_step',
           '_align_by_date_index',
           '_clean_data',
           '_get_end_convention',
           'timedelta_intervals',
           'yf_intervals']


def _get_defining_class(meth):
    if inspect.ismethod(meth):
        for cls in inspect.getmro(meth.__self__.__class__):
            if meth.__name__ in cls.__dict__:
                return cls
    if inspect.isfunction(meth):
        return getattr(inspect.getmodule(meth),
                       meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0],
                       None)
    return None


def _shift_by_time_step(time: str, returns: pd.DataFrame):
    value = time.split('t+')
    if len(value) > 0:
        shift = value[1]
        returns = returns.shift(-1 * int(shift))  # shift returns back
    else:
        returns = returns
    return returns


def _delocalize_datetime(df: pd.DataFrame):
    try:
        df.index = pd.to_datetime(df.index)
    except Exception as e:
        print(f'could not convert index to datetime of returns, moving on.')

    try:
        df.index = df.index.tz_localize(None)
    except Exception as e:
        print(f'could not localize index of returns, moving on.')

    return df


def _align_by_date_index(df1: pd.DataFrame, df2: pd.DataFrame):
    valid_dates = df1.index
    if valid_dates[0] < df2.index[0]:
        valid_dates = valid_dates[valid_dates >= df2.index[0]]
    if valid_dates[-1] > df2.index[-1]:
        valid_dates = valid_dates[valid_dates <= df2.index[-1]]

    df1 = df1.loc[valid_dates]
    df1 = df1.to_period('D').to_timestamp()
    df1.index = pd.to_datetime(df1.index)

    df2 = df2.loc[valid_dates]
    df2 = df2.to_period('D').to_timestamp()
    df2.index = pd.to_datetime(df2.index)

    return df1, df2


def _clean_data(X: pd.DataFrame, y: pd.Series, drop_columns=False, col_thresh=0.5):
    if drop_columns:
        num_values_required = len(X) * (1 - col_thresh)
        X.dropna(axis=1, thresh=num_values_required, inplace=True)
    X['returns'] = y
    X.dropna(subset=['returns'], inplace=True) # only look for NaNs in returns, otherwise keep NaNs
    y = X['returns']
    X.drop('returns', axis=1, inplace=True)
    X.replace([np.inf, -np.inf], 0, inplace=True)
    return X, y


def _get_end_convention(date: datetime, interval: str):
    temp_df = pd.DataFrame(index=[date])
    temp_df.index = pd.to_datetime(temp_df.index)
    temp_df.index = temp_df.index.tz_localize(None)
    temp_df = temp_df.resample(interval, convention='end').ffill()
    end_convention = temp_df.index[0]
    return end_convention


timedelta_intervals = {
    '1m': 525600,
    '5m': 105120,
    '15m': 35040,
    '30m': 17520,
    '60m': 8760,
    '90m': 5840,
    '1h': 4380,
    'D': 365,
    '5D': 73,
    'W': 52,
    'M': 12,
    '3M': 4,
    'Y': 1
}

yf_intervals = {
    '1m': '1m',
    '2m': '2m',
    '5m': '5m',
    '15m': '15m',
    '30m': '30m',
    '60m': '60m',
    '90m': '90m',
    '1h': '1h',
    'D': '1d',
    '5D': '5d',
    'W': '1wk',
    'M': '1mo',
    '3M': '3mo',
}
