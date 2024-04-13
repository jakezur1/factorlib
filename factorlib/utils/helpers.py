import inspect
import numpy as np
import pandas as pd
import warnings

from datetime import datetime

from factorlib.utils.system import FactorlibUserWarning, print_warning


def offset_datetime(date: datetime, interval: str, sign=1):
    if interval == 'D' or interval == 'B':
        date += sign * pd.DateOffset(days=1)
    elif interval == 'W':
        date += sign * pd.DateOffset(days=7)
    elif interval == 'M':
        date += sign * pd.DateOffset(months=1)
    elif interval == 'Y':
        date += sign * pd.DateOffset(years=1)
    return date


def shift_by_time_step(time: str, returns: pd.DataFrame, backwards: bool = False):
    value = time.split('t+')
    if len(value) > 0:
        shift = value[1]
        if backwards:
            shift = int(shift) * -1
        returns = returns.groupby('ticker').shift(-1 * int(shift))  # shift returns back
    else:
        returns = returns
        print_warning(message='The time_step you have passed to wfo(...) is invalid or equal to 0. Please see the '
                              'docstring in factor_model.py for information on time_step formatting.',
                      category=FactorlibUserWarning.TimeStep)
    return returns


def get_subset_by_date_bounds(df: pd.DataFrame, start_date: datetime = None, end_date: datetime = None):
    if start_date is None:
        if len(df.index.names) > 1:
            start_date = df.index.get_level_values('date')[0]
        else:
            start_date = df.index[0]
    if end_date is None:
        if len(df.index.names) > 1:
            end_date = df.index.get_level_values('date')[-1]
        else:
            end_date = df.index[-1]
    if len(df.index.names) > 1:
        return df.loc[(slice(start_date, end_date), slice(None)), :]
    else:
        return df.loc[slice(start_date, end_date)]


def clean_data(X: pd.DataFrame, y: pd.Series):
    X['returns'] = y
    X.dropna(subset=['returns'], inplace=True)  # only look for NaNs in returns, otherwise keep NaNs
    y = X['returns']
    X.drop('returns', axis=1, inplace=True)
    X.replace([np.inf, -np.inf], 0, inplace=True)
    return X, y


def calc_compounded_returns(returns: pd.Series):
    return returns.add(1).cumprod() - 1

    
def _get_nearest_month_begin(date: datetime):
    start_of_month = pd.Timestamp(date.year, date.month, 1)
    start_of_next_month = start_of_month + pd.offsets.MonthBegin(1)

    if (date - start_of_month) < (start_of_next_month - date):
        return start_of_month
    else:
        return start_of_next_month


def _get_nearest_month_end(date: datetime):
    start_of_month = pd.Timestamp(date.year, date.month, 1)
    end_of_previous_month = start_of_month + pd.offsets.MonthEnd(-1)
    end_of_current_month = start_of_month + pd.offsets.MonthEnd(1)

    if (date - end_of_previous_month) < (end_of_current_month - date):
        return end_of_previous_month
    else:
        return end_of_current_month


def _set_index_names_adaptive(df: pd.DataFrame):
    names = ['date', 'ticker'] if df.index.get_level_values(1).dtype == datetime \
        else ['ticker', 'date']
    return names
