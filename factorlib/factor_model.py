import warnings
from typing import Literal
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import time
from datetime import datetime, timedelta
from sklearn.ensemble import *
from statsmodels.regression.rolling import RollingOLS
from xgboost import XGBRegressor

from .factor import Factor
from .utils import *

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

ModelType = Literal['hgb', 'gbr', 'adaboost', 'rf', 'et', 'linear', 'voting', 'xgb']


class FactorModel:
    def __init__(self, tickers=None, interval='D'):
        assert tickers is not None, 'tickers cannot be None'

        self.tickers = tickers
        self.interval = interval
        self.factors = pd.DataFrame(columns=pd.MultiIndex.from_product([tickers, []]))
        self.model = None
        self.earliest_start = None
        self.latest_end = None

    def add_factor(self, factor: Factor, replace=False):
        assert set(self.tickers).issubset(set(factor.tickers)), 'Factor tickers must include model tickers'
        self.factors = pd.concat([self.factors, factor.data], axis=1)

        # set the maximum bounds for a wfo or a backtest
        if self.earliest_start is None:
            self.earliest_start = factor.start
        else:
            if self.earliest_start < factor.start:
                self.earliest_start = factor.start
        if self.latest_end is None:
            self.latest_end = factor.end
        else:
            if self.latest_end > factor.end:
                self.latest_end = factor.end

    def predict(self, factors: pd.DataFrame):
        return self.model.predict(factors)

    def wfo(self, returns: pd.DataFrame, train_interval: timedelta,
            start_date: datetime = None,
            end_date: datetime = None,
            anchored=True,
            k_pct=0.2,
            long_pct=0.5,
            long_only=False,
            short_only=False,
            pred_time='t+1',
            train_freq=None,
            candidates=None, **kwargs):

        assert (self.interval == 'D' or self.interval == 'W' or self.interval == 'M' or self.interval == 'Y'), \
            'Walk forward optimization currently only supports daily, weekly, monthly, or yearly intervals'

        if train_freq is not None:
            print('the train_freq parameter does not have stable implementation yet. '
                  'Defaulting to monthly (\'M\') training.')
            if timedelta_intervals[train_freq] > timedelta_intervals[self.interval]:
                print('The chose train_freq is a shorter interval than the model interval. '
                      'Defaulting to monthly (\'M\') training.')
                train_freq = 'M'

        assert (not (long_only and short_only)), 'long_only and short_only cannot both be True'

        if start_date is not None:
            assert (start_date > self.earliest_start), 'start_date must be after earliest start date ' \
                                                       '(based on the data provided)'
        else:
            start_date = self.earliest_start

        if end_date is not None:
            assert (end_date < self.latest_end), 'end_date must be before latest end date'
        else:
            end_date = self.latest_end

        start_date = _get_end_convention(start_date, self.interval)
        end_date = _get_end_convention(end_date, self.interval)

        print('Starting Walk-Forward Optimization from', start_date, 'to', end_date, 'with a',
              train_interval.days / 365, 'year training interval')

        # shift returns back by 'time' time steps
        shifted_returns = _shift_by_time_step(pred_time, returns)

        # ensure that index is datetime and delocalized
        shifted_returns = _delocalize_datetime(shifted_returns)

        # align factor dates to be at the latest first date and earliest last date
        _, shifted_returns = _align_by_date_index(self.factors, shifted_returns)

        # stack the factors and returns of each ticker for easier training
        start = time.time()
        training_data = self.factors.stack(level=0)
        start = time.time()
        stacked_returns = shifted_returns.stack(level=0)

        # set the frequency of training
        frequency = None
        if train_freq is None:
            if timedelta_intervals[self.interval] >= timedelta_intervals['M']:
                frequency = 'M'
            else:
                frequency = self.interval
        else:
            frequency = train_freq

        training_start = start_date
        training_start = _get_end_convention(training_start, frequency)
        training_end = start_date + train_interval
        training_end = _get_end_convention(training_end, frequency)
        assert training_end < end_date, 'Training interval exceeds the total amount of data provided.'
        training_end = _get_end_convention(training_end, self.interval)
        self.model = XGBRegressor(n_jobs=-1, tree_method='hist', random_state=42, **kwargs)

        # perform walk forest optimization on factors data and record expected returns
        # at each time step

        # initialize statistics data
        expected_returns = pd.DataFrame()
        expected_returns_index = []
        training_spearman = pd.Series(dtype=object)

        # using for loop for tqdm progress bar
        loop_start = training_end
        loop_end = self.offset_datetime(end_date, sign=-1, override_interval=frequency)
        loop_range = pd.date_range(loop_start, loop_end, freq=frequency)

        for index, date in enumerate(tqdm(loop_range)):
            # check if we should train the model on this iteration
            train_this_iteration = False
            if timedelta_intervals[self.interval] <= timedelta_intervals['M']:
                train_this_iteration = True
            else:
                last_day_of_month = _get_end_convention(datetime(training_end.year, training_end.month, 20),
                                                        self.interval)
                if self.interval == 'D':
                    if pd.Timestamp(training_end).is_month_end:
                        train_this_iteration = True
                elif self.interval == 'W':
                    # check if date is in the last week of the month
                    if (last_day_of_month - training_end).days < 7:
                        train_this_iteration = True
            if index == 0:
                train_this_iteration = True

            if train_this_iteration:
                start = time.time()
                X_train = training_data.loc[training_start:training_end]
                y_train = stacked_returns.loc[training_start:training_end]
                X_train, y_train = X_train.align(y_train, axis=0)
                X_train, y_train = _clean_data(X_train, y_train, drop_columns=True)

                start = time.time()
                valid_columns = X_train.columns
                self.model.fit(X_train, y_train)
                # print('Took', time.time() - start, 'seconds to fit model')

                if index != 0:
                    training_predictions = self.predict(X_train)
                    training_predictions = pd.DataFrame(training_predictions, index=X_train.index)
                    training_predictions = training_predictions.unstack(level=1).droplevel(0, axis=1)
                    X_train = X_train.unstack(level=1).swaplevel(1, 0, axis=1)
                    y_train = y_train.unstack(level=1)
                    returns_for_spearman = returns.loc[X_train.index]
                    spearman = returns_for_spearman.corrwith(training_predictions, method='spearman', axis=1).mean()
                    spearman = pd.Series(spearman, index=[training_predictions.index[-1]])
                    training_spearman = pd.concat([training_spearman, spearman])

                # get predictions
                # this is our OOS sample test (that's one timestep ahead)
                start = time.time()
                pred_start = self.offset_datetime(training_end)
                pred_start = pd.to_datetime(pred_start).to_period('D').to_timestamp()
                pred_start = pd.to_datetime(pred_start)
                pred_start = _get_end_convention(pred_start, self.interval)
                pred_end = _get_end_convention(pred_start, frequency)

                curr_predictions = pd.DataFrame()
                for ticker in self.tickers:
                    prediction_data = self.factors[ticker][valid_columns].loc[pred_start:pred_end]
                    # if prediction_data.isna().sum().sum() >= 1:
                    #     curr_predictions[ticker] = np.nan
                    #     continue
                    curr_predictions[ticker] = self.model.predict(prediction_data).flatten()
                    expected_returns_index.extend(prediction_data.index.values)

                expected_returns = pd.concat([expected_returns, curr_predictions], axis=0)

            # calculate new intervals to train
            if not anchored:
                training_start = self.offset_datetime(training_start, override_interval=frequency)
                training_start = np.array([training_start], dtype='datetime64[D]')[0]
                training_start = pd.to_datetime(training_start)
                training_start = _get_end_convention(training_start, self.interval)

            training_end = self.offset_datetime(training_end, override_interval=frequency)
            training_end = np.array([training_end], dtype='datetime64[D]')[0]
            training_end = pd.to_datetime(training_end)
            training_end = _get_end_convention(training_end, self.interval)

        training_spearman = training_spearman.resample(self.interval).bfill()

        expected_returns_index = np.array(expected_returns_index, dtype='datetime64[D]')
        expected_returns_index = np.unique(expected_returns_index)
        expected_returns.index = expected_returns_index
        print('Expected returns: ')
        print(f'{expected_returns}\n')

        if candidates is not None:
            assert len(candidates.keys()) == len(expected_returns.index)
            binary_mask = pd.DataFrame(np.nan, columns=self.tickers, index=expected_returns.index)
            for date in candidates.keys():
                binary_mask.loc[date, candidates[date]] = 1

            expected_returns = expected_returns * binary_mask

        # get positions
        positions = expected_returns.apply(self._get_positions, axis=1,
                                           k_pct=k_pct, long_pct=long_pct,
                                           long_only=long_only, short_only=short_only)
        positions.index = positions.index.tz_localize(None)

        # align positions and returns
        returns = _delocalize_datetime(returns)
        positions, returns = _align_by_date_index(positions, returns)

        # calculate back tested returns
        returns_per_stock = returns.mul(positions.shift(1))  # you have to shift positions by 1 day
        portfolio_returns = returns_per_stock.sum(axis=1)

        # importing here to avoid circular import
        from .statistics import Statistics
        return Statistics(portfolio_returns, self, predicted_returns=expected_returns, stock_returns=returns,
                          position_weights=positions, training_spearman=training_spearman)

    def save(self, name):
        with open(f'{name}.p', 'wb') as f:
            pickle.dump(self, f)

    def load(self, path):
        with open(path, 'rb') as f:
            loaded_model = pickle.load(f)
        self.__dict__.update(loaded_model.__dict__)

    def predict(self, factors: pd.DataFrame):
        return self.model.predict(factors)

    def _get_positions(self, row, k_pct=0.2, long_pct=0.5, long_only=False, short_only=False):
        """Given a row of returns and a percentage of stocks to long and short,
        return a row of positions of equal long and short positions, with weights
        equal to long_pct and 1 - long_pct respectively."""

        num_na = int(row.isna().sum())
        indices = np.argsort(row)[:-num_na]  # sorted in ascending order
        if num_na == 0:
            indices = np.argsort(row)  # sorted in ascending order
        k = int(np.floor(len(indices) * k_pct))
        bottomk = indices[:k]
        topk = indices[-k:]
        positions = [0] * len(row)

        if long_only:
            long_pct = 1.0
        elif short_only:
            long_pct = 0.0

        for i in topk:
            positions[i] = round((1 / k) * long_pct, 3)
        for i in bottomk:
            positions[i] = round((-1 / k) * (1 - long_pct), 3)
        return pd.Series(positions, index=self.tickers)

    def _get_model(self, model, **kwargs):
        if model == 'hgbm':
            self.model = HistGradientBoostingRegressor(**kwargs)
        elif model == 'gbr':
            self.model = GradientBoostingRegressor(**kwargs)
        elif model == 'adaboost':
            self.model = AdaBoostRegressor(**kwargs)
        elif model == 'rf':
            self.model = RandomForestRegressor(**kwargs)
        elif model == 'et':
            self.model = ExtraTreesRegressor(**kwargs)
        elif model == 'xgb':
            self.model = XGBRegressor(**kwargs)
        return self.model

    def offset_datetime(self, date: datetime, sign=1, override_interval=None):
        if override_interval is not None:
            model = FactorModel(self.tickers, override_interval)
            date = model.offset_datetime(date, sign=sign)
            date = _get_end_convention(date, override_interval)
        elif self.interval == 'D':
            date += sign * pd.DateOffset(days=1)
        elif self.interval == 'W':
            date += sign * pd.DateOffset(days=7)
        elif self.interval == 'M':
            date += sign * pd.DateOffset(months=1)
        elif self.interval == 'Y':
            date += sign * pd.DateOffset(years=1)
        return date
