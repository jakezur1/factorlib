import warnings
from typing import Literal
from tqdm import tqdm
import numpy as np
import pandas as pd
import yfinance as yf
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
            k=100,
            long_only=False,
            pred_time='t+1',
            train_freq=None, **kwargs):

        assert (self.interval == 'D' or self.interval == 'W' or self.interval == 'M' or self.interval == 'Y'), \
            'Walk forward optimization currently only supports daily, weekly, monthly, or yearly intervals'

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

        training_start = start_date
        training_end = start_date + train_interval
        assert training_end < end_date, 'Training interval exceeds the total amount of data provided.'
        training_end = _get_end_convention(training_end, self.interval)
        self.model = XGBRegressor(n_jobs=-1, tree_method='hist', random_state=42, **kwargs)

        # perform walk forest optimization on factors data and record expected returns
        # at each time step
        expected_returns = pd.DataFrame()
        expected_returns_index = []

        # using for loop for tqdm progress bar
        loop_start = training_end
        loop_end = self.offset_datetime(end_date, sign=-1)
        loop_range = pd.date_range(loop_start, loop_end,
                                   freq=self.interval)
        for index, date in enumerate(tqdm(loop_range)):
            X_train = pd.DataFrame()
            y_train = pd.Series(dtype=object)
            start = time.time()
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
                for ticker in self.tickers:
                    X_train = pd.concat([X_train, self.factors[ticker].loc[training_start:training_end]], axis=0)
                    y_train = pd.concat([y_train, shifted_returns[ticker].loc[training_start:training_end]], axis=0)
                X_train, y_train = _clean_data(X_train, y_train, drop_columns=True)
                # print('\nTook', time.time() - start, 'seconds to curate data')

                start = time.time()
                valid_columns = X_train.columns
                self.model.fit(X_train, y_train)
                # print('Took', time.time() - start, 'seconds to fit model')

            # get predictions
            # this is our OOS sample test (that's one timestep ahead)
            start = time.time()
            test_end = self.offset_datetime(training_end)
            test_end = pd.to_datetime(test_end).to_period('D').to_timestamp()
            test_end = pd.to_datetime(test_end)
            test_end = _get_end_convention(test_end, self.interval)

            curr_predictions = pd.DataFrame()
            for ticker in self.tickers:
                prediction_data = self.factors[ticker][valid_columns].loc[test_end].to_frame().T
                # if prediction_data.isna().sum().sum() >= 1:
                #     curr_predictions[ticker] = np.nan
                #     continue
                curr_predictions[ticker] = self.model.predict(prediction_data).flatten()
                expected_returns_index.append(prediction_data.index)
            expected_returns = pd.concat([expected_returns, curr_predictions], axis=0)

            # print('\nTook', time.time() - start, 'seconds to get expected returns')

            # calculate new intervals to train
            if not anchored:
                training_start = self.offset_datetime(training_start)
                training_start = np.array([training_start], dtype='datetime64[D]')[0]
                training_start = pd.to_datetime(training_start)
                training_start = _get_end_convention(training_start, self.interval)

            training_end = self.offset_datetime(training_end)
            training_end = np.array([training_end], dtype='datetime64[D]')[0]
            training_end = pd.to_datetime(training_end)
            training_end = _get_end_convention(training_end, self.interval)

        expected_returns_index = np.array(expected_returns_index, dtype='datetime64[D]')
        expected_returns_index = np.unique(expected_returns_index)
        expected_returns.index = expected_returns_index
        print('Expected returns: ')
        print(f'{expected_returns}\n')

        # get positions
        positions = expected_returns.apply(self._get_positions, axis=1,
                                           args=(k, long_only))
        positions.index = positions.index.tz_localize(None)

        # align positions and returns
        returns = _delocalize_datetime(returns)
        positions, returns = _align_by_date_index(positions, returns)

        # calculate back tested returns
        returns_per_stock = returns.mul(positions.shift(1))  # you have to shift positions by 1 day
        portfolio_returns = returns_per_stock.sum(axis=1)

        # importing here to avoid circular import
        from .statistics import Statistics
        return Statistics(portfolio_returns, self, predicted_returns=expected_returns, stock_returns=returns)

    def fit(self, returns: pd.DataFrame, model: ModelType, voting_models=None,
            pred_time='t+1',
            window=60,
            random_state=42,
            test_split=0.3, **kwargs):

        # shift returns back by 'time' time steps
        returns = _shift_by_time_step(pred_time, returns)

        # ensure that index is datetime and delocalized
        returns = _delocalize_datetime(returns)

        # align factor dates to be at the latest first date and earliest last date
        _, returns = _align_by_date_index(self.factors, returns)
        dates = returns.index

        # dates_train, dates_test = train_test_split(dates, test_size=test_split, random_state=42, shuffle=True)
        X_train = pd.DataFrame()
        # X_test = pd.DataFrame()
        y_train = pd.Series(dtype=float)
        # y_test = pd.Series(dtype=float)
        for ticker in self.tickers:
            X_train = pd.concat([X_train, self.factors[ticker].loc[dates]], axis=0)
            y_train = pd.concat([y_train, returns[ticker].loc[dates]], axis=0)

            # X_test = pd.concat([X_test, self.factors[ticker].loc[dates_test]], axis=0)
            # y_test = pd.concat([y_test, returns[ticker].loc[dates_test]], axis=0)

        X_train, y_train = _clean_data(X_train, y_train)
        # X_test, y_test = _clean_data(X_test, y_test)

        if model == 'linear':
            self.model = RollingOLS(y_train, X_train, window=window).fit()
            return self.model
        elif model == 'voting':
            assert len(voting_models) > 1
            self.model = VotingRegressor(estimators=[(model, self._get_model(model, **kwargs))
                                                     for model in voting_models])
        else:
            self.model = self._get_model(model, **kwargs)
        self.model.fit(X_train, y_train)

        print(f'model score on train: {self.model.score(X_train, y_train)}')
        # print(f'model score on test: {self.model.score(X_test, y_test)}')
        return self.model

    def predict(self, factors: pd.DataFrame):
        return self.model.predict(factors)

    def backtest(self, start_date, end_date,
                 returns=None,
                 training_start_date=None,
                 candidates=None,
                 k=10,
                 long_pct=0.5):
        if returns is None:
            stocks_data = yf.download(self.tickers, start=start_date, end=end_date,
                                      interval=yf_intervals[self.interval])['Adj Close']
            stocks_data.index = pd.to_datetime(stocks_data.index)
            stocks_data.fillna(value=0, inplace=True)
            returns = stocks_data.pct_change(1)
            returns.dropna(inplace=True)
            returns.index = pd.to_datetime(returns.index).tz_localize(None)
        else:
            returns = returns.loc[start_date:end_date]

        returns.index = pd.to_datetime(returns.index)

        expected_returns = pd.DataFrame()
        for ticker in self.tickers:
            expected_returns[ticker] = self.model.predict(self.factors[ticker].loc[start_date:end_date]).flatten()
            expected_returns.index = self.factors[ticker].loc[start_date:end_date].index

        predicted_returns = expected_returns

        if candidates is not None:
            assert len(candidates.keys()) == len(expected_returns.index)
            binary_mask = pd.DataFrame(0, columns=self.tickers, index=expected_returns.index)
            for date in candidates.keys():
                binary_mask.loc[date, candidates[date]] = 1

            expected_returns = expected_returns * binary_mask

        # positions are predicted one day before

        print("Expected Returns:")
        print(expected_returns)
        positions = expected_returns.apply(self._get_positions, axis=1,
                                           args=(min(k, len(self.tickers) // 2), long_pct))
        positions.index = positions.index.tz_localize(None)

        positions.index = returns.index

        if returns.index[0] > positions.index[0]:
            positions = positions.loc[returns.index[0]:returns.index[-1]]
        else:
            returns = returns.loc[positions.index[0]:positions.index[-1]]

        returns_per_stock = returns.mul(positions.shift(1))  # you have to shift positions by 1 day
        portfolio_returns = returns_per_stock.sum(axis=1)

        # importing here to avoid circular import
        from .statistics import Statistics
        return Statistics(portfolio_returns, self, predicted_returns=predicted_returns, stock_returns=returns)

    def _get_positions(self, row, k=5, long_only=False):
        """Given a quintile and a row, use pandas qcut to
        create equal long short positions"""
        # qcut also handles nans, so we don't have to worry about them
        labels = pd.qcut(row, q=k, labels=False, duplicates='drop')
        positions = pd.Series([0.0] * len(row), index=self.tickers)
        if not long_only:
            positions[labels == 0] = -1 / len(labels == 0)  # bottom quantile
        positions[labels == k - 1] = 1 / len(labels == k - 1)  # top quantile
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

    def offset_datetime(self, date: datetime, sign=1):
        if self.interval == 'D':
            date += sign * pd.DateOffset(days=1)
        elif self.interval == 'W':
            date += sign * pd.DateOffset(days=7)
        elif self.interval == 'M':
            date += sign * pd.DateOffset(months=1)
        elif self.interval == 'Y':
            date += sign * pd.DateOffset(years=1)
        return date
