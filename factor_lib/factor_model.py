import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import *
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import numpy as np
import yfinance as yf
import quantstats as qs
from typing import Literal
from factor_lib.factor import Factor, yf_intervals
# from atom import ATOMClassifier
from xgboost import XGBRegressor

ModelType = Literal['hgb', 'gbr', 'adaboost', 'rf', 'et', 'linear', 'voting', 'xgb']


class FactorModel:
    def __init__(self, tickers=['AAPL', 'MSFT', 'TSLA'], interval='D'):
        self.tickers = tickers
        self.interval = interval
        self.factors = pd.DataFrame(columns=pd.MultiIndex.from_product([tickers, []]))
        self.model = None

    def add_factor(self, factor: Factor, replace=False):
        assert factor.tickers == self.tickers, 'Factor tickers must match model tickers'
        self.factors = pd.concat([self.factors, factor.data], axis=1)

    def fit(self, returns: pd.DataFrame, model: ModelType, voting_models=None,
            time='t',
            window=60,
            random_state=42,
            test_split=0.3, **kwargs):

        # Set up Training Data
        value = time.split('t+')
        if len(value) > 0:
            shift = value[1]
            returns = returns.shift(-1 * int(shift)) # shift returns back
        else:
            returns = returns

        try:
            returns.index = pd.to_datetime(returns.index)
        except Exception as e:
            print(f'could not convert index to datetime of returns, moving on.')

        try:
            returns.index = returns.index.tz_localize(None)
        except Exception as e:
            print(f'could not localize index of returns, moving on.')

        # align factor dates to be at the latest first date and earliest last date, so all values are accounted for
        # self.factors.dropna(inplace=True)
        valid_dates = self.factors.index
        if valid_dates[0] < returns.index[0]:
            valid_dates = valid_dates[valid_dates >= returns.index[0]]
        if valid_dates[-1] > returns.index[-1]:
            valid_dates = valid_dates[valid_dates <= returns.index[-1]]

        returns = returns.loc[valid_dates]
        returns = returns.to_period('D').to_timestamp()
        returns.index = pd.to_datetime(returns.index)
        dates = returns.index

        dates_train, dates_test = train_test_split(dates, test_size=test_split, random_state=42, shuffle=True)
        X_train = pd.DataFrame()
        X_test = pd.DataFrame()
        y_train = pd.Series(dtype=float)
        y_test = pd.Series(dtype=float)

        train_returns_concatenated = pd.Series(dtype=object)
        test_returns_concatenated = pd.Series(dtype=object)
        for ticker in self.tickers:
            X_train = pd.concat([X_train, self.factors[ticker].loc[dates_train]], axis=0)
            y_train = pd.concat([y_train, returns[ticker].loc[dates_train]], axis=0)

            X_test = pd.concat([X_test, self.factors[ticker].loc[dates_test]], axis=0)
            y_test = pd.concat([y_test, returns[ticker].loc[dates_test]], axis=0)

            train_returns_concatenated = pd.concat([train_returns_concatenated, returns[ticker].loc[dates_train]], axis=0)
            test_returns_concatenated = pd.concat([test_returns_concatenated, returns[ticker].loc[dates_test]], axis=0)

        X_train['returns'] = train_returns_concatenated
        X_train = X_train.dropna()
        y_train = X_train['returns']
        X_train.drop('returns', axis=1, inplace=True)

        X_test['returns'] = test_returns_concatenated
        X_test = X_test.dropna()
        y_test = X_test['returns']
        X_test.drop('returns', axis=1, inplace=True)

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

        print(f'model score: {self.model.score(X_test, y_test)}')
        return self.model

    def predict(self, factors: pd.DataFrame):
        return self.model.predict(factors)

    def backtest(self, start_date, end_date, candidates=None, k=10, long_pct=0.5):
        expected_returns = pd.DataFrame()
        for ticker in self.tickers:
            expected_returns[ticker] = self.model.predict(self.factors[ticker].loc[start_date:end_date]).flatten()
            expected_returns.index = self.factors[ticker].loc[start_date:end_date].index

        if candidates is not None:
            assert len(candidates.keys()) == len(expected_returns.index)
            binary_mask = pd.DataFrame(0, columns=self.tickers, index=expected_returns.index)
            for date in candidates.keys():
                binary_mask.loc[date, candidates[date]] = 1

            expected_returns = expected_returns * binary_mask

        positions = expected_returns.apply(self._get_positions, axis=1,
                                           args=(min(k, len(self.tickers) // 2), long_pct))[1:]
        positions.index = positions.index.tz_localize(None)

        stocks_data = yf.download(self.tickers, start=start_date, end=end_date,
                                  interval=yf_intervals[self.interval])['Adj Close']
        stocks_data = stocks_data.resample(self.interval, convention='end').ffill()
        stocks_data.fillna(value=0, inplace=True)
        returns = stocks_data.pct_change(1)
        returns.dropna(inplace=True)
        returns.index = pd.to_datetime(returns.index).tz_localize(None)

        if returns.index[0] > positions.index[0]:
            positions = positions.loc[returns.index[0]:returns.index[-1]]
        else:
            returns = returns.loc[positions.index[0]:positions.index[-1]]

        returns.index = pd.to_datetime(returns.index)
        positions.index = returns.index

        returns_per_stock = returns.mul(positions)
        portfolio_returns = returns_per_stock.sum(axis=1)

        returns = returns * (1 / len(self.tickers))

        # importing here to avoiod circular import
        from factor_lib.statistics import Statistics
        return Statistics(portfolio_returns, self)


    def _get_positions(self, row, k, long_pct):
        indices = np.argsort(row)
        topk = indices[:k]
        bottomk = indices[-k:]
        positions = [0] * len(row)

        for i in topk:
            positions[i] = (1 / k) * long_pct
        for i in bottomk:
            positions[i] = round((-1 / k) * (1 - long_pct), 2)

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