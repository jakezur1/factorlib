import datetime

import pandas as pd
import numpy as np
import quantstats as qs
import yfinance as yf
from factor_model_lib.factor_model import FactorModel
from scipy import stats
import random
from dateutil.relativedelta import relativedelta
from datetime import datetime


class Statistics:
    def __init__(self, portfolio_returns, model: FactorModel,
                 extra_baselines: [pd.Series] = None, rebalancing_option=False):
        self.testing_model = model
        self.portfolio_returns = portfolio_returns
        self.portfolio_returns.index = pd.to_datetime(self.portfolio_returns.index).tz_localize(None) \
            .floor('D')
        self.portfolio_returns = portfolio_returns.resample(self.testing_model.interval, convention='end').ffill()
        correct_index = self.portfolio_returns[1:].index

        # need to populate these with actual returns to compare
        stock_prices = yf.download(tickers=self.testing_model.tickers,
                                   start=self.portfolio_returns.index[0],
                                   end=self.portfolio_returns.index[-1])['Adj Close']
        stock_prices.index = pd.to_datetime(stock_prices.index).tz_localize(None).floor('D')
        stock_prices = stock_prices.resample(self.testing_model.interval, convention='end').ffill()

        stock_returns = stock_prices.pct_change().dropna()
        stock_returns = stock_returns.loc[correct_index]

        self.buy_hold_baseline = stock_returns
        for ticker in self.testing_model.tickers:
            self.buy_hold_baseline[ticker] /= len(self.testing_model.tickers)
        self.buy_hold_baseline = self.buy_hold_baseline.sum(axis=1)
        self.buy_hold_baseline.index = pd.to_datetime(self.buy_hold_baseline.index).tz_localize(None).floor('D')
        self.buy_hold_baseline = self.buy_hold_baseline.resample(self.testing_model.interval, convention='end').ffill()
        self.buy_hold_baseline = self.buy_hold_baseline.loc[correct_index]
        self.buy_hold_baseline = pd.DataFrame(data={
            'buy_hold_returns': self.buy_hold_baseline
        })

        spy_prices = yf.download(tickers='SPY', start=self.portfolio_returns.index[0],
                                 end=self.portfolio_returns.index[-1])['Adj Close']

        spy_prices.index = pd.to_datetime(spy_prices.index).tz_localize(None).floor('D')
        spy_prices = spy_prices.resample(self.testing_model.interval, convention='end').ffill()
        spy_returns = spy_prices.pct_change().dropna()
        spy_prices = spy_prices.loc[correct_index]

        spy_returns = pd.DataFrame(data={
            'spy_returns': spy_returns
        })
        self.spy_baseline = spy_returns

        stock_prices = yf.download(tickers=self.testing_model.tickers, start=self.portfolio_returns.index[0],
                                   end=self.portfolio_returns.index[-1])['Adj Close']
        self.random_baseline = stock_prices.pct_change().dropna()
        self.random_baseline = self.random_baseline.apply(self._get_random_positions, axis=1,
                                                          args=[min(20, len(self.testing_model.tickers) // 2)])
        self.random_baseline = self.random_baseline.sum(axis=1)
        self.random_baseline.index = pd.to_datetime(self.random_baseline.index).tz_localize(None).floor('D')
        self.random_baseline = self.random_baseline.resample(self.testing_model.interval, convention='end').ffill()
        self.random_baseline = self.random_baseline.loc[correct_index]

        self.random_baseline = pd.DataFrame(data={
            'random_returns': self.random_baseline
        })

        self.portfolio_returns = portfolio_returns.iloc[1:]
        self.all_returns = [self.portfolio_returns, self.spy_baseline, self.buy_hold_baseline, self.random_baseline]
        if extra_baselines is not None:
            self.all_returns.extend(extra_baselines)

    def get_full_qs(self):
        qs.reports.full(self.portfolio_returns)

    def find_factor_significance(self):
        buy_hold_pt_test = stats.ttest_rel(self.buy_hold_baseline['buy_hold_returns'], self.portfolio_returns.values)
        spy_pt_test = stats.ttest_rel(self.spy_baseline['spy_returns'], self.portfolio_returns.values)
        random_pt_test = stats.ttest_rel(self.random_baseline['random_returns'], self.portfolio_returns.values)
        return buy_hold_pt_test, spy_pt_test, random_pt_test

    def print_statistics_report(self):
        print()
        print('{:<35s}'.format('FACTOR MODEL ANALYSIS REPORT'))
        print()
        print('{:<30s}'.format('Relative to baseline models:'))
        self._print_ascii_border_top(header=True)
        print('{:^1s} {:^8s} {:^1s} {:^12s} {:^1s} {:^12s} {:^1s} {:^12s} {:^1s} {:^12s} {:^1s}'
              .format('|', ' metric:', '|', 'Your Model', '|', 'SPY BL', '|', 'Buy/Hold BL', '|', ' Random BL', '|'))
        self._print_ascii_border_top(header=True)
        self._print_single_stat('sharpe', 1.004, 1.156, 1.133, 0.75)
        # print('{:10s} {:10s}  {:7.2f}'.format('xxx', '123', 98))
        # print('{:10s} {:3d}  {:7.2f}'.format('xxx', 123, 98))
        # print('{:10s} {:3d}  {:7.2f}'.format('yyyy', 3, 1.0))
        # print('{:10s} {:3d}  {:7.2f}'.format('zz', 42, 123.34))

    def _print_ascii_border_top(self, header=False):
        if header:
            print('{:^1s} {:^8s} {:^1s} {:^12s} {:^1s} {:^12s} {:^1s} {:^12s} {:^1s} {:^12s} {:^1s}'
                  .format('|', '========', '|', '============', '|', '============', '|', '============', '|',
                          '============', '|'))
        else:
            print('{:^1s} {:^8s} {:^1s} {:^12s} {:^1s} {:^12s} {:^1s} {:^12s} {:^1s} {:^12s} {:^1s}'
                  .format('|', '--------', '|', '------------', '|', '------------', '|', '------------', '|',
                          '------------', '|'))

    def _print_single_stat(self, stat_name, model_stat, spy_stat, buy_hold_stat, random_stat):
        print('{:^1s} {:<8s} {:^1s} {:>12.2f} {:^1s} {:>12.2f} {:^1s} {:>12.2f} {:^1s} {:>12.2f} {:^1s}'
              .format('|', stat_name, '|', model_stat, '|', spy_stat, '|', buy_hold_stat, '|', random_stat, '|'))
        self._print_ascii_border_top()

    def _get_random_positions(self, row, k):
        indices = np.argsort(row)
        random.shuffle(indices)

        # calculate long weights, must equal 1
        long_weights = np.random.random(len(indices))
        long_weights /= (np.sum(long_weights))

        topk = indices[:k]
        bottomk = indices[-k:]
        positions = [0] * len(row)
        for index, i in enumerate(topk):
            positions[i] = (1 / k) * long_weights[index]
        for index, i in enumerate(bottomk):
            positions[i] = round((-1 / k) * (1 - long_weights[index]), 2)

        return pd.Series(positions, index=self.testing_model.tickers)
