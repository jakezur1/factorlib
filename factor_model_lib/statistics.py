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
from prettytable import PrettyTable


class Statistics:
    def __init__(self, portfolio_returns, model: FactorModel,
                 extra_baselines: [pd.Series] = None, rebalancing_option=False):
        qs.extend_pandas()
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
        self.buy_hold_baseline = stock_returns / len(self.testing_model.tickers)
        self.buy_hold_baseline = self.buy_hold_baseline.sum(axis=1)
        self.buy_hold_baseline = self.buy_hold_baseline.loc[correct_index]
        self.buy_hold_baseline = pd.DataFrame(data={
            'buy_hold': self.buy_hold_baseline
        })

        spy_prices = yf.download(tickers='SPY', start=self.portfolio_returns.index[0],
                                 end=self.portfolio_returns.index[-1])['Adj Close']
        spy_prices = spy_prices.resample(self.testing_model.interval, convention='end').ffill()
        spy_returns = spy_prices.pct_change().dropna()
        spy_returns.index = pd.to_datetime(spy_returns.index).tz_localize(None).floor('D')
        spy_returns = spy_returns.loc[correct_index]
        spy_returns = pd.DataFrame(data={
            'spy': spy_returns
        })
        self.spy_baseline = spy_returns

        stock_prices = yf.download(tickers=self.testing_model.tickers, start=self.portfolio_returns.index[0],
                                   end=self.portfolio_returns.index[-1])['Adj Close']
        stock_prices = stock_prices.resample(self.testing_model.interval, convention='end').ffill()
        stock_returns = stock_prices.pct_change().dropna()

        positions = stock_prices
        positions = positions.apply(self._get_random_positions, axis=1,
                                    args=[min(20, len(self.testing_model.tickers) // 2)])
        self.random_baseline = stock_returns
        self.random_baseline = self.random_baseline.mul(positions)
        self.random_baseline = self.random_baseline.sum(axis=1)
        self.random_baseline.index = pd.to_datetime(self.random_baseline.index).tz_localize(None).floor('D')
        self.random_baseline = self.random_baseline.loc[correct_index]
        self.random_baseline = pd.DataFrame(data={
            'random': self.random_baseline
        })

        self.portfolio_returns = portfolio_returns.iloc[1:]
        self.portfolio_returns = self.portfolio_returns.to_frame()
        self.portfolio_returns.columns = ['factors']
        self.all_returns = [self.portfolio_returns, self.spy_baseline, self.buy_hold_baseline, self.random_baseline]
        if extra_baselines is not None:
            self.all_returns.extend(extra_baselines)

    def get_full_qs(self):
        qs.reports.full(self.portfolio_returns)

    def find_factor_significance(self):
        factor_significances = []
        for returns in self.all_returns:
            factor_significance = stats.ttest_rel(returns[returns.columns[0]],
                                                  self.portfolio_returns['factors'])
            factor_significances.append(round(factor_significance[1], 5))

        return factor_significances

    def print_statistics_report(self):
        print()
        print('{:<35s}'.format('FACTOR MODEL ANALYSIS REPORT'))
        print()
        print('{:<30s}'.format('Relative to baseline models:'))
        column_headers = [x.columns[0] for x in self.all_returns]
        column_headers.insert(0, 'metric:')
        statsTable = PrettyTable(column_headers)

        t_tests = ['paired t-test']
        t_tests.extend(self.find_factor_significance())
        sharpes = ['sharpe']
        sortinos = ['sortino']
        cagrs = ['cagr']
        avg_rtns = ['avg rtns']
        for returns in self.all_returns:
            sharpes.append(round(returns.sharpe().values[0], 3))
            sortinos.append(round(qs.stats.sortino(returns).values[0], 3))
            cagrs.append(str(round(qs.stats.cagr(returns).values[0] * 100, 2)) + '%')
            avg_rtns.append(str(round(qs.stats.avg_return(returns).values[0] * 100, 2)) + '%')

        statsTable.add_row(t_tests)
        statsTable.add_row(sharpes)
        statsTable.add_row(sortinos)
        statsTable.add_row(cagrs)
        print(statsTable)

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
