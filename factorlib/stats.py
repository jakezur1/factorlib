import pandas as pd
import numpy as np
import quantstats as qs
import yfinance as yf
import shap
import joblib

from typing import Optional
from pathlib import Path
from prettytable import PrettyTable
from datetime import datetime
from pprint import pprint

from factorlib.utils.helpers import calc_compounded_returns, offset_datetime
from factorlib.utils.datetime_maps import timedelta_intervals


class Statistics:
    def __init__(self, name: str = None,
                 interval: str = None,
                 factors: pd.DataFrame = None,
                 portfolio_returns: pd.Series = None,
                 expected_returns: pd.DataFrame = None,
                 true_returns: pd.DataFrame = None,
                 position_weights: pd.DataFrame = None,
                 shap_values: dict[int, shap.Explainer] = None,
                 training_ic: Optional[pd.Series] = None,
                 extra_baselines: [pd.Series] = None,
                 load_path: Optional[Path | str] = None,
                 trial=None, save_dir=None):

        if load_path is not None:
            self.load_path = load_path
        else:
            qs.extend_pandas()
            self.name = name
            self.interval = interval
            self.factors = factors
            self.trial = trial
            self.save_dir = save_dir

            self.portfolio_returns = portfolio_returns
            self.expected_returns = expected_returns
            self.true_returns = true_returns

            self.position_weights = position_weights

            self.training_ic = training_ic
            self.shap_values = shap_values

            self.testing_ic = expected_returns.corrwith(true_returns, method='spearman', axis=1).expanding(1).mean()[
                              10:]

            print('Downloading SPY Baseline...')
            self.spy_baseline = yf.download(tickers='SPY',
                                            start=expected_returns.index[0],
                                            end=offset_datetime(date=expected_returns.index[-1],
                                                                interval=self.interval,
                                                                sign=2))['Adj Close']
            self.spy_baseline = self.spy_baseline.pct_change()
            self.spy_baseline.name = 'spy'

            self.spy_baseline, self.portfolio_returns = self.spy_baseline.align(self.portfolio_returns,
                                                                                join='left', axis=0)
            self.true_returns, self.expected_returns = self.true_returns.align(self.expected_returns,
                                                                               join='left', axis=0)

            self.extra_baselines = extra_baselines
            self.save(self.save_dir)

    def save(self, save_dir: Optional[Path] = Path('.')):
        with open(save_dir / self.name / f'{self.name}_{self.trial}.stats', 'wb') as f:
            joblib.dump(self, f)

    def load(self):
        with open(self.load_path, 'rb') as f:
            loaded_model = joblib.load(f)
        self.__dict__.update(loaded_model.__dict__)
        return self

    def stats_report(self):
        print()
        print('{:<35s}'.format('FACTORLIB STATS REPORT'))
        print('{:<35s}'.format('=============================='))
        print('{:<30s}'.format('Relative to baseline models:'))

        cum_returns = ['cum. returns']
        sharpe = ['sharpe']
        sortino = ['sortino']
        cagr = ['cagr']
        avg_rtn = ['avg rtrns']
        max_drawdown = ['max drawdown']
        volatility = ['volatility']
        win_rate = ['win rate']

        print('Information coefficient (spearman): ' + str(self.spearman_rank()))
        all_returns = [self.portfolio_returns, self.spy_baseline]

        column_headers = [x.name for x in all_returns]
        column_headers.insert(0, 'metric:')
        statsTable = PrettyTable(column_headers)

        if self.extra_baselines is not None:
            all_returns.extend(self.extra_baselines)
        for returns in all_returns:
            cum_returns.append(str(round((calc_compounded_returns(returns) * 100).iloc[-1], 2)) + '%')
            sharpe.append(round(returns.sharpe(periods=timedelta_intervals[self.interval]), 2))
            sortino.append(round(returns.sortino(periods=timedelta_intervals[self.interval]), 2))
            cagr.append(str(round(returns.cagr() * 100, 2)) + '%')
            avg_rtn.append(str(round(returns.avg_return() * 100, 2)) + '%')
            max_drawdown.append(str(round(returns.max_drawdown() * 100, 2)) + '%')
            volatility.append(str(round(returns.volatility(periods=timedelta_intervals[self.interval]) * 100, 2)) + '%')
            win_rate.append(str(round(returns.win_rate() * 100, 2)) + '%')

        statsTable.add_row(cum_returns)
        statsTable.add_row(sharpe)
        statsTable.add_row(sortino)
        statsTable.add_row(cagr)
        statsTable.add_row(avg_rtn)
        statsTable.add_row(max_drawdown)
        statsTable.add_row(volatility)
        statsTable.add_row(win_rate)
        print(statsTable)

    def get_factors(self):
        pprint(self.factors.columns.tolist())

    def snapshot(self):
        qs.plots.snapshot(self.portfolio_returns, benchmark='SPY',
                          periods_per_year=timedelta_intervals[self.interval])

    def beeswarm_shaps(self, period: 0 | 1 | 2):
        shap.plots.beeswarm(self.shap_values[period], max_display=len(self.factors.columns))

    def waterfall_shaps(self, period: 0 | 1 | 2):
        shap.plots.beeswarm(self.shap_values[period], max_display=len(self.factors.columns))

    def spearman_rank(self):
        spearman_ranks = self.true_returns.corrwith(self.expected_returns, method='spearman', axis=1)
        spearman_rank = spearman_ranks.mean()
        return spearman_rank
