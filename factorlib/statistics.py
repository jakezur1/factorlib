import pandas as pd
import numpy as np
import quantstats as qs
import yfinance as yf
from .factor_model import FactorModel
from scipy import stats
import random
from prettytable import PrettyTable
from .utils import timedelta_intervals, _compsum
import matplotlib.pyplot as plt
import pickle
import shap


class Statistics:
    def __init__(self, portfolio_returns=None, model: FactorModel = None,
                 predicted_returns: pd.DataFrame = None,
                 position_weights: pd.DataFrame = None,
                 training_spearman: pd.Series = None,
                 shap_values: np.array = None,
                 stock_returns: pd.DataFrame = None,
                 extra_baselines: [pd.Series] = None):

        qs.extend_pandas()

        try:
            self.model = model
            self.predicted_returns = predicted_returns
            self.stock_returns = stock_returns
            self.portfolio_returns = portfolio_returns
            self.position_weights = position_weights
            self.training_spearman = training_spearman
            self.shap_values = shap_values
            self.testing_spearman = predicted_returns.corrwith(stock_returns, axis=1).expanding(1).mean()[10:]

            self.portfolio_returns.index = pd.to_datetime(self.portfolio_returns.index).tz_localize(None) \
                .floor('D')
            self.portfolio_returns = portfolio_returns.resample(self.model.interval, convention='end').ffill()
            correct_index = self.portfolio_returns[1:].index

            bh_returns = stock_returns.loc[correct_index]
            self.buy_hold_baseline = bh_returns / len(self.model.tickers)
            self.buy_hold_baseline = self.buy_hold_baseline.sum(axis=1)
            self.buy_hold_baseline = self.buy_hold_baseline.loc[correct_index]
            self.buy_hold_baseline = pd.DataFrame(data={
                'buy_hold': self.buy_hold_baseline
            })

            start_spy = self.model.offset_datetime(self.portfolio_returns.index[0], sign=-1)
            end_spy = self.model.offset_datetime(self.portfolio_returns.index[-1])
            spy_prices = yf.download(tickers='SPY', start=start_spy, end=end_spy)['Adj Close']
            # yfinance will not download the data for the start days if those days are weekends
            spy_prices = spy_prices.resample(self.model.interval, convention='end').ffill()
            spy_returns = spy_prices.pct_change().dropna()
            spy_returns = spy_returns.reindex(pd.date_range(start=start_spy, end=end_spy, freq='D'), fill_value=0.0)
            spy_returns.index = np.array(spy_returns.index, dtype='datetime64[D]')
            spy_returns.index = pd.to_datetime(spy_returns.index).tz_localize(None)

            spy_returns = spy_returns.loc[correct_index]
            spy_returns = pd.DataFrame(data={
                'spy': spy_returns
            })
            self.spy_baseline = spy_returns

            stock_returns = stock_returns.loc[self.portfolio_returns.index[0]:self.portfolio_returns.index[-1]]

            positions = stock_returns[self.model.tickers]
            positions = positions.apply(self._get_random_positions, axis=1,
                                        args=[min(20, len(self.model.tickers) // 2)]).shift(-1)
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

        except Exception as e:
            print('Empty Statistics object created. Load object using load(path: str) method.')

    def to_csv(self, name: str, save_weights: bool = True, save_predictions: bool = True):
        self.portfolio_returns['factors'].to_csv(name + '_factors.csv')
        if save_weights:
            self.position_weights.to_csv(name + '_weights.csv')
        if save_predictions:
            self.predicted_returns.join(self.stock_returns, lsuffix='_predicted', rsuffix='_actual') \
                .to_csv(name + '_predictions.csv')

    def get_full_qs(self):
        qs.reports.full(self.portfolio_returns['factors'], benchmark=self.spy_baseline,
                        periods_per_year=timedelta_intervals[self.model.interval])

    def get_html(self):
        qs.reports.html(self.portfolio_returns['factors'], output='factor_model.html',
                        periods_per_year=timedelta_intervals[self.model.interval])

    def compute_paired_t_test(self):
        paired_t_tests = []
        for returns in self.all_returns:
            paired_t_test = stats.ttest_rel(returns[returns.columns[0]],
                                            self.portfolio_returns['factors'])
            paired_t_tests.append(round(paired_t_test[1], 5))

        return paired_t_tests

    def compute_spearman_rank(self):
        spearman_ranks = self.stock_returns.corrwith(self.predicted_returns, method='spearman', axis=1)
        spearman_rank = spearman_ranks.mean()
        return spearman_rank

    def compute_correlations(self):
        new_df = self.model.factors[self.model.tickers[0]].corr()
        for ticker in self.model.tickers[1:]:
            new_df.add(self.model.factors[ticker].corr())
        corr = new_df / len(self.model.tickers)
        corr = corr.style.background_gradient(axis=None, cmap='YlGn')
        return corr

    def get_factor_names(self):
        for factor in self.model.factors.columns.get_level_values(1).unique():
            print(factor)

    def plot_beeswarm_shaps(self, num_features: int = None):
        if num_features is None:
            num_features = len(self.model.factors.columns.get_level_values(1).unique())
        shap.plots.beeswarm(self.shap_values, num_features)

    def save(self, name, save_plots: bool = True):
        if save_plots:
            plt.savefig(name + '_plots.png')
        with open(f'{name}.p', 'wb') as f:
            pickle.dump(self, f)

    def load(self, path):
        with open(path, 'rb') as f:
            loaded_stats = pickle.load(f)
        self.__dict__.update(loaded_stats.__dict__)

    def print_statistics_report(self):
        print()
        print('{:<35s}'.format('FACTOR MODEL ANALYSIS REPORT'))
        print('{:<35s}'.format('=============================='))
        print('{:<30s}'.format('Relative to baseline models:'))
        column_headers = [x.columns[0] for x in self.all_returns]
        column_headers.insert(0, 'metric:')
        statsTable = PrettyTable(column_headers)

        t_tests = ['paired t-test']
        t_tests.extend(self.compute_paired_t_test())
        cum_returns = ['cum. returns']
        sharpe = ['sharpe']
        sortino = ['sortino']
        cagr = ['cagr']
        avg_rtn = ['avg rtrns']
        max_drawdown = ['max drawdown']
        volatility = ['volatility']
        win_rate = ['win rate']
        print('Spearman correlation: ' + str(self.compute_spearman_rank()))
        self.compute_correlations()
        for returns in self.all_returns:
            cum_returns.append(str(round((_compsum(returns) * 100).iloc[-1].values[0], 2)) + '%')
            sharpe.append(round(returns.sharpe(periods=timedelta_intervals[self.model.interval]).values[0], 3))
            sortino.append(round(returns.sortino(periods=timedelta_intervals[self.model.interval]).values[0], 3))
            cagr.append(str(round(returns.cagr().values[0] * 100, 2)) + '%')
            avg_rtn.append(str(round(returns.avg_return().values[0] * 100, 2)) + '%')
            max_drawdown.append(str(round(returns.max_drawdown().values[0] * 100, 2)) + '%')
            volatility.append(str(round(returns.volatility(periods=timedelta_intervals[self.model.interval])
                                        .values[0] * 100, 2)) + '%')
            win_rate.append(str(round(returns.win_rate().values[0] * 100, 2)) + '%')

        statsTable.add_row(cum_returns)
        statsTable.add_row(sharpe)
        statsTable.add_row(sortino)
        statsTable.add_row(cagr)
        statsTable.add_row(avg_rtn)
        statsTable.add_row(max_drawdown)
        statsTable.add_row(volatility)
        statsTable.add_row(win_rate)
        print(statsTable)
        print()

        fig, axs = plt.subplots(2, 1, figsize=(15, 20))
        fontsize = 18
        line_width = 3

        # plot sum of the position weights (to ensure it is constant throughout trading)
        x = self.position_weights.index
        y = self.position_weights.sum(axis=1)
        axs[0].plot(x, y, linewidth=line_width)
        axs[0].set_title('Position Weights', size=fontsize)
        axs[0].tick_params(axis='both', which='major', labelsize=15)

        # plot RMSE (testing and training)
        # RMSE is unfortunately useless for this data
        # axs[1].plot(self.training_mse.index, np.sqrt(self.training_mse.values * 100), label='Training MSE',
        #             linewidth=line_width)
        # axs[1].plot(self.testing_mse.index, np.sqrt(self.testing_mse.values * 100), label='Testing MSE',
        #             linewidth=line_width)
        # axs[1].set_title('RMSE', size=fontsize)
        # axs[1].legend(loc='upper left', prop={'size': fontsize})
        # axs[1].tick_params(axis='both', which='major', labelsize=15)

        # plot rolling spearman ranks (testing and training)
        axs[1].plot(self.training_spearman.index, self.training_spearman.values, label='Training Spearman',
                    linewidth=line_width)
        axs[1].plot(self.testing_spearman.index, self.testing_spearman.values, label='Testing Spearman',
                    linewidth=line_width)
        axs[1].set_title('Rolling Spearman Rank', size=fontsize)
        axs[1].legend(loc='upper left', prop={'size': fontsize})
        axs[1].tick_params(axis='both', which='major', labelsize=15)

        fig.tight_layout(pad=5.0)

        qs.plots.snapshot(self.portfolio_returns['factors'])

    def _get_random_positions(self, row, k):
        indices = np.argsort(row)  # ascending order
        random.shuffle(indices)

        # calculate long weights, must equal 1
        long_weights = np.random.random(len(indices))
        long_weights /= (np.sum(long_weights))

        bottom_k = indices[:k]
        top_k = indices[-k:]
        positions = [0] * len(row)
        for index, i in enumerate(top_k):
            positions[i] = (1 / k) * long_weights[index]
        for index, i in enumerate(bottom_k):
            positions[i] = round((-1 / k) * (1 - long_weights[index]), 2)

        return pd.Series(positions, index=self.model.tickers)
