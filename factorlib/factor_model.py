import shap
import numpy as np
import pandas as pd
import quantstats as qs
import joblib
import os

from typing import Optional
from datetime import datetime
from tqdm.auto import tqdm
from pathlib import Path
from time import perf_counter_ns

from sklearn.ensemble import *
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from xgboost import XGBRegressor
import lightgbm as lgb

from factorlib.factor import Factor
from factorlib.types import ModelType, FactorlibUserWarning, PortOptOptions
from factorlib.utils.helpers import _set_index_names_adaptive, shift_by_time_step, get_subset_by_date_bounds, \
    _get_nearest_month_end, _get_nearest_month_begin, clean_data, offset_datetime
from factorlib.utils.datetime_maps import timedelta_intervals
from factorlib.utils.system import show_processing_animation, print_dynamic_line, print_warning, _spinner_animation

shap.initjs()
tqdm.pandas()


class FactorModel:
    def __init__(self, name: Optional[str] = None,
                 tickers: Optional[list[str]] = None,
                 interval: str = 'B',
                 model_type: Optional[ModelType] = None,
                 load_path: Optional[Path | str] = None):
        """
        This class represents a factor for the FactorModel class. It is responsible for formatting user data, and
        transforming it for use in a `FactorModel`.
        """

        if load_path is None:
            assert name is not None, '`name` cannot be None. This will be the name of the file that your model is saved as.'
            assert tickers is not None, '`tickers` cannot be None. If you are using candidates (recommended), you still ' \
                                        'need to pass a list of tickers for which you have returns data. These are ' \
                                        'the tickers that the model will learn off of.'
            if model_type is None:
                print_warning(message='model_type = None. Defaulting model_type to LGBMRegressor '
                                      '(`lgbm` option) from the LightGBM Library.',
                              category=FactorlibUserWarning.ParameterOverride)
                model_type = 'lgbm'

            self.name = name
            self.tickers = tickers
            self.interval = interval
            self.factors = pd.DataFrame()
            self.model_type = model_type
            self.model = None
            self.prev_model = None
            self.earliest_start = None
            self.latest_end = None
            self.trial = 0
        else:
            self.load_path = load_path

    @show_processing_animation(message_func=lambda self, *args, **kwargs: f'Adding {args[0].name} to {self.name}',
                               animation=_spinner_animation)
    def add_factor(self, factor: Factor, replace=False):
        if factor.interval is not self.interval:
            factor.data = factor.data.reset_index(level='ticker').groupby(by='ticker').resample(
                self.interval).ffill().reset_index(level='ticker', drop=True).set_index('ticker', append=True)

        l_suffix = '_no_way_someone_accidentally_has_a_factor_with_this_column_name'
        if self.factors.empty:
            self.factors = factor.data
        else:
            self.factors = self.factors.join(factor.data, how='outer', lsuffix=l_suffix)

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

        columns_to_replace = [column for column in self.factors.columns if l_suffix in column]

        if replace:
            self.factors.drop(columns=columns_to_replace, inplace=True)
        else:
            assert (len(columns_to_replace) == 0), f'You have added a factor named: \'{factor.name}\' with ' \
                                                   f'duplicate features that are already in your model. This is ' \
                                                   f'either a mistake, or replace the column by setting ' \
                                                   f'add_factor(factor, replace=True).'
        del factor

    def fit(self, X, y, **kwargs):
        if self.model_type == ModelType.lightgbm or self.model_type == 'lgbm':
            params = dict(kwargs)
            params['objective'] = 'regression'
            params['n_jobs'] = -1
            params['random_state'] = 42
            if 'num_boost_round' in params:
                del params['num_boost_round']
            train_dataset = lgb.Dataset(X, y)
            if self.prev_model is not None:
                self.model = lgb.train(params=params,
                                       train_set=train_dataset,
                                       num_boost_round=kwargs['num_boost_round'], init_model=self.prev_model)
            else:
                self.model = lgb.train(params=params,
                                       train_set=train_dataset,
                                       num_boost_round=kwargs['num_boost_round'])
            self.prev_model = self.model
        else:
            self.model.fit(X, y)

    def predict(self, factors: pd.DataFrame):
        return self.model.predict(factors)

    def wfo(self, returns: pd.DataFrame,
            train_interval: pd.DateOffset,
            start_date: datetime,
            end_date: datetime,
            anchored: bool = True,
            k_pct: float = 0.2,
            long_pct: float = 0.5,
            long_only: bool = False,
            short_only: bool = False,
            port_opt: PortOptOptions = None,
            pred_time: str = 't+1',
            train_freq: str = None,
            candidates: dict = None,
            calc_training_ic: bool = False,
            save_dir: Path = Path('.'), **kwargs):

        print()
        print_dynamic_line()
        print(f'Starting Walk-Forward Optimization (trail {self.trial}) from', start_date, 'to',
              end_date, 'with a', train_interval.years, 'year training interval')
        self.trial += 1
        self.prev_model = None

        assert (self.interval == 'B' or self.interval == 'D' or self.interval == 'M'), \
            'Walk forward optimization currently only supports daily, business daily, and monthly data intervals.'

        if train_freq is not None:
            print_warning(message='The train_freq parameter does not have stable implementation yet. '
                                  'Defaulting to monthly (\'M\') training.',
                          category=FactorlibUserWarning.NotImplemented)
            possible_intervals = list(timedelta_intervals)
            i = 0
            if timedelta_intervals[train_freq] > timedelta_intervals[self.interval]:
                while (timedelta_intervals[train_freq] > timedelta_intervals[self.interval] and
                       train_freq not in ['M', 'Q', 'Y']):
                    train_freq = possible_intervals[i]
                    i += 1

                print_warning(message=f'The train_freq is a shorter interval than the model interval. '
                                      f'Defaulting to {train_freq} training.',
                              category=FactorlibUserWarning.AmbiguousInterval)

        assert (not (long_only and short_only)), 'long_only and short_only cannot both be True.'

        assert (start_date > self.earliest_start), 'start_date must be after earliest start date of your factors.'

        assert (end_date < self.latest_end), 'end_date must be before latest end date of your factors.'

        if 'ticker' not in returns.index.names:
            if 'ticker' in returns.columns:
                returns.set_index('ticker', append=True, inplace=True)
            else:
                returns = returns.stack()
                returns.name = 'returns'
            returns.index.names = _set_index_names_adaptive(returns)

        returns.sort_index(inplace=True)

        assert (returns.index.get_level_values('date')[0] <= start_date), 'You must provide returns for the ' \
                                                                          'full interval that you would like ' \
                                                                          'to perform wfo(...). Increase your ' \
                                                                          'start_date, or provide the correct returns.'
        assert (returns.index.get_level_values('date')[-1] >= end_date), 'You must provide returns for the ' \
                                                                         'full interval that you would like ' \
                                                                         'to perform wfo(...). Decrease your ' \
                                                                         'end_date, or provide the correct returns.'

        start_date = _get_nearest_month_begin(start_date)
        end_date = _get_nearest_month_end(end_date)

        self.model = self._get_model(self.model_type)

        returns = shift_by_time_step(pred_time, returns)
        returns = get_subset_by_date_bounds(returns, start_date, end_date)
        # grouped_returns = returns.groupby(level='date')
        # for last_day, last_days_returns in grouped_returns:
        #     pass
        # del grouped_returns

        self.factors.sort_index(inplace=True)
        factors = get_subset_by_date_bounds(self.factors, start_date, end_date)

        # align factors and returns
        y = returns
        y, factors = y.align(factors, join='left', axis=0)

        # get the first 5 years of training data, so we don't iterate over them
        train_start = start_date
        train_end = train_start + pd.DateOffset(years=5)
        # round train_end to end of the nearest month so that we can be uniform for iteration going forward
        train_end = _get_nearest_month_end(train_end)

        # get the monthly groups again after initial training period to iterate
        iterate_data = get_subset_by_date_bounds(factors, start_date=train_end, end_date=end_date)
        monthly_groups = iterate_data.groupby(pd.Grouper(level='date', freq=train_freq))
        del iterate_data

        shap_values = {}
        training_spearman = None
        expected_returns = None
        for index, (month, group) in enumerate(tqdm(monthly_groups)):
            if index == len(monthly_groups) - 1:
                continue

            X_train = get_subset_by_date_bounds(factors, train_start, train_end)
            y_train = get_subset_by_date_bounds(y, train_start, train_end)

            y_train, X_train = y_train.align(X_train, join="left", axis=0)
            X_train, y_train = clean_data(X_train, y_train)

            start = perf_counter_ns()
            self.fit(X_train, y_train, **kwargs)
            end = perf_counter_ns()
            del y_train

            # calculate daily training spearman correlations
            training_predictions: pd.DataFrame = None
            if calc_training_ic:
                training_predictions = self.predict(X_train)

                training_predictions = pd.DataFrame(training_predictions, index=X_train.index)

            # Get shap values here, so we can delete X_train the second we don't need it
            # get beginning middle and end shap values and add them to dictionary
            # we use len(monthly_groups) - 2 because we skip the last interval because monthly_groups is end-inclusive
            if index == 0 or index == (len(monthly_groups) - 2) or index == len(monthly_groups) / 2:
                explainer = shap.Explainer(self.model)
                shap_values[index] = explainer(X_train)

            del X_train

            # continue with spearman calculation
            if calc_training_ic:
                returns_for_spearman = get_subset_by_date_bounds(returns, train_start, train_end)

                # only calculate spearman rank for days that have not already been calculated
                daily_spearman_correlations = None
                if daily_spearman_correlations is not None:
                    returns_for_spearman.loc[~returns_for_spearman.index.isin(daily_spearman_correlations.index)]

                # align the training predictions with returns for spearman
                returns_for_spearman, training_predictions = returns_for_spearman.align(training_predictions,
                                                                                        join='left',
                                                                                        axis=0)
                training_predictions.columns = returns_for_spearman.columns

                daily_spearman_correlations = training_predictions.groupby(level='date').corrwith(returns_for_spearman,
                                                                                                  method='spearman')
                del returns_for_spearman

                training_spearman = pd.concat([training_spearman, daily_spearman_correlations])
                del daily_spearman_correlations

            pred_start = _get_nearest_month_begin(month)
            pred_end = _get_nearest_month_end(pred_start + pd.DateOffset(months=1))

            X_pred = get_subset_by_date_bounds(factors, pred_start, pred_end)

            y_pred = self.predict(X_pred)
            y_pred = pd.DataFrame(y_pred, index=X_pred.index)
            del X_pred

            expected_returns = pd.concat([expected_returns, y_pred], axis=0)
            del y_pred

            if not anchored:
                if index == 0:
                    train_start = _get_nearest_month_begin(
                        train_start + pd.DateOffset(months=1) + pd.DateOffset(years=4))
                else:
                    train_start = _get_nearest_month_begin(train_start + pd.DateOffset(months=1))
            train_end = pred_end

        expected_returns = expected_returns[expected_returns.index.get_level_values('ticker').notna()]
        expected_returns = expected_returns.unstack(level='ticker')
        expected_returns.columns = expected_returns.columns.droplevel(
            [name for name in expected_returns.columns.names if name != 'ticker'])

        all_valid_candidates = []
        all_candidates = []
        if candidates is not None:
            # need to ensure we have candidates for every day
            candidates = pd.Series(candidates)
            candidates.index = pd.to_datetime(candidates.index)
            candidates = get_subset_by_date_bounds(candidates, start_date, end_date)

            # bfill after ffill to account for the first date, literally only one-edge case where this doesn't work
            candidates = candidates.resample('B').ffill().fillna(method='bfill')

            candidates = candidates.loc[(candidates.index >= pd.to_datetime(expected_returns.index[0])) &
                                        (candidates.index <= pd.to_datetime(expected_returns.index[-1]))]
            expected_returns, candidates = expected_returns.align(candidates, join='left', axis=0)
            candidates.fillna(method='ffill', inplace=True)

            # need to get rid of candidates that aren't in returns
            tickers_with_returns = returns.index.get_level_values('ticker').unique().tolist()
            nan_mask = pd.DataFrame(np.nan, columns=self.tickers, index=expected_returns.index)
            for date, sp500_list in candidates.items():
                all_candidates.extend(sp500_list)
                sp500_list = [ticker for ticker in sp500_list if ticker in tickers_with_returns]
                if date in nan_mask.index:
                    nan_mask.loc[date][sp500_list] = 1
                all_valid_candidates.extend(sp500_list)

            nan_mask, expected_returns = nan_mask.align(expected_returns, axis=1, join='left')
            expected_returns = expected_returns.mul(nan_mask)

            all_valid_candidates = np.unique(all_valid_candidates).tolist()
            all_candidates = np.unique(all_candidates).tolist()
            all_returns_present = all_valid_candidates == all_candidates
            if not all_returns_present:
                print_warning(
                    message='You have passed a dict of candidates, but you do not have returns for all possible '
                            'candidates in a given year. We have filtered your candidates down to the provided '
                            f'returns, which includes {len(all_valid_candidates)} / {len(all_candidates)}. Depending '
                            f'on the number of tickers for which you don\'t have returns, you may want to stop wfo() '
                            f'and obtain the correct set of returns. Continuing...',
                    category=FactorlibUserWarning.MissingData)

            self.tickers = np.unique(all_valid_candidates).tolist()
            del all_candidates
            del all_valid_candidates

        returns = shift_by_time_step(pred_time, returns, backwards=True)
        # returns = pd.concat([returns, last_days_returns])
        # mask = returns.index.duplicated(keep='first')
        # returns = returns[~mask]

        # get 3 months prior to start of expected returns
        returns_start_date = offset_datetime(expected_returns.index.get_level_values('date')[0],
                                             interval='M',
                                             sign=-3)
        returns = get_subset_by_date_bounds(returns,
                                            start_date=returns_start_date,
                                            end_date=expected_returns.index.get_level_values('date')[-1])

        returns = returns.unstack(level='ticker')
        returns.columns = returns.columns.droplevel(
            [name for name in expected_returns.columns.names if name != 'ticker'])
        expected_returns = expected_returns[self.tickers]
        returns = returns[self.tickers]

        print('\nCalculating Position Weights:')
        positions = expected_returns.progress_apply(self._get_positions, axis=1,
                                                    k_pct=k_pct, long_pct=long_pct,
                                                    long_only=long_only, short_only=short_only,
                                                    returns=returns,
                                                    port_opt=port_opt)

        positions, returns = positions.align(returns, join='left', axis=0)

        returns_per_stock = returns.mul(positions.shift(1))  # you have to shift positions by 1 day
        portfolio_returns = returns_per_stock.sum(axis=1)
        portfolio_returns.name = 'factors'

        qs.plots.snapshot(portfolio_returns, benchmark='SPY',
                          periods_per_year=timedelta_intervals[self.interval])

        if save_dir is not None:
            self.save(save_dir)

        from factorlib.stats import Statistics
        return Statistics(name=self.name, interval=self.interval, factors=factors,
                          expected_returns=expected_returns, true_returns=returns,
                          portfolio_returns=portfolio_returns, position_weights=positions,
                          training_ic=training_spearman, shap_values=shap_values,
                          trial=self.trial, save_dir=save_dir)

    def save(self, save_dir: Optional[Path] = Path('.')):
        save_path = save_dir / self.name

        if not save_path.exists():
            save_path.mkdir(exist_ok=True)

        with open(save_path / f'{self.name}_{self.trial}.alpha', 'wb') as f:
            joblib.dump(self, f)

    def load(self):
        with open(self.load_path, 'rb') as f:
            loaded_model = joblib.load(f)
        self.__dict__.update(loaded_model.__dict__)
        return self

    def rename(self, new_name: str):
        self.name = new_name

    def _get_positions(self, row, k_pct: float = 0.2,
                       long_pct: float = 0.5,
                       long_only: bool = False,
                       short_only: bool = False,
                       returns: pd.DataFrame = None,
                       port_opt: PortOptOptions = None):
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

        if port_opt is None:
            if long_only:
                long_pct = 1.0
            elif short_only:
                long_pct = 0.0

            for i in topk:
                positions[i] = round((1 / k) * long_pct, 8)
            for i in bottomk:
                positions[i] = round((-1 / k) * (1 - long_pct), 8)
        elif port_opt == 'mvp':
            positions = self._mvp_option(row, positions, long_pct, returns, topk, bottomk)
        elif port_opt == 'hrp':
            positions = self._hrp_option(row, positions, long_pct, returns, topk, bottomk)
        elif port_opt == 'iv':
            positions = self._iv_option(row, positions, long_pct, returns, topk, bottomk)

        return pd.Series(positions, index=self.tickers)

    def _mvp_option(self, row, positions,
                    long_pct: float = None,
                    returns: pd.DataFrame = None,
                    topk: pd.Series = None,
                    bottomk: pd.Series = None):
        all_indices = pd.concat([topk, bottomk])

        current_date = row.name
        three_months_prior = offset_datetime(current_date, interval='M', sign=-3)
        past_returns = get_subset_by_date_bounds(returns,
                                                 start_date=three_months_prior,
                                                 end_date=current_date).iloc[:-1]

        subset_returns = past_returns.loc[:, all_indices.index]
        cov_matrix = subset_returns.cov()

        # Define the objective function (Portfolio Variance)
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        # Constraints: Sum of long weights = long_pct and sum of short weights = 1 - long_pct
        long_constraint = {'type': 'eq', 'fun': lambda weights: np.sum(weights[:len(topk)]) - long_pct}
        short_constraint = {'type': 'eq', 'fun': lambda weights: np.sum(weights[len(topk):]) + (1 - long_pct)}

        constraints = [long_constraint, short_constraint]

        # Set bounds: long positions for top k stocks, short positions for bottom k stocks
        bounds = [(0, 1) for _ in range(len(topk))] + [(-1, 0) for _ in range(len(bottomk))]
        initial_guess = [1. / len(all_indices) for _ in range(len(all_indices))]

        # Perform the optimization to determine the best weights
        solution = minimize(portfolio_variance, initial_guess, bounds=bounds, constraints=constraints)

        # Extract the optimized weights from the solution and assign them to the positions
        optimized_weights = solution.x
        for i, weight in zip(all_indices, optimized_weights):
            positions[i] = round(weight, 8)

        # Normalize to ensure the absolute sum of weights is 1
        total_abs_sum = np.sum(np.abs(positions))
        positions /= total_abs_sum

        return positions

    def _iv_option(self, row, positions,
                   long_pct: float = None,
                   returns: pd.DataFrame = None,
                   topk: pd.Series = None,
                   bottomk: pd.Series = None):
        all_indices = pd.concat([topk, bottomk])

        current_date = row.name
        three_months_prior = offset_datetime(current_date, interval='M', sign=-3)
        past_returns = get_subset_by_date_bounds(returns,
                                                 start_date=three_months_prior,
                                                 end_date=current_date).iloc[:-1]
        past_returns = past_returns[all_indices.index]
        variance = past_returns.var()
        inverse_variance = 1 / variance

        topk_weights = inverse_variance[topk.index]
        bottomk_weights = inverse_variance[bottomk.index]

        # Normalize weights for the given long_pct
        normalized_topk_weights = topk_weights / topk_weights.sum() * long_pct
        normalized_bottomk_weights = bottomk_weights / bottomk_weights.sum() * (1 - long_pct)

        for ticker, rank in topk.items():
            positions[rank] = normalized_topk_weights[ticker]
        for ticker, rank in bottomk.items():
            positions[rank] = -normalized_bottomk_weights[ticker]

        # Normalize to ensure the absolute sum of weights is 1
        total_abs_sum = np.sum(np.abs(positions))
        positions /= total_abs_sum

        return positions

    def _hrp_option(self, row, positions,
                    long_pct: float = None,
                    returns: pd.DataFrame = None,
                    topk: pd.Series = None,
                    bottomk: pd.Series = None):
        all_indices = pd.concat([topk, bottomk])

        current_date = row.name
        three_months_prior = offset_datetime(current_date, interval='M', sign=-3)
        past_returns = get_subset_by_date_bounds(returns,
                                                 start_date=three_months_prior,
                                                 end_date=current_date).iloc[:-1]
        hrp_weights = self._calc_hrp(past_returns[all_indices.index])

        # Calculate the sum of absolute long and short weights
        long_sum = sum(hrp_weights[ticker] for ticker in topk.index)
        short_sum = sum(hrp_weights[ticker] for ticker in bottomk.index)

        # Adjustment factors based on long_pct
        total_weight = long_sum + abs(short_sum)
        long_adjustment = long_pct * total_weight / long_sum
        short_adjustment = (1 - long_pct) * total_weight / abs(short_sum)

        # Adjust the weights so the absolute values sum to 1, and we're market neutral
        for ticker, rank in topk.items():
            positions[rank] = hrp_weights[ticker] * long_adjustment
        for ticker, rank in bottomk.items():
            positions[rank] = -hrp_weights[ticker] * short_adjustment

        # Normalize to ensure the absolute sum of weights is 1
        total_abs_sum = np.sum(np.abs(positions))
        positions /= total_abs_sum
        return positions

    @staticmethod
    def _get_inverse_var_pf(cov):
        ivp = 1. / np.diag(cov)
        ivp /= ivp.sum()
        return ivp

    def _get_cluster_var(self, cov, c_items):
        cov_ = cov.loc[c_items, c_items]
        w_ = self._get_inverse_var_pf(cov_).reshape(-1, 1)
        c_var = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
        return c_var

    @staticmethod
    def _get_quasi_diag(link):
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()

    def _calc_hrp(self, returns):
        cov = returns.cov()
        corr = returns.corr()
        dist = ((1 - corr) / 2.) ** .5
        link = linkage(pdist(dist), 'single')
        sort_ix = self._get_quasi_diag(link)
        sort_ix = corr.index[sort_ix].tolist()
        hrp = pd.Series(1, index=sort_ix)
        cluster_vars = [self._get_cluster_var(cov, [sort_ix[i]]) for i in range(len(sort_ix))]
        for i in range(0, len(sort_ix) - 1):
            scale = cluster_vars[i] / (cluster_vars[i] + cluster_vars[i + 1])
            hrp[sort_ix[i]] = scale
            hrp[sort_ix[i + 1]] = 1 - scale
            cluster_vars[i + 1] = self._get_cluster_var(cov, [sort_ix[i], sort_ix[i + 1]])
        hrp /= hrp.sum()
        return hrp

    def _get_model(self, model_type: ModelType, **kwargs):
        if model_type == ModelType.hist_gradient_boost or model_type == 'hgb':
            self.model = HistGradientBoostingRegressor(**kwargs)
        elif model_type == ModelType.gradient_boost or model_type == 'gbr':
            self.model = GradientBoostingRegressor(**kwargs)
        elif model_type == ModelType.adaboost or model_type == 'adb':
            self.model = AdaBoostRegressor(**kwargs)
        elif model_type == ModelType.random_forest or model_type == 'rf':
            self.model = RandomForestRegressor(**kwargs)
        elif model_type == ModelType.xgboost or model_type == 'xgb':
            self.model = XGBRegressor(**kwargs)
        # elif model_type == 'catboost':
        #     self.model = CatBoostRegressor(**kwargs)
        elif model_type == ModelType.lightgbm or model_type == 'lgbm':
            self.model = None
        else:
            print_warning(
                message='The model you chose (model_type) has not been implemented yet. Please submit a request to '
                        'the factorlib github repository to expedite the implementation of this model. Defaulting '
                        'to LightGBM.', category=FactorlibUserWarning.NotImplemented)
            self.model = None

        return self.model
