import shap
import sys
import threading
import time
import numpy as np
import pandas as pd
import quantstats as qs
import pickle as pkl

from typing import Optional
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

from sklearn.ensemble import *
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import lightgbm as lgb

from factorlib.factor import Factor
from factorlib.utils.types import ModelType, FactorlibUserWarning
from factorlib.utils.helpers import _set_index_names_adaptive, shift_by_time_step, get_subset_by_date_bounds, \
    _get_nearest_month_end, _get_nearest_month_begin, clean_data
from factorlib.utils.datetime_maps import timedelta_intervals
from factorlib.utils.system import show_processing_animation, print_dynamic_line, print_warning, _spinner_animation

shap.initjs()


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
        if not load_path:
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
            self.earliest_start = None
            self.latest_end = None
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
        if self.model_type == 'lgbm':
            params = dict(kwargs)
            params['objective'] = 'regression'
            params['n_jobs'] = -1
            params['random_state'] = 42
            if 'num_boost_round' in params:
                del params['num_boost_round']
            train_dataset = lgb.Dataset(X, y)
            self.model = lgb.train(params=params,
                                   train_set=train_dataset,
                                   num_boost_round=kwargs['num_boost_round'])
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
            pred_time: str = 't+1',
            train_freq: str = None,
            candidates: dict = None,
            calc_training_ic: bool = False,
            save_dir: Path = None, **kwargs):

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

        all_valid_candidates = []
        all_candidates = []
        if candidates is not None:
            # need to ensure we have candidates for every day
            candidates = pd.Series(candidates)
            candidates.index = pd.to_datetime(candidates.index)
            candidates = get_subset_by_date_bounds(candidates, start_date, end_date)

            # bfill after ffill to account for the first date, literally only one-edge case where this doesn't work
            candidates = candidates.resample('B').ffill().fillna(method='bfill')

            # need to get rid of candidates that aren't in returns
            tickers_with_returns = returns.index.get_level_values('ticker').unique().tolist()
            for name, item in candidates.items():
                valid_tickers = [candidate for candidate in item if candidate in tickers_with_returns]
                all_valid_candidates.extend(valid_tickers)
                all_candidates.extend(item)
                candidates[name] = valid_tickers

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

        start_date = _get_nearest_month_begin(start_date)
        end_date = _get_nearest_month_end(end_date)

        print()
        print_dynamic_line()
        print('Starting Walk-Forward Optimization from', start_date, 'to',
              end_date, 'with a', train_interval.years, 'year training interval')

        self.model = self._get_model(self.model_type)

        returns = shift_by_time_step(pred_time, returns)
        returns = get_subset_by_date_bounds(returns, start_date, end_date)
        grouped_returns = returns.groupby(level='date')
        for last_day, last_days_returns in grouped_returns:
            pass
        del grouped_returns

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

            self.fit(X_train, y_train, **kwargs)
            del y_train

            # calculate daily training spearman correlations
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
                    returns_for_spearman.loc[
                        ~returns_for_spearman.index.isin(daily_spearman_correlations.index)]

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
                train_start = _get_nearest_month_begin(train_start + pd.DateOffset(months=1))
            train_end = pred_end

        expected_returns = expected_returns[expected_returns.index.get_level_values('ticker').notna()]
        expected_returns = expected_returns.unstack(level='ticker')
        expected_returns.columns = expected_returns.columns.droplevel(
            [name for name in expected_returns.columns.names if name != 'ticker'])

        if candidates is not None:
            candidates = candidates.loc[(candidates.index >= pd.to_datetime(expected_returns.index[0])) &
                                        (candidates.index <= pd.to_datetime(expected_returns.index[-1]))]
            expected_returns, candidates = expected_returns.align(candidates, join='left', axis=0)
            candidates.fillna(inplace=True, method='ffill')

            nan_mask = pd.DataFrame(np.nan, columns=self.tickers, index=expected_returns.index)
            for date, sp500_list in candidates.items():
                sp500_list = [ticker for ticker in sp500_list if ticker in self.tickers]
                if date in nan_mask.index:
                    nan_mask.loc[date][sp500_list] = 1

            nan_mask, expected_returns = nan_mask.align(expected_returns, axis=1, join='left')
            expected_returns = expected_returns.mul(nan_mask)

        positions = expected_returns.apply(self._get_positions, axis=1,
                                           k_pct=k_pct, long_pct=long_pct,
                                           long_only=long_only, short_only=short_only)
        positions_start = positions.index[0]
        positions_end = positions.index[-1]

        returns = shift_by_time_step(pred_time, returns, backwards=True)
        returns = pd.concat([returns, last_days_returns])
        mask = returns.index.duplicated(keep='first')
        returns = returns[~mask]

        returns = get_subset_by_date_bounds(returns, positions_start, positions_end)

        returns = returns.unstack(level='ticker')
        returns.columns = returns.columns.droplevel(
            [name for name in expected_returns.columns.names if name != 'ticker'])
        returns, positions = returns.align(positions, join='left', axis=0)

        returns_per_stock = returns.mul(positions.shift(1))  # you have to shift positions by 1 day
        portfolio_returns = returns_per_stock.sum(axis=1)
        portfolio_returns.name = 'factors'

        qs.plots.snapshot(portfolio_returns, benchmark='SPY',
                          periods_per_year=timedelta_intervals[self.interval])

        if save_dir is not None:
            self.save(save_dir)

    def save(self, save_dir: Optional[Path] = Path('.')):
        with open(save_dir / self.name / f'{self.name}.alpha', 'wb') as f:
            pkl.dump(self, f)

    def load(self):
        with open(self.load_path, 'rb') as f:
            loaded_model = pkl.load(f)
        self.__dict__.update(loaded_model.__dict__)
        return self

    def _get_positions(self, row, k_pct: float = 0.2,
                       long_pct: float = 0.5,
                       long_only: bool = False,
                       short_only: bool = False):
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
            positions[i] = round((1 / k) * long_pct, 8)
        for i in bottomk:
            positions[i] = round((-1 / k) * (1 - long_pct), 8)

        return pd.Series(positions, index=self.tickers)

    def _get_model(self, model_type: str, **kwargs):
        if model_type == 'hgb':
            self.model = HistGradientBoostingRegressor(**kwargs)
        elif model_type == 'gbr':
            self.model = GradientBoostingRegressor(**kwargs)
        elif model_type == 'adb':
            self.model = AdaBoostRegressor(**kwargs)
        elif model_type == 'rf':
            self.model = RandomForestRegressor(**kwargs)
        elif model_type == 'xgb':
            self.model = XGBRegressor(**kwargs)
        # elif model_type == 'catboost':
        #     self.model = CatBoostRegressor(**kwargs)
        elif model_type == 'lgbm':
            self.model = None
        else:
            print(message='The model you chose (model_type) has not been implemented yet. Please submit a request to '
                          'the factorlib github repository to expedite the implementation of this model. Defaulting '
                          'to LightGBM.', category=FactorlibUserWarning.ParameterOverride)
            self.model = None

        return self.model
