import os
import pandas as pd
import numpy as np
import pickle as pkl

from datetime import datetime
from pathlib import Path
from factorlib.factor import Factor
from factorlib.factor_model import FactorModel
from factorlib.utils.helpers import _get_nearest_month_end, _set_index_names_adaptive, _get_nearest_month_begin, \
    get_subset_by_date_bounds

#
# returns = pd.read_csv('./data/training_returns.csv', index_col=0)
# returns.index = pd.to_datetime(returns.index)
#
# returns_stacked = returns.stack()
# returns_stacked.name = 'price_data'
# returns_stacked.index.names = _set_index_names_adaptive(returns_stacked)
#
# fff_daily = pd.read_csv('./data/fff_daily.csv', index_col=6)
# fff_daily.index.name = 'date'
# spy_data_daily = pd.read_csv('./data/spy_data_daily.csv', index_col=0)
# spy_data_daily.index.name = 'date'
#
# tickers = ['A', 'MSFT', 'ZTS']
# interval = 'B'
# factor = Factor(name='returns', interval=interval, data=returns_stacked, tickers=tickers)
# factor1 = Factor(name='fff', interval=interval, data=fff_daily, tickers=tickers, general_factor=True)
# factor2 = Factor(name='spy', interval=interval, data=spy_data_daily, tickers=tickers, price_data=True)
#
# factor_model = FactorModel(tickers=tickers, interval=interval)
# factor_model.add_factor(factor)
# factor_model.add_factor(factor1)
# factor_model.add_factor(factor2)
#
# data = factor_model.factors
#
# anchored = True
# training_interval = pd.DateOffset(years=5)
# train_freq = 'M'
#
# # align factors and returns
# y, data = returns_stacked.align(data, join='left', axis=0)
#
# # get monthly groups of factors
# monthly_groups = data.groupby(pd.Grouper(level='date', freq='M'))
#
# # get the first 5 years of training data, so we don't iterate over them
# train_start = data.index.get_level_values('date')[0]
# train_start = _get_nearest_month_end(train_start)
# train_end = data.index.get_level_values('date')[0] + pd.DateOffset(years=5)
# train_end = _get_nearest_month_end(train_end)
#
# # get the monthly groups again after initial training period to iterate
# iterate_data = get_subset_by_date_bounds(data, start_date=train_end)
# monthly_groups = iterate_data.groupby(pd.Grouper(level='date', freq=train_freq))
#
# for i, (month, group) in enumerate(monthly_groups):
#     if i == (len(monthly_groups) - 1):
#         continue
#
#     X_train = get_subset_by_date_bounds(data, train_start, train_end)
#
#     pred_start = _get_nearest_month_begin(month)
#     pred_end = _get_nearest_month_end(pred_start + pd.DateOffset(months=1))
#
#     X_pred = get_subset_by_date_bounds(data, pred_start, pred_end)
#
# print('hello')
#

INTERVAL = 'B'
DATA_FOLDER = Path('./data')
returns = pd.read_parquet(DATA_FOLDER / 'sp500_returns.parquet.brotli')
tickers = np.unique(returns['ticker']).tolist()

factor_model = FactorModel(name='test_00', tickers=tickers, interval=INTERVAL, model_type='xgb')

factor_data = pd.read_parquet(DATA_FOLDER / 'factor_return.parquet.brotli')
factor = Factor(name='ret', interval=INTERVAL, data=factor_data, tickers=tickers)
factor_model.add_factor(factor)
del factor_data, factor

factor1_data = pd.read_parquet(DATA_FOLDER / 'factor_pca_return.parquet.brotli')
factor1 = Factor(name='pca_ret', interval=INTERVAL, data=factor1_data, tickers=tickers)
factor_model.add_factor(factor1, replace=True)
del factor1_data, factor1

factor2_data = pd.read_parquet(DATA_FOLDER / 'factor_fund_ratio_div_price.parquet.brotli')
factor2 = Factor(name='fund_ratio', interval=INTERVAL, data=factor2_data, tickers=tickers)
factor_model.add_factor(factor2, replace=True)
del factor2_data, factor2

with open('./data/sp500_candidates.pkl', 'rb') as p:
    candidates = pkl.load(p)

kwargs = {
    'random_state': 42,
    'n_jobs': -1,
    'boosting': 'gbdt',
    'objective': 'regression',
    'verbose': -1,
    'max_depth': -1,
    'learning_rate': 0.15,
    'num_leaves': 15,
    'feature_fraction': 0.85,
    'min_gain_to_split': 0.02,
    'min_data_in_leaf_opts': 60,
    'metric': 'mse',
    'num_threads': 8,
    'lambda_l2': 0.01,
    'extra_trees': False,
    'num_boost_round': 1000
}
factor_model.wfo(returns,
                 train_interval=pd.DateOffset(years=5), train_freq='M', anchored=False,
                 start_date=datetime(2017, 11, 5), end_date=datetime(2022, 12, 20),
                 candidates=candidates, **kwargs)
