import os
import pandas as pd
import numpy as np
import joblib

from datetime import datetime
from pathlib import Path
from factorlib.factor import Factor
from factorlib.factor_model import FactorModel
from factorlib.types import PortOptOptions, ModelType
from factorlib.utils.system import get_raw_data_dir, get_experiments_dir


INTERVAL = 'B'
DATA_FOLDER = get_raw_data_dir()
returns = pd.read_parquet(DATA_FOLDER / 'sp500_returns.parquet.brotli')
tickers = np.unique(returns['ticker']).tolist()

factor_model = FactorModel(name='test_00', tickers=tickers, interval=INTERVAL, model_type=ModelType.lightgbm)

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

with open('data/raw/sp500_candidates.pkl', 'rb') as p:
    candidates = joblib.load(p)

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
stats = factor_model.wfo(returns,
                         train_interval=pd.DateOffset(years=5), train_freq='M', anchored=False,
                         start_date=datetime(2017, 1, 5), end_date=datetime(2022, 12, 20),
                         candidates=candidates,
                         save_dir=get_experiments_dir(), **kwargs,
                         port_opt=PortOptOptions.MeanVariance)
print('hello world')
