import pandas as pd
import joblib

from datetime import datetime
from pathlib import Path

from factorlib.factor_model import FactorModel
from factorlib.types import PortOptOptions
from factorlib.utils.system import get_raw_data_dir, get_experiments_dir


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
INTERVAL = 'B'
returns = pd.read_parquet(get_raw_data_dir() / 'sp500_returns.parquet.brotli')

factor_model = FactorModel(load_path=Path('experiments/base_model/base_model_0.alpha')).load()
stats = factor_model.wfo(returns,
                         train_interval=pd.DateOffset(years=5), train_freq='M', anchored=False,
                         start_date=datetime(2017, 11, 5), end_date=datetime(2022, 12, 20),
                         candidates=candidates,
                         save_dir=get_experiments_dir(), **kwargs,
                         port_opt=PortOptOptions.MeanVariance)
stats.save(save_dir=get_experiments_dir())
print('hello')
