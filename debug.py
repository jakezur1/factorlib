import pandas as pd
import pickle as pkl

from datetime import datetime
from pathlib import Path

from factorlib.factor_model import FactorModel
from factorlib.types import PortOptOptions


with open('data/raw/sp500_candidates.pkl', 'rb') as p:
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
INTERVAL = 'B'
DATA_FOLDER = Path('./data')
returns = pd.read_parquet(DATA_FOLDER / 'sp500_returns.parquet.brotli')

factor_model = FactorModel(load_path=Path('experiments/test_00/test_00.alpha')).load()
factor_model.wfo(returns,
                 train_interval=pd.DateOffset(years=5), train_freq='M', anchored=False,
                 start_date=datetime(2017, 11, 5), end_date=datetime(2022, 12, 20),
                 candidates=candidates,
                 save_dir=Path('./experiments'), **kwargs,
                 port_opt=PortOptOptions.InverseVariance)

