import pandas as pd
import numpy as np
import os
import joblib as joblib

from tqdm.auto import tqdm
from datetime import datetime

from factorlib.utils.system import get_data_dir

tqdm.pandas()

raw_data_dir = get_data_dir() / 'raw'
factor_dir = raw_data_dir / 'PredictorsIndiv'

permnos = pd.read_csv(raw_data_dir / 'permno.csv', index_col=1)
permnos.index.name = 'permno'
permnos.reset_index(inplace=True)
permnos = dict(zip(permnos['permno'], permnos['ticker']))


def convert_open_asset_pricing_date(df: pd.DataFrame):
    df.index = df.index.astype(str)
    years = df.index.str.slice(0,4)
    months = df.index.str.slice(4,6)
    df.index = pd.to_datetime(years + '-' + months)
    return df


def remove_dupes(df: pd.DataFrame):
    df = df.set_index('ticker', append=True)
    df.index.names = ['date', 'ticker']
    df = df[~df.index.duplicated('first')]
    return df


with open(raw_data_dir / 'sp500_candidates.pkl', 'rb') as p:
    candidates = joblib.load(p)

all_candidates = []
for date, candidate in candidates.items():
    all_candidates.extend(candidate)
all_candidates = np.unique(all_candidates).tolist()

batch_size = 20
batch_num = 0
batch_count = 0
for dir, dirnames, filenames in os.walk(factor_dir):
    batch = pd.DataFrame()
    for file in tqdm(filenames):
        factor_data = pd.read_csv(os.path.join(factor_dir, file), index_col=1)
        factor_data = convert_open_asset_pricing_date(factor_data)
        if batch_count == batch_size:
            batch = batch.loc[:, batch.isnull().sum() / len(batch) < .3]
            batch.to_csv(get_data_dir() / 'factors' / 'open_asset_pricing' / 'predictors' /
                         f'oap_predictors_{str(batch_num)}.csv')
            batch_num += 1
            batch = pd.DataFrame()

        factor_data['ticker'] = factor_data['permno'].map(permnos)
        if len(factor_data['ticker'].unique()) < 500:
            continue
        factor_data.dropna(subset=['ticker'], inplace=True)
        factor_data.drop(columns=['permno'], inplace=True)

        factor_data = remove_dupes(factor_data)
        factor_data = factor_data.reset_index(level='ticker')
        factor_data = factor_data[factor_data['ticker'].isin(all_candidates)]
        factor_data = factor_data[factor_data.index.get_level_values(level='date') > datetime(2005, 1, 1)]
        factor_data.set_index('ticker', append=True, inplace=True)

        factor_data = factor_data.reset_index(level='ticker').groupby('ticker').resample('B').ffill().drop(
            columns=['ticker'])

        if batch.empty:
            batch = factor_data
        else:
            batch = batch.join(factor_data, how='outer')
        del factor_data
        batch_count += 1
