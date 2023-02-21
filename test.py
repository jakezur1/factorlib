import os

import pandas as pd
import numpy as np
import yfinance as yf
import getFamaFrenchFactors as gff
from datetime import datetime
from factorlib.factor_model import FactorModel
from factorlib.factor import Factor
from factorlib.transforms import *

interval = 'M'
start = '2002-01-01'
end = '2021-01-01'

print('Reading in Stock Data...')

stocks_data = pd.read_csv('./data/spy_data_daily.csv', index_col=0)
stocks_data.index = pd.to_datetime(stocks_data.index).tz_localize(None) \
            .floor('D')
stocks_data = stocks_data.resample(interval, convention='end').ffill()

print('Reading in Fundamentals Data...')

fundamentals = df = pd.read_csv('./data/fundamentals_spy_only.csv', header=[0, 1], index_col=0)
fundamentals.index = pd.to_datetime(fundamentals.index)

tickers = []
for col in fundamentals.columns:
    tickers.append(col[0])

new_tickers = list(set(tickers) & set(stocks_data.columns))
fundamentals = fundamentals[new_tickers]
fundamentals = Factor(tickers=new_tickers, interval=interval, data=fundamentals, name='fundamentals')

print("New Tickers: ", len(new_tickers))
returns_data = stocks_data.pct_change(1)

print('Grabbing FF5...')
ff5 = gff.famaFrench5Factor(frequency='m')
ff5.drop('RF', axis=1, inplace=True)
ff5.rename(columns={"date_ff_factors": 'Date'}, inplace=True)
ff5.set_index('Date', inplace=True)
ff5.resample(interval).ffill()

print('Adding Factors...')
ff5 = Factor(tickers=new_tickers, interval=interval, data=ff5, general_factor=True)

log_prices = Factor(tickers=new_tickers, interval=interval, data=stocks_data, price_data=True, name='log_prices',
                    transforms=[log_transform])
sma_3 = Factor(tickers=new_tickers, interval=interval, data=returns_data, price_data=True, name='sma_3',
               transforms=[SMA(window=3).transform])
sma_6 = Factor(tickers=new_tickers, interval=interval, data=returns_data, price_data=True, name='sma_6',
               transforms=[SMA(window=6).transform])
sma_12 = Factor(tickers=new_tickers, interval=interval, data=returns_data, price_data=True, name='sma_12',
                transforms=[SMA(window=12).transform])
kalman_filter = Factor(tickers=new_tickers, interval=interval, data=returns_data, price_data=True, name='kalman_filter',
                       transforms=[KalmanFilter().transform])
butters = Factor(tickers=new_tickers, interval=interval, data=returns_data, price_data=True, name='butters',
                 transforms=[Butterworth().transform])
gaussian = Factor(tickers=new_tickers, interval=interval, data=returns_data, price_data=True, name='butters',
                  transforms=[Gaussian().transform])
median = Factor(tickers=new_tickers, interval=interval, data=returns_data, price_data=True, name='median',
                transforms=[Median().transform])
# wavelet = Factor(tickers=stocks_list, interval=interval, data=stocks_data, price_data=True, name='wavelet',
#                  transforms=[Wavelet().transform])
time_decomposition = Factor(tickers=new_tickers, interval=interval, data=stocks_data, price_data=True,
                            name='time_decomposition', transforms=[TimeDecomposition().transform])

model = FactorModel(tickers=new_tickers, interval=interval)

model.add_factor(ff5)
model.add_factor(log_prices)
model.add_factor(sma_3)
model.add_factor(sma_6)
model.add_factor(sma_12)
model.add_factor(fundamentals)
# model.add_factor(kalman_filter)
# model.add_factor(butters)
# model.add_factor(gaussian)
# model.add_factor(median)
# model.add_factor(wavelet)
# model.add_factor(time_decomposition)

print('Fitting Alpha Factor Model...')
model.fit(returns_data.loc[datetime(2002, 1, 1):datetime(2022, 11, 1)],
          'xgb', time='t+1', subsample=0.8, reg_lambda=1.2, reg_alpha=0.5)
statistics = model.backtest(datetime(2014, 1, 1), datetime(2022, 11, 1), returns=returns_data)
statistics.find_factor_significance()
statistics.print_statistics_report()
# statistics.get_full_qs()
# statistics.get_html()
