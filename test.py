import pandas as pd
import numpy as np
import yfinance as yf
import getFamaFrenchFactors as gff
from datetime import datetime
from factor_model_lib.factor_model import FactorModel
from factor_model_lib.factor import Factor
from factor_model_lib.transforms import *

interval = 'M'

# original getting data with yfinance
stocks_list = ['AAPL', 'AMZN']
stocks_data = yf.download(stocks_list, interval='1d', start='2014-12-08')['Adj Close']
stocks_data = stocks_data.resample(interval, convention='end').ffill()
stocks_data.index = pd.to_datetime(stocks_data.index).tz_localize(None)
returns_data = stocks_data.pct_change(1)

model = FactorModel(tickers=stocks_list, interval=interval)

ff5 = gff.famaFrench5Factor(frequency='m')
ff5.drop('RF', axis=1, inplace=True)
ff5.rename(columns={"date_ff_factors": 'Date'}, inplace=True)
ff5.set_index('Date', inplace=True)
ff5.resample(interval).ffill()
ff5 = Factor(tickers=stocks_list, interval=interval, data=ff5, general_factor=True)

log_prices = Factor(tickers=stocks_list, interval=interval, data=stocks_data, price_data=True, name='log_prices',
                    transforms=[log_transform])
sma_3 = Factor(tickers=stocks_list, interval=interval, data=stocks_data, price_data=True, name='sma_3',
               transforms=[SMA(window=3).transform])
sma_6 = Factor(tickers=stocks_list, interval=interval, data=stocks_data, price_data=True, name='sma_6',
               transforms=[SMA(window=6).transform])
sma_12 = Factor(tickers=stocks_list, interval=interval, data=stocks_data, price_data=True, name='sma_12',
                transforms=[SMA(window=12).transform])
kalman_filter = Factor(tickers=stocks_list, interval=interval, data=stocks_data, price_data=True, name='kalman_filter',
                       transforms=[KalmanFilter().transform])
butters = Factor(tickers=stocks_list, interval=interval, data=stocks_data, price_data=True, name='butters',
                 transforms=[Butterworth().transform])
gaussian = Factor(tickers=stocks_list, interval=interval, data=stocks_data, price_data=True, name='butters',
                  transforms=[Gaussian().transform])
median = Factor(tickers=stocks_list, interval=interval, data=stocks_data, price_data=True, name='median',
                transforms=[Median().transform])
# wavelet = Factor(tickers=stocks_list, interval=interval, data=stocks_data, price_data=True, name='wavelet',
#                  transforms=[Wavelet().transform])
time_decomposition = Factor(tickers=stocks_list, interval=interval, data=stocks_data, price_data=True,
                            name='time_decomposition', transforms=[TimeDecomposition().transform])

model.add_factor(ff5)
model.add_factor(log_prices)
model.add_factor(sma_3)
model.add_factor(sma_6)
model.add_factor(sma_12)
model.add_factor(kalman_filter)
model.add_factor(butters)
model.add_factor(gaussian)
model.add_factor(median)
# model.add_factor(wavelet)
# model.add_factor(time_decomposition)

print(model.factors.sort_index(axis=1, level=0))
model.fit(returns_data, 'xgb', time='t+1')
statistics = model.backtest(datetime(2018, 1, 1), datetime(2022, 11, 1))
statistics.find_factor_significance()
statistics.print_statistics_report()