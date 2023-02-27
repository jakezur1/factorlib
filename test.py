import os

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
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
stocks_data.index = pd.to_datetime(stocks_data.index).tz_localize(None).floor('D')
stocks_data = stocks_data.resample(interval, convention='end').ffill()

print('Reading in Fundamentals Data...')

fundamentals = df = pd.read_csv('./data/fundamentals_spy_only.csv', header=[0, 1], index_col=0)
fundamentals.index = pd.to_datetime(fundamentals.index)

tickers = fundamentals.columns.get_level_values(0).unique().tolist()

new_tickers = list(set(tickers) & set(stocks_data.columns))
fundamentals = fundamentals[new_tickers]

print("Universe of Tickers: ", len(new_tickers), " Total")
returns_data = stocks_data.pct_change(1)

print('Grabbing FF5...')
ff5 = gff.famaFrench5Factor(frequency='m')
ff5.drop('RF', axis=1, inplace=True)
ff5.rename(columns={"date_ff_factors": 'Date'}, inplace=True)
ff5.set_index('Date', inplace=True)
ff5.resample(interval).ffill()

print('Grabbing Indices...')
indices_df = yf.download('SPY BND TLT QQQ GDX TMF WTI VIX', start=start, end=end, interval='1d')['Adj Close']
indices_df.index = pd.to_datetime(indices_df.index).tz_localize(None).floor('D')
indices_df = indices_df.resample(interval, convention='end').ffill()
indices_returns = indices_df.pct_change(1)

print('Adding Factors...')

# Fundamentals
# delta_fundamentals = Factor(tickers=new_tickers, interval=interval, data=fundamentals.pct_change(1),
#                             name='delta_fundamentals')
ranked_fundamentals = Factor(tickers=new_tickers, interval=interval, data=fundamentals, name='ranked_fundamentals',
                             transforms=[Rank(replace_original=True).transform])
fundamentals = Factor(tickers=new_tickers, interval=interval, data=fundamentals, name='fundamentals')

# General Factors
ff5 = Factor(tickers=new_tickers, interval=interval, data=ff5, general_factor=True)
indices_factor = Factor(tickers=new_tickers, interval=interval, data=indices_returns, general_factor=True)

# Returns / Price Factors
log_prices = Factor(tickers=new_tickers, interval=interval, data=stocks_data, price_data=True, name='log_prices',
                    transforms=[log_diff_transform])
ranked_returns = Factor(tickers=new_tickers, interval=interval, data=returns_data, price_data=True,
                        name='ranked_returns',
                        transforms=[Rank(replace_original=True).transform])
ranked_volatility = Factor(tickers=new_tickers, interval=interval, data=returns_data, price_data=True,
                           name='ranked_volatility',
                           transforms=[Volatility(window=60).transform, Rank(replace_original=True).transform])
volatility = Factor(tickers=new_tickers, interval=interval, data=returns_data, price_data=True,
                           name='vols',
                           transforms=[Volatility(window=60).transform])
stock_vol = Factor(tickers=new_tickers, interval=interval, data=returns_data, price_data=True, name='stock_vol')
sma_3 = Factor(tickers=new_tickers, interval=interval, data=returns_data, price_data=True, name='sma_3',
               transforms=[SMA(window=3).transform])
sma_6 = Factor(tickers=new_tickers, interval=interval, data=returns_data, price_data=True, name='sma_6',
               transforms=[SMA(window=6).transform])
sma_12 = Factor(tickers=new_tickers, interval=interval, data=returns_data, price_data=True, name='sma_12',
                transforms=[SMA(window=12).transform])
sma_150 = Factor(tickers=new_tickers, interval=interval, data=returns_data, price_data=True, name='sma_150',
                 transforms=[SMA(window=150).transform])

# momentum here is just taking the diff over the last X window
price_momentum_diff = Factor(tickers=new_tickers, interval=interval, data=returns_data,
                             price_data=True, name='momentum_60', transforms=[Momentum(window=60).transform])
short_momentum_diff = Factor(tickers=new_tickers, interval=interval, data=returns_data,
                             price_data=True, name='momentum_10', transforms=[Momentum(window=10).transform])
medium_momentum_diff = Factor(tickers=new_tickers, interval=interval, data=returns_data,
                              price_data=True, name='momentum_30', transforms=[Momentum(window=30).transform])

# kalman_filter = Factor(tickers=new_tickers, interval=interval, data=returns_data, price_data=True, name='kalman_filter',
#                        transforms=[KalmanFilter().transform])
# butters = Factor(tickers=new_tickers, interval=interval, data=returns_data, price_data=True, name='butters',
#                  transforms=[Butterworth().transform])
# gaussian = Factor(tickers=new_tickers, interval=interval, data=returns_data, price_data=True, name='butters',
#                   transforms=[Gaussian().transform])
# median = Factor(tickers=new_tickers, interval=interval, data=returns_data, price_data=True, name='median',
#                 transforms=[Median().transform])
# wavelet = Factor(tickers=stocks_list, interval=interval, data=stocks_data, price_data=True, name='wavelet',
#                  transforms=[Wavelet().transform])
# time_decomposition = Factor(tickers=new_tickers, interval=interval, data=stocks_data, price_data=True,
#                             name='time_decomposition', transforms=[TimeDecomposition().transform])

model = FactorModel(tickers=new_tickers, interval=interval)

model.add_factor(ff5)
model.add_factor(log_prices)
model.add_factor(indices_factor)
# model.add_factor(ranked_returns)
model.add_factor(sma_3)
model.add_factor(sma_6)
model.add_factor(volatility)
# model.add_factor(ranked_volatility)
# model.add_factor(stock_vol)
model.add_factor(sma_12)
model.add_factor(sma_150)
model.add_factor(fundamentals)
model.add_factor(price_momentum_diff)
model.add_factor(short_momentum_diff)
model.add_factor(medium_momentum_diff)
# model.add_factor(delta_fundamentals)
# model.add_factor(ranked_fundamentals)
# model.add_factor(kalman_filter)
# model.add_factor(butters)
# model.add_factor(gaussian)
# model.add_factor(median)
# model.add_factor(wavelet)
# model.add_factor(time_decomposition)

print('Fitting Alpha Factor Model...')
# model.fit(returns_data.loc[datetime(2002, 1, 1):datetime(2022, 11, 1)],
#           'xgb', time='t+1', subsample=0.8, reg_lambda=1.2, reg_alpha=0.5)
# statistics = model.backtest(datetime(2014, 1, 1), datetime(2022, 11, 1), returns=returns_data, long_pct=1)
statistics = model.wfo(returns_data, train_interval=timedelta(days=252 * 5),
                       anchored=False, k=5, subsample=0.5, max_depth=3, colsample_bytree=0.5, reg_alpha=0.2)
statistics.find_factor_significance()
statistics.print_statistics_report()
statistics.get_full_qs()
# statistics.get_html()
