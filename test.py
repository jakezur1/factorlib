import yfinance as yf
import pandas_ta as ta
from datetime import datetime, timedelta
from fastai.tabular.all import add_datepart
from factorlib.factor_model import FactorModel
from factorlib.factor import Factor
from factorlib.transforms import *
from factorlib.statistics import Statistics

interval = 'D'
start = '2002-01-01'
end = '2021-01-01'

print('Reading in Stock Data...')
stocks_data = pd.read_csv('./data/spy_data_daily.csv', index_col=0)
stocks_data.index = pd.to_datetime(stocks_data.index).tz_localize(None).floor('D')
stocks_data = stocks_data.resample(interval, convention='end').ffill()

print('Adding in Date Features...')
date_df = pd.DataFrame({'date': stocks_data.index})
date_df = add_datepart(date_df, 'date', drop=False, time=False)
date_df.index = stocks_data.index

print('Reading in Fundamentals Data...')
fundamentals = df = pd.read_csv('./data/fundamentals_spy_only.csv', header=[0, 1], index_col=0)
fundamentals.index = pd.to_datetime(fundamentals.index)

tickers = fundamentals.columns.get_level_values(0).unique().tolist()
new_tickers = list(set(tickers) & set(stocks_data.columns))
fundamentals = fundamentals[new_tickers]

print("Universe of Tickers: ", len(new_tickers), " Total")
returns_data = stocks_data.pct_change(1)

print('Grabbing FF5...')
ff5 = pd.read_csv('./data/fff-daily.csv', index_col='Date', parse_dates=['Date'])
ff5.resample(interval).ffill()

print('Grabbing Indices...')
indices_df = yf.download('BND TLT QQQ GDX TMF WTI VIX', start=start, end=end, interval='1d')['Adj Close']
indices_df.index = pd.to_datetime(indices_df.index).tz_localize(None).floor('D')
indices_df = indices_df.resample(interval, convention='end').ffill()
indices_returns = indices_df.pct_change(1)

print('Adding Factors...')
# Fundamentals
delta_fundamentals = Factor(tickers=new_tickers, interval=interval, data=fundamentals,
                            name='delta_fundamentals', transforms=[Momentum(window=1, pct_change=True).transform])
ranked_fundamentals = Factor(tickers=new_tickers, interval=interval, data=fundamentals, name='ranked_fundamentals',
                             transforms=[Rank(replace_original=True).transform])
fundamentals = Factor(tickers=new_tickers, interval=interval, data=fundamentals, name='fundamentals')

# Date Factors
date = Factor(tickers=new_tickers, interval=interval, data=date_df, general_factor=True, name='date_factor')

# General Factors
ff5 = Factor(tickers=new_tickers, interval=interval, data=ff5, general_factor=True)
indices_factor = Factor(tickers=new_tickers, interval=interval, data=indices_returns, general_factor=True,
                        name='index')

# Technical Indicator Factors
# need to run by series instead of .apply
# rsi_data = stocks_data.apply(ta.momentum.rsi, length=14, axis=0)
# rsi = Factor(tickers=new_tickers, interval=interval, data=rsi_data, name='rsi')

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
                    name='vols', transforms=[Volatility(window=60).transform])
returns_shifted_one = Factor(tickers=new_tickers, interval=interval, data=returns_data.shift(1), price_data=True,
                             name='returns_shifted_1')
returns_shifted_seven = Factor(tickers=new_tickers, interval=interval, data=returns_data.shift(7), price_data=True,
                               name='returns_shifted_7')
returns_shifted_20 = Factor(tickers=new_tickers, interval=interval, data=returns_data.shift(20), price_data=True,
                            name='returns_shifted_20')
sma_3 = Factor(tickers=new_tickers, interval=interval, data=stocks_data, price_data=True, name='sma_3',
               transforms=[SMA(window=3).transform])
sma_6 = Factor(tickers=new_tickers, interval=interval, data=stocks_data, price_data=True, name='sma_6',
               transforms=[SMA(window=6).transform, Momentum(window=1, pct_change=True).transform])
sma_12 = Factor(tickers=new_tickers, interval=interval, data=stocks_data, price_data=True, name='sma_12',
                transforms=[SMA(window=12).transform, Momentum(window=1, pct_change=True).transform])
sma_30 = Factor(tickers=new_tickers, interval=interval, data=stocks_data, price_data=True, name='sma_30',
                transforms=[SMA(window=30).transform, Momentum(window=1, pct_change=True).transform])

# momentum here is just taking the diff over the last X window
price_momentum_diff = Factor(tickers=new_tickers, interval=interval, data=stocks_data,
                             price_data=True, name='momentum_60', transforms=[Momentum(window=60, pct_change=True).transform])
short_momentum_diff = Factor(tickers=new_tickers, interval=interval, data=stocks_data,
                             price_data=True, name='momentum_10', transforms=[Momentum(window=10, pct_change=True).transform])
medium_momentum_diff = Factor(tickers=new_tickers, interval=interval, data=stocks_data,
                              price_data=True, name='momentum_30', transforms=[Momentum(window=30, pct_change=True).transform])

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
model.add_factor(returns_shifted_20)
model.add_factor(returns_shifted_one)
model.add_factor(returns_shifted_seven)
model.add_factor(volatility)
model.add_factor(sma_3)
model.add_factor(sma_6)
model.add_factor(sma_12)
model.add_factor(sma_30)
model.add_factor(fundamentals)
model.add_factor(price_momentum_diff)
model.add_factor(short_momentum_diff)
model.add_factor(medium_momentum_diff)

print('Fitting Alpha Factor Model...')
# model.fit(returns_data.loc[datetime(2002, 1, 1):datetime(2022, 11, 1)],
#           'xgb', time='t+1', subsample=0.8, reg_lambda=1.2, reg_alpha=0.5)
# statistics = model.backtest(datetime(2014, 1, 1), datetime(2022, 11, 1), returns=returns_data, long_pct=1)
statistics = model.wfo(returns_data,
                       train_interval=timedelta(days=365 * 5), anchored=True,  # interval parameters
                       start_date=datetime(2014, 1, 1),
                       k_pct=0.2, long_only=True,)  # weight parameters

# statistics = Statistics()
# statistics.load('./results/wfo_stats.p')
statistics.print_statistics_report()
# statistics.get_full_qs()

statistics.get_html()
statistics.to_csv('./results/wfo_results')
statistics.save('./results/wfo_stats')
