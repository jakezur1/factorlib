import yfinance as yf
import pandas as pd

# original getting data with yfinance
with open('./data/spy.txt', 'r') as f:
    stocks_list = f.read().splitlines()
stocks_data = yf.download(stocks_list, interval='1d', period='max')['Adj Close']
stocks_data.index = pd.to_datetime(stocks_data.index)
stocks_data.to_csv('./data/spy_data_daily.csv')