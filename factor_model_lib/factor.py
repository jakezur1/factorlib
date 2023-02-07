import pandas as pd
import numpy as np
import yfinance as yf

yf_intervals = {
    '1m': '1m',
    '2m': '2m',
    '5m': '5m',
    '15m': '15m',
    '30m': '30m',
    '60m': '60m',
    '90m': '90m',
    '1h': '1h',
    'D': '1d',
    '5D': '5d',
    'W': '1wk',
    'M': '1mo',
    '3M': '3mo',
}


class Factor:
    def __init__(self, tickers: list, name: str = None, data: pd.DataFrame = None,
                 interval: str = 'D', general_factor: bool = False,
                 transforms=[], price_data=False):
        self.tickers = tickers
        self.name = name
        self.interval = interval
        self.data = data.resample(interval).ffill()
        self.transforms = transforms

        try:
            self.data.index = pd.to_datetime(self.data.index)
        except Exception as e:
            print(f'could not convert index to datetime of factor {self.name}, moving on.')

        try:
            self.data.index = self.data.index.tz_localize(None)
        except Exception as e:
            print(f'could not localize index of factor: {self.name}, moving on.')

        assert not(general_factor and price_data), \
            "general_factor and price_data cannot both be True"
        if general_factor:
            multi_index_factors = pd.DataFrame(columns=pd.MultiIndex.from_product([self.tickers, data.columns]))
            for ticker in self.tickers:
                multi_index_factors[ticker] = data
            self.data = multi_index_factors
        elif price_data:
            multi_index_prices = pd.DataFrame(columns=pd.MultiIndex.from_product([self.tickers, ['close']]))
            for ticker in self.tickers:
                prices_to_add = data[ticker].to_frame()
                prices_to_add.columns = ['close']
                multi_index_prices[ticker] = prices_to_add
            self.data = multi_index_prices

        if len(transforms) > 0:
            for transform in transforms:
                self.data = transform(self.data)
