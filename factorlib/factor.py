import pandas as pd
import numpy as np
import yfinance as yf
from .transforms import yf_intervals

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
            index = data.index
            multi_index = pd.MultiIndex.from_product([self.tickers, data.columns])
            factor_values = np.tile(data.values, (1, len(self.tickers)))
            multi_index_factors = pd.DataFrame(factor_values, columns=multi_index, index=index)
            self.data = multi_index_factors
        elif price_data:
            index = data.index
            data = data[self.tickers]
            multi_index = pd.MultiIndex.from_product([self.tickers, ['close']])
            multi_index_prices = pd.DataFrame(data.values, columns=multi_index, index=index)
            multi_index_prices.columns = multi_index_prices.columns.set_levels(data.columns, level=0)
            self.data = multi_index_prices

        if len(transforms) > 0:
            for transform in transforms:
                self.data = transform(self.data)
