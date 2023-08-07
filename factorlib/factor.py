import pandas as pd
import numpy as np

from typing import Optional
from datetime import datetime

from factorlib.utils.helpers import _set_index_names_adaptive
from factorlib.utils.system import show_processing_animation, _spinner_animation


class Factor:
    """
    This class represents a factor for the FactorModel class. It is responsible for formatting user data, and
    transforming it for use in a `FactorModel`.
    """

    def __init__(self, name: str = None,
                 data: pd.DataFrame = None,
                 interval: str = 'D',
                 tickers: Optional[list[str]] = None,
                 price_data: bool = False,
                 general_factor: bool = False,
                 transforms: Optional[list[any]] = None,
                 transform_columns: Optional[list[str]] = None,
                 categorical: Optional[list[str]] = None,
                 normalize: bool = False):
        """
        :param name: The name of the factor. If multiple factors are included in this object, name it by a
                     general category that separates this dataset from other factor objects.
        :param data: The pandas dataframe that will serve as the data for this factor. If the factor is a
                     general_factor, it must have an index called 'date'. If the factor is a price_factor it must have
                     an index called 'date', and each column must be the ticker for which the column represents. If
                     the column is not a general_factor or a price_factor (most cases) it must have a multi-index where
                     the levels of the columns are name 'date' and 'ticker'. Level orders do not matter.
        :param interval: The desired interval of the time series. This is the interval between each row of the
                         date column. The data will be resampled to this interval.
        :param price_data: True if this factor is formatted as price data. Price data is formatted with an index called
                           'date', and columns named after the tickers for which that column represents.
        :param general_factor: True if this factor is a general factor for all tickers. If so, the tickers parameter is
                               not optional and may not be left as None.
        :param tickers: A list of tickers. This will only be used if the factor is a `general_factor`. If so, it will
                        create a tickers index by performing cartesian multiplication between the dates and the
                        given list of tickers.
        :param transforms: A list of functions or functors that will perform transforms on the data. These functions
                           must only take in a pandas dataframe or series. If the desired transform requires more
                           parameters than just the data to operate on, create a functor class and pass function
                           parameters as member variables of the functor class. See factorlib.transforms for examples.
        :param transform_columns: A list of columns for which the transforms should be applied. If left None, transforms
                                  will be applied to the entire dataframe, except for categorical columns.
        :param categorical: The columns that should be considered as categorical variables for XGBoost during
                            walk-forward optimization.
        :param normalize: TODO: normalize row-wise if normalize=True
        """

        @show_processing_animation(message_func=lambda name, *args, **kwargs: f'Creating factor: {name}',
                                   animation=_spinner_animation)
        def inner_init(_name, _data, _interval, _tickers, _price_data, _general_factor, _transforms, _transform_columns,
                       _categorical, _normalize):
            assert ('date' in data.index.names), f'Exiting due to missing \'date\' index in factor named: {name}. ' \
                                                 f'See factorlib.factor docstring for factor formatting details.'

            self.name = name
            self.data = data
            self.interval = interval
            self.tickers = tickers
            self.transforms = transforms
            self.categorical = categorical

            self.data = self.data.reset_index().set_index('date')
            self.data.index = pd.to_datetime(self.data.index)
            self.data.sort_index(inplace=True)

            if not general_factor and not price_data:
                assert ('ticker' in self.data.columns), 'The data you provided is neither a general_factor, nor ' \
                                                        'price_data, but it does not have a ticker index or column. ' \
                                                        'Please reformat your data to include a ticker index, or ' \
                                                        'set the general_factor=True or price_data=True if it is ' \
                                                        'appropriate.'
                self.data.set_index('ticker', append=True, inplace=True)

            self.start = self.data.index.get_level_values('date')[0]
            self.end = self.data.index.get_level_values('date')[-1]

            if general_factor:
                assert (tickers is not None), 'All general factors mut be supplied with the `tickers` parameter. ' \
                                              'Ideally, this should be fully comprehensive list of all tickers that ' \
                                              'you plan to use in the factor model to which this factor will be added.'
                self.data = self._create_multi_index(self.data)
            elif price_data:
                self.data = self.data.stack()
                self.data.name = name + '_price_data'

            self.data.index.names = _set_index_names_adaptive(self.data)

            # TODO: Ideally we should be able to infer the frequency of the index. Extremely hard when holidays can be
            #  excluded for 'B' day interval, which is most common
            self.data = self.data.reset_index(level='ticker').groupby(by='ticker').resample(
                self.interval).ffill().reset_index(level='ticker', drop=True).set_index('ticker', append=True)

            if _transform_columns is None:
                transform_columns = self.data.columns
                if categorical is not None:
                    transform_columns = [column for column in self.data.columns if column not in categorical]

            if categorical is not None:
                for column in self.data.columns:
                    if column in categorical:
                        self.data[column] = self.data[column].astype(pd.Categorical)

            if self.tickers is not None:
                self.data = self.data[self.data.index.get_level_values('ticker').isin(self.tickers)]

            if transforms is not None:
                for transform in transforms:
                    self.data[transform_columns] = transform(self.data[transform_columns])

        inner_init(name, data, interval, tickers, price_data, general_factor, transforms, transform_columns,
                   categorical, normalize)

    def _create_multi_index(self, factor_data, tickers: Optional[list[str]] = None):
        if tickers is None:
            tickers = self.tickers

        factor_values = pd.concat([factor_data] * len(tickers), ignore_index=True).values

        multi_index = pd.MultiIndex.from_product([tickers, factor_data.index])
        multi_index_factor = pd.DataFrame(factor_values, columns=factor_data.columns, index=multi_index)

        return multi_index_factor
