import os
import shutil
import ray
import pandas as pd
import warnings

from pathlib import Path
from typing import Optional
from time import perf_counter_ns
from tqdm import tqdm

from factorlib.utils.helpers import _set_index_names_adaptive
from factorlib.types import SpliceBy
from factorlib.utils.system import print_dynamic_line

os.environ['RAY_verbose_spill_logs'] = '0'
warnings.filterwarnings('ignore')


class BaseFactor:
    """
    Behold. The BaseFactor class. This class provides parallel processing functionality for creating custom factor data
    sets. Simply create a derived class, and override the generate_data static method to create factors with parallel
    processing. Below are a few guidelines and requirements of the class, as well as a fairly complicated example of
    performing rolling cross-sectional K-means clustering across a dataset of stock returns.

    Guidelines/requirements:
        - All raw data that will be used in the creation of the factor must be loaded and read in the constructor.
        - Once all data has been read into dataframes, all the data must be merged into one input dataframe and assigned
          to the `self.data` attribute.
        - All helper functions created in the class to use in `generate_data(data: pd.DataFrame)` must be static
          methods.
            - The data passed to these helper functions will be the batches and slices of self.data.
        - In derived classes, adding member attributes that you would like to use in generate_data will be automatically
          passed to generate_data through the **kwargs attribute. Access the variable by using kwargs['attribute_name'].
            - Passing these attributes as kwargs is automatically handled under the hood by base_factor.
        - After parallel processing, the data processed on each core is concatenated row-wise to form one single output
          dataframe. However, if you need to perform more post-processing, such as re-ordering index levels or setting
          column names, you can override post_process(self, output_files: list[pd.DataFrame]) to perform this
          post-processing.
            - Note, if choose to do so, you are responsible for concatenating the list of dataframes as well. We do this
              to ensure the most customizable interface for you.

    Example (K-means clustering):

    class FactorKmeanCluster(BaseFactor):
        def __init__(self, name: str,
                     splice_size: int = 22,
                     batch_size: int = 8,
                     splice_by: str = SpliceBy.date.value,
                     rolling: int = 0,
                     general_factor: bool = False,
                     tickers: Optional[list[str]] = None,
                     data_dir: Optional[Path] = Path('./data/factor_data'),
                     n_clusters: int = 0):

            super().__init__(name=name, splice_size=splice_size, batch_size=batch_size, splice_by=splice_by, rolling=rolling,
                             general_factor=general_factor, tickers=tickers, data_dir=data_dir)

            # declare extra attributes specific to this derived class
            self.n_clusters = n_clusters

            # load in price_data
            price_data = pd.read_parquet(get_raw_data_dir() / 'data_price.parquet.brotli')

            # pre-process price_data
            price_data = set_timeframe(price_data, '2006-01-01', '2023-01-01')
            price_data = set_length(price_data, year=2)
            window_size = 10
            price_data = create_smooth_return(self.data, windows=[1], window_size=window_size)
            ret = price_data[[f'RET_01']]
            ret = ret['RET_01'].unstack('ticker')
            ret.iloc[:window_size + 1] = ret.iloc[:window_size + 1].fillna(0)

            # set the final input data to self.data
            self.data = (ret - ret.mean()) / ret.std()

        @staticmethod
        @ray.remote
        def generate_data(data: pd.DataFrame, **kwargs):
            data.drop(columns=data.columns[data.isna().sum() > len(data) / 2], inplace=True)
            data.fillna(0, inplace=True)

            curr_date = data.index[-1]

            # Run kmeans
            n_clusters = kwargs['n_clusters']
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0, n_init=10)
            result_clusters = kmeans.fit_predict(data.T)

            # Create a dataframe that matches loadings to ticker
            data = pd.DataFrame(result_clusters, columns=[curr_date], index=data.columns)
            data = data.T

            return data

        def post_process(self, output_files: list[pd.DataFrame]):
            final_output = pd.concat(output_files)
            final_output = final_output.stack()
            final_output.columns = [f'{self.name}_{self.n_clusters}']
            final_output.index.names = ['date', 'ticker'] if final_output.index.get_level_values(1).dtype == datetime \
                else ['ticker', 'date']

            if final_output.index.names[0] == 'date':
                final_output = final_output.swaplevel().to_frame()

            final_output.sort_index(level=['ticker', 'date'], inplace=True)
            return final_output
    """

    def __init__(self, name: str,
                 splice_size: int = 22,
                 batch_size: int = 8,
                 splice_by: SpliceBy = SpliceBy.ticker,
                 rolling: int = 0,
                 general_factor: bool = False,
                 tickers: Optional[list[str]] = None,
                 data_dir: Optional[Path] = Path('./data'), **kwargs):

        """
        :param name: The name of the factor. This will be the name of the folder inside data_dir, and the filename of
                    the exported factor.
        :param splice_size: The size of each splice. The meaning of this number is dependent on the value of `slice_by`
        :param batch_size: The number of slices in each batch
        :param splice_by: This determines the index with which to groupby when creating splices.
                          If splice_by == 'ticker', splice_size represents the number of tickers in each splice.
                          If splice_by == 'date', splice_size represents the number of date intervals in each splice
        :param general_factor: A factor is a general factor if the data is the same for every ticker.
                               Example: macroeconomic data.
        :param tickers: A list of tickers to multiply the data across if the factor is a general_factor. Unused if
                        general_factor == False.
        :param data_dir: The output directory of the output file. This should be the directory that holds all of your
                         factors. Output file structure is as follows:
                                 │── data_dir
                                 │   │ ── name1_dir  # factor 1
                                 │   │    │ ── name1.parquet.brotli  # factor1 dataset
                                 │   │ ── name2_dir  # factor 2
                                 │   │    │ ── name2.parquet.brotli  # factor2 dataset
        """

        self.name = name
        self.splice_size = splice_size
        self.batch_size = batch_size
        self.splice_by = splice_by.value
        self.rolling = rolling
        self.general_factor = general_factor
        self.tickers = tickers
        self.data_dir = data_dir
        self.data = None

        if self.splice_by == 'ticker':
            if self.rolling > 0:
                raise(ValueError, 'If splice_by is set to ticker, there is no need to set rolling to true. Simply '
                                  'apply your rolling functionality in the generate_data function with normal pandas '
                                  'operations.')

    @staticmethod
    @ray.remote
    def generate_data(data: pd.DataFrame, **kwargs):
        pass

    def generate_factor(self):
        print_dynamic_line()
        ray.init()
        print('Transforming data...')
        kwargs = self._get_derived_attributes()
        splices = self._splice_data()
        batches = self._batch_data(splices)
        self._parallel_process(batches, **kwargs)
        print(f'Factor: {self.name} exported to {self.data_dir}.')
        ray.shutdown()
        print_dynamic_line()

    def post_process(self, output_files: list[pd.DataFrame], **kwargs):
        return pd.concat(output_files, axis=0)

    def _parallel_process(self, batches: list[list[pd.DataFrame]], **kwargs):
        # Create folder
        shutil.rmtree(self.data_dir / self.name, ignore_errors=True)
        os.makedirs(self.data_dir / self.name)
        output_files = []
        for batch in tqdm(batches):
            ray_data = (self._execute_ray(self.generate_data, batch, **kwargs))
            output_files.append(pd.concat(ray_data, axis=0))

        output_files = self.post_process(output_files)

        output_files.to_parquet(self.data_dir / self.name / f'{self.name}.parquet.brotli', compression='brotli')

    def _splice_data(self):
        if self.general_factor:
            if 'ticker' in self.data.index.names:
                raise TypeError('You have specified this factor as a `general_factor`, but it has an index for '
                                'tickers. Set general_factor = False to resolve this issue.')

            self.data = self._create_multi_index(self.data)

        if len(self.data.index.names) > 1:
            self.data.index.names = _set_index_names_adaptive(self.data)
        else:
            if self.rolling == 0:
                warnings.warn('Your index only has one level and you have not specificed a rolling option. '
                              'Unless this is intentional the program will most likely crash.')

        data_to_iterate = self.data.groupby(level=self.splice_by, group_keys=False)
        if self.rolling != 0:
            data_to_iterate = self._get_rolling_groups()

        data_spliced = []
        splice_data = []
        for count, (_, df) in enumerate(data_to_iterate):
            splice_data.append(df)
            if (count + 1) % self.splice_size == 0:  # Reached splice size, save and reset
                curr_df = pd.concat(splice_data, axis=0)
                curr_df = self._set_index_adaptive(curr_df)

                data_spliced.append(curr_df)
                splice_data = []

        if len(splice_data) > 0:
            curr_df = pd.concat(splice_data, axis=0)
            curr_df = self._set_index_adaptive(curr_df)
            data_spliced.append(curr_df)

        return data_spliced

    def _batch_data(self, splice_data: list[pd.DataFrame]):
        factor_batches = []
        batch = []

        for i, item in enumerate(splice_data):
            batch.append(item)
            if (i + 1) % self.batch_size == 0:
                factor_batches.append(batch)
                batch = []

        # Append remaining items in batch, if any
        if len(batch) > 0:
            factor_batches.append(batch)
        return factor_batches

    @staticmethod
    def _execute_ray(operation, batch_data, **kwargs):
        kwargs_ref = {key: ray.put(kwargs[key]) for key in kwargs}
        results = [operation.remote(batch, **kwargs_ref) for batch in batch_data]
        results = ray.get(results)
        return results

    def _create_multi_index(self, factor_data):
        factor_values = pd.concat([factor_data] * len(self.tickers), ignore_index=True).values

        multi_index = pd.MultiIndex.from_product([self.tickers, factor_data.index])
        multi_index_factor = pd.DataFrame(factor_values, columns=factor_data.columns, index=multi_index)

        return multi_index_factor

    def _get_derived_attributes(self):
        # Get attributes of the current instance
        derived_attributes = vars(self)

        # Get attributes of an instance of the base class
        base_instance = BaseFactor(name='default_name')
        base_attributes = vars(base_instance)

        # Return attributes unique to the derived class
        return {key: value for key, value in derived_attributes.items() if key not in base_attributes}

    @staticmethod
    @ray.remote
    def _get_window(data: pd.DataFrame, window_size: int, i: int):
        return data.iloc[i: i + window_size]

    def _get_rolling_groups(self, batch_size=10):
        if 'ticker' in self.data.index.names:
            data_unstacked = self.data.unstack('ticker')
        else:
            data_unstacked = self.data

        print("--------------------------------------------------------")
        print('Calculating rolling groups for parallel processing...')
        start = perf_counter_ns()

        groups = []
        window_futures = []
        for i in tqdm(range(0, len(data_unstacked) - self.rolling + 1)):
            future = self._get_window.remote(data_unstacked, self.rolling, i)
            window_futures.append(future)

            if len(window_futures) == batch_size or i == len(data_unstacked) - self.rolling:
                batch_results = ray.get(window_futures)
                groups.extend(batch_results)
                window_futures = []  # Clear the futures for the next batch

        end = perf_counter_ns()
        print(f'Rolling groups created. \nTime elapsed: {round((end - start) / 1e9, 3)} seconds.')

        groups_with_fillers = zip(['filler'] * len(groups), groups)
        return groups_with_fillers

    @staticmethod
    def _set_index_adaptive(df: pd.DataFrame):
        if len(df.index.names) > 1:
            df.reset_index(inplace=True)
            df.set_index(['date', 'ticker'], inplace=True)
        else:
            df.reset_index(inplace=True)
            df.set_index(['date'], inplace=True)
        return df
