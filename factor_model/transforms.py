import inspect
import numpy as np
import pandas as pd
import pywt
from pykalman import KalmanFilter as kf
from scipy.signal import butter, lfilter, medfilt
from scipy.ndimage import gaussian_filter
import statsmodels.api as sm

# TODO: Avellanada PCA transformation
# TODO: signature transformation from https://kormilitzin.github.io/the-signature-method-in-machine-learning/


def log_transform(data: pd.DataFrame):
    logged_data = np.log(data)

    logged_data = _rename_columns_after_transform(logged_data, transform='logged')
    return logged_data


class SMA:
    def __init__(self, window=30):
        self.window = window

    def transform(self, data: pd.DataFrame):
        sma = data.rolling(self.window).mean()

        sma = _rename_columns_after_transform(sma, transform='sma', attribute=str(self.window))
        return sma


class EMA:
    def __init__(self, window=30):
        self.window = window

    def transform(self, data: pd.DataFrame):
        ema = data.ewm(span=self.window).mean()

        ema = _rename_columns_after_transform(ema, transform='ema', attribute=str(self.window))
        return ema


class ZScore:
    def __init__(self, window=30):
        self.window = window

    def transform(self, data):
        roller = data.rolling(window=self.window)
        mean = roller.mean().shift(1)
        stddev = roller.std(ddof=0).shift(1)
        zscore = (data - mean) / stddev

        zscore = _rename_columns_after_transform(zscore, transform='zscore', attribute=str(self.window))
        return zscore


class Rank:
    def __init__(self, window=30, ascending=False, replace_original=False):
        self.window = window
        self.ascending = ascending
        self.replace_original = replace_original

    def transform(self, data):
        # still need to rename columns
        if self.replace_original:
            ranked_data = data.rank(axis=1, ascending=self.ascending)
        else:
            ranked_data = data.rank(axis=1, ascending=self.ascending)
            ranked_data = pd.concat([data, ranked_data], axis=1)
        return ranked_data


class Momentum:
    def __init__(self, window=30):
        self.window = window

    def transform(self, data):
        momentum = data.diff(self.window)
        momentum = _rename_columns_after_transform(momentum, transform='momentum', attribute=str(self.window))
        return momentum


# DO NOT USE YET
class KalmanFilter:
    def __init__(self, transition_matrices=[1], observation_matrices=[1], initial_state_mean=0,
                 initial_state_covariance=1, observation_covariance=1, transition_covariance=0.001):
        if transition_matrices is None:
            transition_matrices = [1]
        self.transition_matrices = transition_matrices
        self.observation_matrices = observation_matrices
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance
        self.observation_covariance = observation_covariance
        self.transition_covariance = transition_covariance

    def transform(self, data):
        k_filter = kf(transition_matrices=self.transition_matrices,
                      observation_matrices=self.observation_matrices,
                      initial_state_mean=self.initial_state_mean,
                      initial_state_covariance=self.initial_state_covariance,
                      observation_covariance=self.observation_covariance,
                      transition_covariance=self.transition_covariance)

        kalman_filtered_means = data.copy(deep=True)
        kalman_filtered_stdevs = data.copy(deep=True)
        for ticker in data.columns.get_level_values(0).unique():
            for factor in data.loc[:, ticker].columns.unique():
                mean, cov = k_filter.filter(data.loc[:, (ticker, factor)])
                mean, std = mean.squeeze(), np.std(cov.squeeze())
                kalman_filtered_means.loc[:, (ticker, factor)] = mean
                kalman_filtered_stdevs.loc[:, (ticker, factor)] = std

        kalman_filtered_means = _rename_columns_after_transform(kalman_filtered_means, transform='kf_mean')
        kalman_filtered_stdevs = _rename_columns_after_transform(kalman_filtered_stdevs, transform='kf_stdev')
        kalman_filtered = pd.concat([kalman_filtered_means, kalman_filtered_stdevs], axis=1)
        return kalman_filtered


class Butterworth:
    def __init__(self, cutoff=0.05, order=3):
        self.cutoff = cutoff
        self.order = order

    def transform(self, data):
        butter_filtered = data.copy(deep=True)
        b, a = butter(self.order, self.cutoff, 'lowpass')
        for ticker in butter_filtered.columns.get_level_values(0).unique():
            for factor in butter_filtered.loc[:, ticker].columns.unique():
                butter_filtered.loc[:, (ticker, factor)] = lfilter(b, a, butter_filtered.loc[:, (ticker, factor)])

        butter_filtered = _rename_columns_after_transform(butter_filtered, transform='butterworth')
        return butter_filtered


class Gaussian:
    def __init__(self, stdev=1):
        self.stdev = stdev

    def transform(self, data):
        gaussian_filtered = data.copy(deep=True)
        for ticker in gaussian_filtered.columns.get_level_values(0).unique():
            for factor in gaussian_filtered.loc[:, ticker].columns.unique():
                gaussian_filtered.loc[:, (ticker, factor)] = gaussian_filter(gaussian_filtered.loc[:, (ticker, factor)],
                                                                             sigma=self.stdev)

        gaussian_filtered = _rename_columns_after_transform(gaussian_filtered, transform='gaussian')
        return gaussian_filtered


class Median:
    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size

    def transform(self, data):
        median_filtered = data.copy(deep=True)
        for ticker in median_filtered.columns.get_level_values(0).unique():
            for factor in median_filtered.loc[:, ticker].columns.unique():
                median_filtered.loc[:, (ticker, factor)] = medfilt(median_filtered.loc[:, (ticker, factor)],
                                                                   kernel_size=self.kernel_size)

        median_filtered = _rename_columns_after_transform(median_filtered, transform='median')
        return median_filtered


# class Wavelet:
#     def __init__(self, window=5, wavelet='db1'):
#         self.window = window
#         self.wavelet = wavelet
#
#     def transform(self, data):
#         wavelet_param_1 = data
#         wavelet_param_2 = data
#         for ticker in data.columns.get_level_values(0).unique():
#             for factor in data.loc[:, ticker].columns.unique():
#                 param_1, param_2 = pywt.dwt(data)
#                 wavelet_param_1.loc[:, (ticker, factor)] = param_1
#                 wavelet_param_2.loc[:, (ticker, factor)] = param_2
#
#         wavelet_param_1 = _rename_columns_after_transform(wavelet_param_1, transform='wavelet_p1')
#         wavelet_param_2 = _rename_columns_after_transform(wavelet_param_2, transform='wavelet_p2')
#         wavelet_filtered = pd.concat([wavelet_param_1, wavelet_param_2], axis=1)
#         return wavelet_filtered


class TimeDecomposition:
    def __init__(self, seasonal=3):
        self.seasonal = seasonal

    def transform(self, data):
        trend_decomposition = data.copy(deep=True)
        seasonal_decomposition = data.copy(deep=True)
        residual_decomposition = data.copy(deep=True)
        for ticker in data.columns.get_level_values(0).unique():
            for factor in data.loc[:, ticker].columns.unique():
                decomposition = sm.tsa.STL(data.loc[:, (ticker, factor)], seasonal=self.seasonal).fit()
                trend_decomposition.loc[:, (ticker, factor)] = decomposition.trend
                seasonal_decomposition.loc[:, (ticker, factor)] = decomposition.seasonal
                residual_decomposition.loc[:, (ticker, factor)] = decomposition.resid

        trend_decomposition = _rename_columns_after_transform(trend_decomposition, transform='trend_decomp')
        seasonal_decomposition = _rename_columns_after_transform(seasonal_decomposition, transform='sznl_decomp')
        residual_decomposition = _rename_columns_after_transform(residual_decomposition, transform='resid_decomp')
        time_decomposition = pd.concat([trend_decomposition, seasonal_decomposition, residual_decomposition], axis=1)
        return time_decomposition


def _rename_columns_after_transform(data, transform, attribute=''):
    new_columns = [column + '_' + transform + attribute for column in data.columns.get_level_values(1)]
    new_columns = dict(zip(data.columns.get_level_values(1), new_columns))
    data.rename(columns=new_columns, inplace=True, level=1)
    return data


def _get_defining_class(meth):
    if inspect.ismethod(meth):
        for cls in inspect.getmro(meth.__self__.__class__):
            if meth.__name__ in cls.__dict__:
                return cls
    if inspect.isfunction(meth):
        return getattr(inspect.getmodule(meth),
                       meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0],
                       None)
    return None
