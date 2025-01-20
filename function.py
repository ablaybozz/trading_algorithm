import ccxt

import pandas as pd
import numpy as np

from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta


def load_binance_data(start_time: int, symbol: str, timeframe: str) -> pd.DataFrame:   
    """
    Loads a limited data of a certain pair from binance, includes open, high, low, close, volume
    Args:
        start_time: timestamp of the earliest data to fetch
        symbol: name of pair, e.g. BTC/USDT
        timeframe: granularity of data, e.g. weeek, day or hr
    Retrurns:
        pd.DataFrame: dataframe of loaded data
        
    """
    exchange=ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=start_time, limit=1000)
    ohlcv_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    ohlcv_df['datetime'] = pd.to_datetime(ohlcv_df['timestamp'], unit='ms')
    return ohlcv_df

def batch_load_hr_data(start_time_str: str, end_time_str: str, symbol: str) -> pd.DataFrame:
    """
    Load iteratievly full data in the given time range from binance
    Args:
        start_time_str: timestamp representing start of time range in text format
        end_time_str: timestamp representing end of time range (inclusively) in text format
        symbol: name of pair
    Returns:
        pd.DataFrame: dataframe of all collected data in given range
    """
    exchange=ccxt.binance()
    start_time = exchange.parse8601(start_time_str)
    end_time = exchange.parse8601(end_time_str)
    timeframe='1h'
    max_timestamp = exchange.parse8601('2024-06-18T00:00:00Z')
    is_first = True
    
    while start_time <= end_time:
        cur_df = load_binance_data(start_time, symbol, timeframe)
        if is_first:
            final_df = cur_df.copy()
            is_first = False
        else:
            final_df = pd.concat([final_df, cur_df], axis=0)
            
        start_time = cur_df['timestamp'].max() + 3_600_000
    return final_df.reset_index(drop=True)

def max_relative_change(series: pd.Series) -> float:
    """
    Finds maximum change in series relative to last value, includes negative changes
    Args:
        series: series of values to select from
    Returns:
        float: the relative value of maximum change in percentage
    """
    neg_change = min((series - series.iloc[-1]) / series.iloc[-1]) * 100
    pos_change = max((series - series.iloc[-1]) / series.iloc[-1]) * 100
    return neg_change if abs(neg_change)>abs(pos_change) else pos_change


def max_rolling_change(df: pd.DataFrame, col: str, time_col:str, window: int = 24) -> pd.DataFrame:
    """
    Finds maximum change relative to each record and adds it to dataframe
    Args:
        df: initial dataframe to work with
        col: name of column to calulcate relative values
        time_col: name of column representing the time, could be timestamp
        window: window to look forwards for searching change
    Returns:
        pd.DataFrame: dataframe with filled max relative changes
    """
    df_copy = df.copy()
    df_copy.sort_values(by=time_col, ascending=False, inplace=True)
    df_copy['max_{}_change'.format(col)] = (
        df_copy[col]
        .rolling(window, min_periods=window)
        .apply(lambda x: max_relative_change(x), raw=False)
    )
    return df_copy.sort_values(by=time_col, ascending=True)

def target_to_bins(value: float, borders: list) -> int:
    """
    Converts a given value to a bin defined by borders of bins
    Args:
        value: value to convert
        borders: list of bin borders
    Returns:
        int: category of bin, numerated ascendingly
    """
    borders.sort()
    for i,lim in enumerate(borders):
        if value<=lim:
            return i
    else:
        return i+1

def col_to_matrix(value_col: pd.Series, time_col: pd.Series, size: int) -> (np.ndarray, np.array):
    """
    Convert a series into matrix of given width so each row has consecuent values
    Also saves the time corresponding to each row
    Args:
        col: column with initial values
        time_col: column with time records
        size: width of the matrix
    Retruns:
        np.ndarray: matrix of given width, drops the remaining values
        np.array: array of time corresponding to each row
    """
    values = value_col.values
    matrix = np.lib.stride_tricks.sliding_window_view(values, size)
    times = time_col.values
    time_array = np.array(times[: len(matrix)])
    return matrix, time_array


