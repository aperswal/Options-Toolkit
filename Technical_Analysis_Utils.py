import math
import re
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
import json
import requests
import praw
from textblob import TextBlob
from collections import deque, defaultdict
from alpha_vantage.techindicators import TechIndicators
import matplotlib.pyplot as plt

def alpha_load_api_key():
    with open('alpha_secret.json') as file:
        data = json.load(file)
        return data['key']

alpha_api_key = alpha_load_api_key()
ts = TimeSeries(key=alpha_api_key, output_format='pandas')

def calculate_sma(symbol, interval='daily', time_period=20, series_type='close'):
    ti = TechIndicators(key=alpha_api_key, output_format='pandas')
    data, meta_data = ti.get_sma(symbol=symbol, interval=interval, time_period=time_period, series_type=series_type)
    data = data.tail(5)  # Get data for the past week
    data.plot()
    plt.title('Simple Moving Average (SMA)')
    plt.show()
    return data

def calculate_ema(symbol, interval='daily', time_period=20, series_type='close'):
    ti = TechIndicators(key=alpha_api_key, output_format='pandas')
    data, meta_data = ti.get_ema(symbol=symbol, interval=interval, time_period=time_period, series_type=series_type)
    data = data.tail(5)  # Get data for the past week
    data.plot()
    plt.title('Exponential Moving Average (EMA)')
    plt.show()
    return data

def calculate_rsi(symbol, interval='daily', time_period=14, series_type='close'):
    ti = TechIndicators(key=alpha_api_key, output_format='pandas')
    data, meta_data = ti.get_rsi(symbol=symbol, interval=interval, time_period=time_period, series_type=series_type)
    data = data.tail(5)  # Get data for the past week
    data.plot()
    plt.title('Relative Strength Index (RSI)')
    plt.show()
    return data

def calculate_golden_cross(symbol, short_window=50, long_window=200):
    # Calculate the short-term simple moving average
    short_sma = calculate_sma(symbol, time_period=short_window)
    # Calculate the long-term simple moving average
    long_sma = calculate_sma(symbol, time_period=long_window)
    # Create a signal when the short SMA crosses the long SMA
    signal = np.where(short_sma > long_sma, 1.0, 0.0)
    return signal

def calculate_death_cross(symbol, short_window=50, long_window=200):
    # Calculate the short-term simple moving average
    short_sma = calculate_sma(symbol, time_period=short_window)
    # Calculate the long-term simple moving average
    long_sma = calculate_sma(symbol, time_period=long_window)
    # Create a signal when the short SMA crosses below the long SMA
    signal = np.where(short_sma < long_sma, -1.0, 0.0)
    return signal

def calculate_rsi_overbought_oversold(symbol, window=14):
    # Calculate the RSI
    rsi = calculate_rsi(symbol, time_period=window)
    # Create a signal when the RSI is overbought or oversold
    signal = np.where(rsi > 70, -1.0, np.where(rsi < 30, 1.0, 0.0))
    return signal

def get_stock_data_intraday(symbol, interval, outputsize='full'):
    """
    Fetches the stock data from Alpha Vantage API.

    Parameters:
    symbol (str): The stock symbol.
    interval (str): The time interval between data points (1min, 5min, etc.).
    outputsize (str): Size of the data output, 'compact' or 'full'.

    Returns:
    DataFrame: Stock data including open, high, low, close, volume.
    """
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={alpha_api_key}'
    response = requests.get(url)
    data = response.json()

    # Check if the response contains the expected data
    time_series_key = f'Time Series ({interval})'
    if time_series_key not in data:
        raise Exception("API response does not contain the expected data.")

    # Convert JSON data to DataFrame
    time_series_data = data[time_series_key]
    stock_data = pd.DataFrame(time_series_data).T
    stock_data.columns = ['open', 'high', 'low', 'close', 'volume']
    stock_data.index = pd.to_datetime(stock_data.index)
    
    return stock_data

def get_stock_data_daily(symbol, outputsize='full'):
    """
    Fetches the daily stock data from Alpha Vantage API.

    Parameters:
    symbol (str): The stock symbol.
    outputsize (str): Size of the data output, 'compact' or 'full'.

    Returns:
    DataFrame: Daily stock data including open, high, low, close, volume.
    """
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize={outputsize}&apikey={alpha_api_key}'
    response = requests.get(url)
    data = response.json()

    time_series_key = 'Time Series (Daily)'
    if time_series_key not in data:
        raise Exception("API response does not contain the expected data.")

    daily_data = data[time_series_key]
    stock_data = pd.DataFrame(daily_data).T
    stock_data.columns = ['open', 'high', 'low', 'close', 'volume']
    stock_data.index = pd.to_datetime(stock_data.index)

    return stock_data

def analyze_volume_trends(stock_data):
    """
    Analyzes volume trends in relation to price over time.

    Parameters:
    stock_data (DataFrame): Stock data including volume and close price.

    Returns:
    str: Analysis of volume trend in relation to price movement.
    """
    # Ensure columns are numeric
    stock_data[['close', 'volume']] = stock_data[['close', 'volume']].apply(pd.to_numeric)

    # Calculate percentage change for price and volume
    stock_data['price_change'] = stock_data['close'].pct_change()
    stock_data['volume_change'] = stock_data['volume'].pct_change()

    # Analyze the trend
    if (stock_data['price_change'].iloc[-1] > 0) and (stock_data['volume_change'].iloc[-1] > 0):
        return "Price and volume increasing, indicating a strong upward trend."
    elif (stock_data['price_change'].iloc[-1] < 0) and (stock_data['volume_change'].iloc[-1] > 0):
        return "Price decreasing but volume increasing, potential reversal signal."
    elif (stock_data['price_change'].iloc[-1] > 0) and (stock_data['volume_change'].iloc[-1] < 0):
        return "Price increasing but volume decreasing, trend may not be sustainable."
    else:
        return "No clear trend identified based on recent price and volume changes."

def detect_potential_reversal(data):
    """
    Detects potential market reversal based on price and volume trends.

    Parameters:
    data (DataFrame): Stock data including volume and close price.

    Returns:
    DataFrame: Indicator of potential reversal points.
    """
    # Ensure columns are numeric
    data[['close', 'volume']] = data[['close', 'volume']].apply(pd.to_numeric)

    # Calculate percentage changes
    data['price_change'] = data['close'].pct_change()
    data['volume_change'] = data['volume'].pct_change()

    # Define reversal conditions
    increasing_price = data['price_change'] > 0
    decreasing_volume = data['volume_change'] < 0
    decreasing_price = data['price_change'] < 0
    increasing_volume = data['volume_change'] > 0

    # Determine potential reversal points
    data['potential_reversal'] = ((increasing_price & decreasing_volume) | (decreasing_price & increasing_volume))

    return data[['close', 'volume', 'potential_reversal']]

if __name__ == '__main__':
    symbol = 'MSFT'
    intraday_interval = '5min'
    intraday_data = get_stock_data_intraday(symbol, intraday_interval)
    
    daily_data = get_stock_data_daily(symbol)

    # Choose which data to use for further analysis
    data_to_analyze = intraday_data  # or daily_data, depending on the analysis you want to perform

    reversal = detect_potential_reversal(data_to_analyze)
    analyze = analyze_volume_trends(data_to_analyze)
    
    print(data_to_analyze)
    print(analyze)
    print(reversal)