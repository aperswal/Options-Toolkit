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

if __name__ == '__main__':
    symbol = 'MSFT'
    print("SMA: ", calculate_sma(symbol))
    print("EMA: ", calculate_ema(symbol))
    print("RSI: ", calculate_rsi(symbol))
    print("Golden Cross: ", calculate_golden_cross(symbol))
    print("Death Cross: ", calculate_death_cross(symbol))
    print("RSI Overbought/Oversold: ", calculate_rsi_overbought_oversold(symbol))
    
