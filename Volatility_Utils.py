# Standard Libraries
import math
import re
from datetime import datetime, timedelta

# Third-party Libraries
import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf
import yoptions as yo
from sklearn.linear_model import LinearRegression
from scipy.signal import argrelextrema
import seaborn as sns
import holidays
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def get_ticker_from_contract(contract_name):
    # Extract the ticker symbol from the option contract name
    match = re.search(r'^([A-Z]+)', contract_name)
    if match:
        return match.group(1)
    else:
        raise ValueError("Couldn't extract ticker from contract name.")

# Update the get_historical_volatility_of_contract function
def get_historical_volatility_of_contract(contract_name):
    # Extract ticker symbol from the contract name
    ticker_symbol = get_ticker_from_contract(contract_name)

    # Check and proceed only if ticker_symbol is valid
    if isinstance(ticker_symbol, str):
        annual_volatility = historical_volatility(ticker_symbol, period='1y', interval='1d')
        return annual_volatility
    else:
        print(f"Invalid ticker symbol: {ticker_symbol}")
        return None

def black_scholes_volatility(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the Black-Scholes price of a European option.

    Parameters:
    S (float): Current price of the underlying asset.
    K (float): Strike price of the option.
    T (float): Time to maturity (in years).
    r (float): Risk-free interest rate (annual rate, expressed as a decimal).
    sigma (float): Volatility of the underlying asset's returns (annualized).
    option_type (str): Type of the option ('call' for a call option, 'put' for a put option).

    Returns:
    float: The Black-Scholes price of the option.
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    option_type = option_type.lower()
    if option_type == 'call' or option_type =='c':
        option_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put' or option_type =='p':
        option_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return option_price

def sabr_volatility(F, K, T, alpha, beta, rho, nu):
    X = K
    if F == K: 
        numer1 = (((1 - beta)**2) * alpha**2) / (24 * F**(2 - 2 * beta))
        numer2 = 0.25 * rho * beta * nu * alpha / (F**(1 - beta))
        numer3 = ((2 - 3 * rho**2) * nu**2) / 24
        VolAtm = alpha * (1 + (numer1 + numer2 + numer3) * T) / (F**(1 - beta))
        return VolAtm
    else:
        z = nu / alpha * (F * X)**((1 - beta) / 2) * np.log(F / X)
        x = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
        numer1 = (((1 - beta)**2) * alpha**2) / (24 * (F * X)**(1 - beta))
        numer2 = 0.25 * rho * beta * nu * alpha / ((F * X)**((1 - beta) / 2))
        numer3 = ((2 - 3 * rho**2) * nu**2) / 24
        numer = alpha * (1 + (numer1 + numer2 + numer3) * T) * z
        denom = ((F * X)**((1 - beta) / 2)) * (1 - beta) * x
        return numer / denom
    
def get_implied_volatility(contract_name):
    contract_df = yo.get_plain_option_ticker(option_ticker=contract_name)    
    implied_volatility = contract_df['Impl. Volatility'].iloc[0]
    return implied_volatility

def historical_volatility(stock_ticker, period='1y', interval='1d'):
    """
    Calculate the historical volatility for a given stock.

    :param stock_ticker: Ticker symbol of the stock (e.g., 'AAPL').
    :param period: The period over which to retrieve historical data. Default is '1y' (1 year).
    :param interval: The interval between data points. Default is '1d' (1 day).
    :return: Annualized historical volatility.
    """

    # Retrieve historical data for the stock
    stock_data = yf.download(stock_ticker, period=period, interval=interval)

    # Ensure the data is sorted by date in ascending order
    stock_data = stock_data.sort_index()

    # Calculate daily returns
    stock_data['Returns'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))

    # Calculate the standard deviation of daily returns
    daily_volatility = stock_data['Returns'].std()

    # Annualize the volatility
    annual_volatility = daily_volatility * np.sqrt(252)

    return annual_volatility

def vega(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return S * norm.pdf(d1) * math.sqrt(T)

def derived_implied_volatility(S, K, T, r, market_price, option_type, tol=1e-6, max_iterations=100):
    sigma = 0.5  # Initial guess
    for _ in range(max_iterations):
        price = black_scholes_volatility(S, K, T, r, sigma, option_type)
        vega_value = vega(S, K, T, r, sigma)

        price_diff = market_price - price
        if abs(price_diff) < tol:
            return sigma

        sigma += price_diff / vega_value  # Newton-Raphson update

        if sigma <= 0:  # Ensure sigma doesn't go negative
            return None

    return None

def get_ticker_volatility(ticker, period='1y', interval='1d'):
    """
    Calculate the historical volatility for a given stock ticker.

    Parameters:
    ticker (str): The stock ticker symbol.
    period (str): Period for historical data (default is '1y' for one year).
    interval (str): Data interval (default is '1d' for daily data).

    Returns:
    float: The annualized historical volatility.
    """
    # Download historical data for the ticker
    stock_data = yf.download(ticker, period=period, interval=interval)

    # Calculate daily log returns
    log_returns = np.log(stock_data['Close'] / stock_data['Close'].shift(1))

    # Calculate the standard deviation of daily log returns (volatility)
    daily_volatility = log_returns.std()

    # Annualize the volatility
    annualized_volatility = daily_volatility * np.sqrt(252)

    return annualized_volatility

if __name__ == '__main__':    
    print("Hello World")
