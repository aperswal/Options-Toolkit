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
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

def get_option_chain(ticker, dividend_yield, option_type, expiration_date, risk_free_rate=None):
    option_data = yo.get_chain_greeks_date(
        stock_ticker=ticker,
        dividend_yield=dividend_yield,
        option_type=option_type,
        expiration_date=expiration_date,
        risk_free_rate=risk_free_rate
    )
    return option_data

def last_price_contract(contract_name):
    option_data = yo.get_option_greeks_ticker(option_ticker=contract_name, dividend_yield=0.04)
    return option_data['Last Price'].iloc[0]

def get_risk_free_rate():
    tbill = yf.Ticker("^IRX")  # 3-month T-bill rate
    hist = tbill.history(period="1d")
    rate = hist['Close'].iloc[0]
    return rate

def get_ticker_from_contract(contract_name):
    match = re.match(r'([A-Z]+)', contract_name)
    if match:
        return match.group(1)
    else:
        raise ValueError("Couldn't extract ticker from contract name.")

def get_expiry(contract_name):
    """
    Get the expiry date of an option contract.

    Parameters:
    contract_name (str): The name of the option contract.

    Returns:
    str: The expiry date of the option contract in 'YYYY-MM-DD' format.
    """
    # Use a regular expression to find the first occurrence of a six-digit sequence
    # This assumes the expiry date is the first six-digit sequence in the contract name
    match = re.search(r'(\d{6})', contract_name)
    if match:
        expiry_str = match.group(1)  # Get the matched six-digit sequence
        # Reformat the string to 'YYYY-MM-DD' format
        expiry_date = datetime.strptime(expiry_str, '%y%m%d').strftime('%Y-%m-%d')
        return expiry_date
    else:
        raise ValueError(f'Unable to extract expiry date from contract name: {contract_name}')

def get_historical_options_data(contract_name):
    return yo.get_historical_option_ticker(option_ticker = contract_name)

def get_data(ticker, period='1y'):
    """
    Download historical data for the given ticker using yfinance.

    Parameters:
    ticker (str): Stock ticker symbol.
    period (str): Duration for which data needs to be fetched. Default is '1y' for one year.

    Returns:
    DataFrame: Historical data.
    """
    data = yf.download(ticker, period=period)
    return data

def time_to_maturity(contract_name):
    expiry = get_expiry(contract_name)
    expiry_date = datetime.strptime(expiry, '%Y-%m-%d').date()
    time_to_maturity = expiry_date - datetime.today().date()
    return time_to_maturity.days

def strike_price(contract_name):
    contract_df = yo.get_plain_option_ticker(option_ticker=contract_name)
    strike = contract_df['Strike'].iloc[0]
    return float(strike)

def get_underlying_price(contract_name):
    return yo.get_underlying_price(option_ticker = contract_name)

def extract_option_type(contract_name):
    # The regex pattern looks for a sequence of 6 digits (the date) 
    # followed by either 'C' or 'P' (the option type).
    match = re.search(r'\d{6}(C|P)', contract_name)
    if match:
        return match.group(1)  # Return the matched option type ('C' or 'P')
    else:
        raise ValueError(f"Couldn't extract option type from: {contract_name}")

def get_nearest_expiry_and_strike_filtered_options(ticker):
    current_price = get_underlying_price(ticker)
    upper_limit = current_price * 1.05  # 5% above current price
    lower_limit = current_price * 0.95  # 5% below current price

    # Fetch call options
    call_options = yo.get_chain_greeks(stock_ticker=ticker, dividend_yield=0, option_type='c', risk_free_rate=None)

    # Fetch put options
    put_options = yo.get_chain_greeks(stock_ticker=ticker, dividend_yield=0, option_type='p', risk_free_rate=None)

    # Combine and filter options
    combined_options = pd.concat([call_options, put_options])
    filtered_options = combined_options[(combined_options['Strike'] >= lower_limit) & 
                                        (combined_options['Strike'] <= upper_limit)]

    return filtered_options

def get_combined_option_chain(ticker, dividend_yield, option_type, start_date, end_date, risk_free_rate=None):
    # Generate a list of dates between start_date and end_date
    date_range = pd.date_range(start=start_date, end=end_date)

    # Initialize an empty DataFrame to hold the combined data
    combined_option_chain = pd.DataFrame()

    for date in date_range:
        # Fetch the option chain for each date
        option_chain = get_option_chain(ticker, dividend_yield, option_type, date.strftime('%Y-%m-%d'), risk_free_rate)
        
        # Combine the data
        combined_option_chain = pd.concat([combined_option_chain, option_chain], ignore_index=True)

    return combined_option_chain

if __name__ == '__main__':    
    print(yo.get_plain_option_ticker('SPY231124C00457000'))