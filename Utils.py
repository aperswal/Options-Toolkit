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

# Constants
ANNUAL_TRADING_DAYS = 252
RISK_FREE_TICKER = "^IRX"

def black_scholes(S, K, T, r, sigma, option_type='call'):
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

def get_historical_data(contract_name):
    return yo.get_historical_option_ticker(option_ticker = contract_name)

def time_to_maturity(contract_name):
    expiry = get_expiry(contract_name)
    expiry_date = datetime.strptime(expiry, '%Y-%m-%d').date()
    time_to_maturity = expiry_date - datetime.today().date()
    return time_to_maturity.days

def strike_price(contract_name):
    contract_df = yo.get_plain_option_ticker(option_ticker=contract_name)
    strike = contract_df['Strike'].iloc[0]
    return strike

def get_underlying_price(contract_name):
    return yo.get_underlying_price(option_ticker = contract_name)

def get_implied_volatility(contract_name):
    contract_df = yo.get_plain_option_ticker(option_ticker=contract_name)    
    implied_volatility = contract_df['Impl. Volatility'].iloc[0]
    return implied_volatility

def extract_option_type(contract_name):
    # The regex pattern looks for a sequence of 6 digits (the date) 
    # followed by either 'C' or 'P' (the option type).
    match = re.search(r'\d{6}(C|P)', contract_name)
    if match:
        return match.group(1)  # Return the matched option type ('C' or 'P')
    else:
        raise ValueError(f"Couldn't extract option type from: {contract_name}")

def get_historical_volatility(contract_name):
    data = get_historical_data(contract_name)
    
    # Ensure the data is sorted by date in ascending order
    data = data.sort_values(by='Date', ascending=True)
    
    # Calculate daily returns
    data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Calculate the standard deviation of daily returns
    daily_volatility = data['Returns'].std()
    
    # Annualize the volatility
    annual_volatility = daily_volatility * np.sqrt(252)
    
    return annual_volatility

def ideal_contract_price(contract_name):
    # Extract all necessary parameters from the contract name and other functions
    S = get_underlying_price(contract_name)  # Current price of the underlying asset
    K = strike_price(contract_name)  # Strike price
    T = time_to_maturity(contract_name) / 365  # Time to maturity in years
    r = get_risk_free_rate() / 100  # Convert interest rate to decimal form
    sigma = get_historical_volatility(contract_name)  # Historical volatility
    option_type = 'call' if extract_option_type(contract_name) == 'C' else 'put'  # Option type
    
    # Calculate the Black-Scholes price
    price = black_scholes(S, K, T, r, sigma, option_type)
    return price


def get_ideal_contract(ticker, expected_price, expected_date, days_after_target=7):
    """
    Determine the ideal option contract.

    Parameters:
    - ticker (str): Ticker symbol of the underlying stock.
    - expected_price (float): Expected price of the underlying stock.
    - expected_date (str): Expected date when the stock might hit the target price. Format: 'YYYY-MM-DD'.
    - days_after_target (int, optional): Number of days after the target date to consider. Defaults to 30.

    Returns:
    pd.Series: Details of the ideal contract.
    """
    
    # Determine the option type based on the expected price vs current price
    current_price = get_underlying_price(ticker)
    option_type = 'c' if expected_price > current_price else 'p'
    
    # Define price range for filtering
    upper_limit = expected_price * 1.1  # 10% above expected price
    lower_limit = expected_price * 0.9  # 10% below expected price
    
    # Get all available options for the given ticker and option type
    option_chain = yo.get_chain_greeks(stock_ticker=ticker, dividend_yield=0, option_type=option_type)
    
    # Filter based on strike price range
    option_chain = option_chain[(option_chain['Strike'] <= upper_limit) & (option_chain['Strike'] >= lower_limit)]
    
    # Extract expiration dates and filter based on expected_date and days_after_target
    option_chain['Expiration'] = option_chain['Symbol'].apply(get_expiry)
    option_chain['Expiration'] = pd.to_datetime(option_chain['Expiration'])
    cutoff_date = pd.to_datetime(expected_date) + timedelta(days=days_after_target)
    option_chain = option_chain[(option_chain['Expiration'] >= expected_date) & (option_chain['Expiration'] <= cutoff_date)]
    
    # For each option, compute the difference between its Black-Scholes price and its market price
    option_chain['Difference'] = option_chain.apply(
        lambda row: ideal_contract_price(row['Symbol']) - row['Last Price'],
        axis=1
    )
    
    # If it's a call option, we want the option where the Black-Scholes price is most above the market price
    # If it's a put option, we want the option where the Black-Scholes price is most below the market price
    if option_type == 'c':
        ideal_contract = option_chain[option_chain['Difference'] == option_chain['Difference'].max()]
    else:
        ideal_contract = option_chain[option_chain['Difference'] == option_chain['Difference'].min()]
    
    # Return the details of the ideal contract
    return ideal_contract.iloc[0]  # Assuming we want to return the first contract if there are ties

def profitability_range(contract_name, expected_low_price, expected_high_price):
    profitability = {}
    
    # Extracting parameters
    r = get_risk_free_rate() / 100  # Convert interest rate to decimal form
    sigma = get_historical_volatility(contract_name)  # Historical volatility
    option_type = 'call' if extract_option_type(contract_name) == 'C' else 'put'  # Option type
    K = strike_price(contract_name)  # Strike price
    
    # Get the date range from today until the expiration of the contract
    start_date = datetime.today().date()
    expiry_date = datetime.strptime(get_expiry(contract_name), '%Y-%m-%d').date()
    date_range = [start_date + timedelta(days=i) for i in range((expiry_date - start_date).days + 1)]
    
    # For each price in the range, compute the price of the option using Black-Scholes
    for price in range(int(expected_low_price), int(expected_high_price) + 1):  # +1 to make the range inclusive
        profitability[price] = []
        for date in date_range:
            T = (expiry_date - date).days / 365  # Time to maturity in years
            bs_price = black_scholes(price, K, T, r, sigma, option_type)  # Black-Scholes price
            profitability[price].append(bs_price)
    
    # Convert dictionary to DataFrame for better presentation
    profitability_df = pd.DataFrame(profitability, index=date_range)
    return profitability_df

def get_nearest_expiry_and_strike_filtered_options(ticker):
    """
    Get options for a given ticker that are with the nearest expiry 
    and a strike price within 5% of the current stock price.
    """
    current_price = get_underlying_price(ticker)
    
    # Define the 5% range for filtering
    upper_limit = current_price * 1.05  # 5% above current price
    lower_limit = current_price * 0.95  # 5% below current price
    
    # Fetch call options
    call_options = yo.get_chain_greeks(stock_ticker=ticker, dividend_yield=0, option_type='c', risk_free_rate=None)
    
    # Fetch put options
    put_options = yo.get_chain_greeks(stock_ticker=ticker, dividend_yield=0, option_type='p', risk_free_rate=None)
    
    # Combine the call and put options
    option_chain = pd.concat([call_options, put_options])
    
    # Filter based on strike price range
    option_chain = option_chain[(option_chain['Strike'] <= upper_limit) & (option_chain['Strike'] >= lower_limit)]
    
    # Filter to only keep the nearest expiration
    option_chain['Expiration'] = option_chain['Symbol'].apply(get_expiry)
    option_chain['Expiration'] = pd.to_datetime(option_chain['Expiration'])
    nearest_expiry = option_chain['Expiration'].min()
    option_chain = option_chain[option_chain['Expiration'] == nearest_expiry]
    
    return option_chain


def under_valued_contracts(ticker):
    """
    Identify option contracts for a given ticker that are priced below their ideal Black-Scholes price.
    """
    option_chain = get_nearest_expiry_and_strike_filtered_options(ticker)
    
    # For each option, compute the difference between its Black-Scholes price and its market price
    option_chain['Difference'] = option_chain.apply(
        lambda row: ideal_contract_price(row['Symbol']) - row['Last Price'],
        axis=1
    )
    
    # Filter the options that are priced below the Black-Scholes price (positive difference)
    undervalued_options = option_chain[option_chain['Difference'] > 0]
    
    return undervalued_options

def over_valued_contracts(ticker):
    """
    Identify option contracts for a given ticker that are priced above their ideal Black-Scholes price.
    """
    option_chain = get_nearest_expiry_and_strike_filtered_options(ticker)
    
    # For each option, compute the difference between its Black-Scholes price and its market price
    option_chain['Difference'] = option_chain.apply(
        lambda row: ideal_contract_price(row['Symbol']) - row['Last Price'],
        axis=1
    )
    
    # Filter the options that are priced above the Black-Scholes price (negative difference)
    overvalued_options = option_chain[option_chain['Difference'] < 0]
    
    return overvalued_options

# TODO: Monte Carlo Simulation
def monte_carlo_simulation(S, r, sigma, T, n_simulations, dt):
    """
    Simulate the stock price paths using Geometric Brownian Motion.
    
    Parameters:
    - S: Initial stock price.
    - r: Risk-free interest rate.
    - sigma: Stock price volatility.
    - T: Time to maturity in years.
    - n_simulations: Number of simulation paths.
    - dt: Time increment (e.g., 1/252 for daily increments).
    
    Returns:
    ndarray: A 2D array where each column is a simulated stock price path.
    """
    # TODO: Implement the simulation
    pass

def monte_carlo_option_price(S, K, T, r, sigma, option_type='call', n_simulations=10000):
    """
    Calculate the option price using Monte Carlo simulations.
    
    Parameters:
    ... [same as black_scholes]
    - n_simulations: Number of simulation paths.
    
    Returns:
    float: The estimated option price.
    """
    # TODO: Use the monte_carlo_simulation function and average the results to get the option price.
    pass

# TODO: Jump Diffusion Simulation
def jump_diffusion_simulation(S, r, sigma, T, lam, mu, delta, n_simulations, dt):
    """
    Simulate the stock price paths using the Merton Jump Diffusion model.
    
    Parameters:
    - S: Initial stock price.
    - r: Risk-free interest rate.
    - sigma: Stock price volatility.
    - T: Time to maturity in years.
    - lam: Expected number of jumps per year.
    - mu: Expected percentage jump size.
    - delta: Standard deviation of percentage jump size.
    - n_simulations: Number of simulation paths.
    - dt: Time increment (e.g., 1/252 for daily increments).
    
    Returns:
    ndarray: A 2D array where each column is a simulated stock price path.
    """
    # TODO: Implement the jump diffusion simulation.
    pass

def jump_diffusion_option_price(S, K, T, r, sigma, lam, mu, delta, option_type='call', n_simulations=10000):
    """
    Calculate the option price using jump diffusion simulations.
    
    Parameters:
    ... [some parameters as monte_carlo_option_price, plus the jump diffusion ones]
    
    Returns:
    float: The estimated option price.
    """
    # TODO: Use the jump_diffusion_simulation function and average the results to get the option price.
    pass


if __name__ == '__main__':    
    '''
    print(get_expiry('BA231027C00200000'))
    print(black_scholes(100, 95, 1, 0.05, 0.2, 'call'))
    print(time_to_maturity('BA231027C00200000'))
    print(strike_price('BA231027C00200000'))
    print(get_ticker_from_contract('BA231027C00200000'))
    print(get_underlying_price('BA231027C00200000'))
    print(get_implied_volatility('BA231027C00200000'))
    print(extract_option_type('BA231027C00200000'))
    print(get_risk_free_rate())
    print(get_historical_volatility('BA231027C00200000'))
    print(ideal_contract_price('BA231027C00200000'))
    print(get_ideal_contract('BA', 200, '2023-10-26'))
    print(profitability_range('BA231027C00200000', 170, 190))   
    print(under_valued_contracts('BA'))
    print(over_valued_contracts('BA'))
    '''