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
from Volatility_Utils import get_implied_volatility, historical_volatility, sabr_volatility, get_historical_volatility_of_contract, derived_implied_volatility, vega
from Data_Utils import get_option_chain, last_price_contract, get_risk_free_rate, get_ticker_from_contract, get_expiry, get_historical_options_data, get_data, time_to_maturity, strike_price, get_underlying_price, extract_option_type


def ideal_contract_price_black_scholes(contract_name):
    # Extract all necessary parameters from the contract name and other functions
    S = get_underlying_price(contract_name)  # Current price of the underlying asset
    K = strike_price(contract_name)  # Strike price
    T = time_to_maturity(contract_name) / 365 # Time to maturity in years
    r = get_risk_free_rate() / 100  # Convert interest rate to decimal form
    sigma = get_implied_volatility(contract_name)  # Implied volatility
    option_type = 'call' if extract_option_type(contract_name) == 'C' else 'put'  # Option type
    
    # Calculate the Black-Scholes price
    price = black_scholes(S, K, T, r, sigma, option_type)
    return price

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

def future_black_scholes_price(contract_name, future_price):
    # Extract all necessary parameters from the contract name and other functions
    S = future_price  # Use the future price of the underlying asset
    K = strike_price(contract_name)  # Strike price
    T = time_to_maturity(contract_name) / 252 # Time to maturity in years
    r = get_risk_free_rate() / 100  # Convert interest rate to decimal form
    sigma = get_implied_volatility(contract_name)  # Implied volatility
    option_type = 'call' if extract_option_type(contract_name) == 'C' else 'put'  # Option type
    
    # Calculate the Black-Scholes price
    price = black_scholes(S, K, T, r, sigma, option_type)
    return price

def black_scholes_vectorized(S, K, T, r, sigma, option_type):
    """
    Vectorized Black-Scholes calculation.
    S, K, T, r, sigma, and option_type are all pandas Series or similar iterable types.
    """
    # Compute d1 and d2 using vectorized operations
    # Implement the Black-Scholes formula vectorized for all inputs
    # Return a Series of calculated Black-Scholes prices

    # Example (simplified):
    d1 = np.log(S / K) + (r + 0.5 * sigma ** 2) * T
    d1 /= sigma * np.sqrt(T)
    d2 = d1 - sigma * np.sqrt(T)

    # Calculate call and put prices
    call_prices = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_prices = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return np.where(option_type == 'call', call_prices, put_prices)

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
    
    # Number of time steps
    n_steps = int(T/dt)
    
    # Initialize stock price paths matrix
    paths = np.zeros((n_steps + 1, n_simulations))
    paths[0] = S
    
    # Simulate paths
    for t in range(1, n_steps + 1):
        # Brownian motion increment
        rand = np.random.randn(n_simulations)
        # Update stock price using GBM formula
        paths[t] = paths[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rand)
    
    return paths

def monte_carlo_option_price(S, K, T, r, sigma, option_type='call', n_simulations=10000):
    """
    Calculate the option price using Monte Carlo simulations.
    
    Parameters:
    ... [same as black_scholes]
    - n_simulations: Number of simulation paths.
    
    Returns:
    float: The estimated option price.
    """
    dt = T / 252  # Daily increment
    paths = monte_carlo_simulation(S, r, sigma, T, n_simulations, dt)
    if option_type.lower() in ['call', 'c']:
        payoffs = np.maximum(paths[-1] - K, 0)
    elif option_type.lower() in ['put', 'p']:
        payoffs = np.maximum(K - paths[-1], 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    option_price = np.mean(payoffs) * np.exp(-r * T)
    return option_price

def mle_gbm(ticker):
    """
    Maximum likelihood estimation for GBM parameters (mu and delta).
    
    Parameters:
    - ticker: Stock ticker symbol
    
    Returns:
    - mu, sigma: GBM parameters
    """
    # Define the date range
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=18*30)).strftime('%Y-%m-%d')  # Approximate 18 months back

    # Fetch stock data
    data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
    
    # Calculate daily log returns
    log_returns = np.log(data / data.shift(1))
    
    # Drop NaN values from log returns
    log_returns = log_returns.dropna()
    
    # Estimate mu and sigma
    mu = log_returns.mean()
    sigma = log_returns.std()
    
    return mu, sigma

def estimate_jump_parameters(ticker):
    """
    Estimate jump parameters (lambda, mu, and delta) using historical data.
    
    Parameters:
    - data: Stock price data
    
    Returns:
    - lam, mu, delta: Jump diffusion parameters
    """
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=18*30)).strftime('%Y-%m-%d')  # Approximate 18 months back
    
    # Fetch stock data
    data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
    
    # Calculate daily log returns
    log_returns = np.log(data / data.shift(1))
    
    # Drop NaN values from log returns
    log_returns = log_returns.dropna()
    
    # Detect jumps (for illustration, using 3 standard deviations as threshold)
    threshold = 3 * log_returns.std()
    jumps = log_returns[np.abs(log_returns) > threshold]
    
    # Estimate jump parameters
    lam = len(jumps) / len(log_returns)
    mu_j = jumps.mean()
    delta_j = jumps.std()
    
    return lam, mu_j, delta_j

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
    n_steps = int(T/dt)
    paths = np.zeros((n_steps, n_simulations))
    
    paths[0] = S
    
    for t in range(1, n_steps):
        # GBM component
        Z = np.random.normal(size=n_simulations)
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        
        # Jump component
        N = np.random.poisson(lam * dt, n_simulations)  # Number of jumps
        Y = np.random.normal(mu, delta, n_simulations)  # Jump size
        jump = N * Y
        
        # Combine GBM and jump components
        paths[t] = paths[t-1] * np.exp(drift + diffusion + jump)
    
    return paths

def jump_diffusion_option_price(S, K, T, r, sigma, lam, mu, delta, option_type='call', n_simulations=10000):
    """
    Calculate the option price using jump diffusion simulations.
    
    Parameters:
    ... [some parameters as monte_carlo_option_price, plus the jump diffusion ones]
    
    Returns:
    float: The estimated option price.
    """
    dt = T / 252  # Daily increment
    paths = jump_diffusion_simulation(S, r, sigma, T, lam, mu, delta, n_simulations, dt)
    if option_type.lower() in ['call', 'c']:
        payoffs = np.maximum(paths[-1] - K, 0)
    elif option_type.lower() in ['put', 'p']:
        payoffs = np.maximum(K - paths[-1], 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    option_price = np.mean(payoffs) * np.exp(-r * T)
    return option_price

def price_my_option(contract_name, model):
    """
    Price an option using either the Black-Scholes model, the Monte Carlo simulation, 
    or the Jump Diffusion simulation.
    """
    # Extracting parameters
    S = get_underlying_price(contract_name)
    K = strike_price(contract_name)
    T = time_to_maturity(contract_name) / 252
    r = get_risk_free_rate() / 100
    sigma = get_implied_volatility(contract_name)     
    option_type = 'call' if extract_option_type(contract_name) == 'C' else 'put'
    
    # Using the chosen model to price the option
    if model == "black_scholes":
        return black_scholes(S, K, T, r, sigma, option_type)
    elif model == "monte_carlo":
        return monte_carlo_option_price(S, K, T, r, sigma, option_type)
    elif model == "jump_diffusion":
        ticker = get_ticker_from_contract(contract_name)
        lam, mu, delta = estimate_jump_parameters(ticker)
        delta = sigma / 2  # This is just a placeholder. You may need to refine this value.
        return jump_diffusion_option_price(S, K, T, r, sigma, lam, mu, delta, option_type)
    else:
        raise ValueError(f"Unknown model: {model}")
    
if __name__ == '__main__':    
    print(get_risk_free_rate())