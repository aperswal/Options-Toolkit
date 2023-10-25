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

def exponential_moving_average(ticker, n=50, period='5m'):
    """
    Calculate exponential moving average.

    Parameters:
    ticker (str): Stock ticker symbol.
    n (int): Number of days over which to calculate EMA.
    period (str): Time frame for the EMA. Default is '5m' for 5 minutes.

    Returns:
    Series: EMA values.
    """
    data = get_data(ticker, period)
    ema = data['Close'].ewm(span=n, adjust=False).mean()
    return ema

def price_rate_of_change(ticker, n=14):
    """
    Calculate Price Rate of Change.

    Parameters:
    ticker (str): Stock ticker symbol.
    n (int): Number of days over which to calculate ROC.

    Returns:
    Series: ROC values.
    """
    data = get_data(ticker)
    roc = ((data['Close'] - data['Close'].shift(n)) / data['Close'].shift(n)) * 100
    return roc

def moving_average(ticker, n=50):
    """
    Calculate simple moving average.

    Parameters:
    ticker (str): Stock ticker symbol.
    n (int): Number of days over which to calculate SMA.

    Returns:
    Series: SMA values.
    """
    data = get_data(ticker)
    sma = data['Close'].rolling(window=n).mean()
    return sma

def vwap(ticker):
    """
    Calculate Volume Weighted Average Price (VWAP).

    Parameters:
    ticker (str): Stock ticker symbol.

    Returns:
    Series: VWAP values.
    """
    data = get_data(ticker)
    vwap = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    return vwap

def simple_moving_forecast(ticker, n=3):
    """
    Predict the next period's closing price using Simple Moving Average.

    Parameters:
    ticker (str): Stock ticker symbol.
    n (int): Number of periods to consider for SMA.

    Returns:
    float: Predicted closing price for the next period.
    """
    sma = moving_average(ticker, n)
    return sma.iloc[-1]

def linear_regression_forecast(ticker, n=10):
    """
    Predict the next period's closing price using Linear Regression.

    Parameters:
    ticker (str): Stock ticker symbol.
    n (int): Number of periods to consider for Linear Regression.

    Returns:
    float: Predicted closing price for the next period.
    """
    data = get_data(ticker, period='5d')

    last_n_days = data.iloc[-n:]

    # Preparing data for Linear Regression
    X = np.array(range(len(last_n_days))).reshape(-1, 1)
    y = last_n_days['Close'].values

    # Training the model
    model = LinearRegression().fit(X, y)

    # Predicting next period's closing price
    next_period = np.array([n]).reshape(-1, 1)
    predicted_close = model.predict(next_period)[0]
    
    return predicted_close

def relative_strength_index(ticker, n=14):
    """
    Calculate Relative Strength Index (RSI).

    Parameters:
    ticker (str): Stock ticker symbol.
    n (int): Number of days over which to calculate RSI.

    Returns:
    Series: RSI values.
    """
    data = get_data(ticker)
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=n, min_periods=1).mean()
    avg_loss = loss.rolling(window=n, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def bollinger_bands(ticker, n=20, std_dev=2):
    """
    Calculate Bollinger Bands.

    Parameters:
    ticker (str): Stock ticker symbol.
    n (int): Number of days over which to calculate the moving average.
    std_dev (int): Number of standard deviations to determine the upper and lower bands.

    Returns:
    DataFrame: Upper Band, SMA and Lower Band.
    """
    data = get_data(ticker)
    sma = data['Close'].rolling(window=n).mean()
    rolling_std = data['Close'].rolling(window=n).std()
    upper_band = sma + (rolling_std * std_dev)
    lower_band = sma - (rolling_std * std_dev)
    return pd.DataFrame({'Upper Band': upper_band, 'SMA': sma, 'Lower Band': lower_band})

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
    T = time_to_maturity(contract_name) / 365
    r = get_risk_free_rate() / 100
    sigma = get_historical_volatility(contract_name)
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
    
def avg_contract_price_with_all_models(contract_name):
    monte_carlo = price_my_option(contract_name, 'monte_carlo')
    jump_diffusion = price_my_option(contract_name, 'jump_diffusion')
    black_scholes = price_my_option(contract_name, 'black_scholes')
    price = (monte_carlo + jump_diffusion + black_scholes) / 3
    return price

def decide_trade(ticker):
    """
    Decide whether to buy a call or put option based on technical indicators.

    Parameters:
    ticker (str): Stock ticker symbol.

    Returns:
    str: 'call' if bullish sentiment, 'put' if bearish sentiment, 'neutral' if indicators are mixed.
    """
    data = get_data(ticker, '5m')
    roc_value = price_rate_of_change(ticker, n=3).iloc[-1]  # Adjusted for 5min timeframe
    sma_value = moving_average(ticker, n=10).iloc[-1]  # Adjusted for 5min timeframe
    vwap_value = vwap(ticker).iloc[-1]
    rsi_value = relative_strength_index(ticker, n=3).iloc[-1]  # Adjusted for 5min timeframe
    bollinger = bollinger_bands(ticker, n=4).iloc[-1]  # Adjusted for 5min timeframe
    close_value = data['Close'].iloc[-1]
    ema_27 = exponential_moving_average(ticker, 27).iloc[-1]
    ema_56 = exponential_moving_average(ticker, 56).iloc[-1]
    ema_108 = exponential_moving_average(ticker, 108).iloc[-1]

    bullish_indicators = 0
    bearish_indicators = 0

    # ROC
    if roc_value > 0:
        bullish_indicators += 1
    else:
        bearish_indicators += 1

    # SMA
    if close_value > sma_value:
        bullish_indicators += 1
    else:
        bearish_indicators += 1

    # VWAP
    if close_value > vwap_value:
        bullish_indicators += 1
    else:
        bearish_indicators += 1

    # RSI
    if rsi_value < 30:
        bullish_indicators += 1
    elif rsi_value > 70:
        bearish_indicators += 1

    # Bollinger Bands
    if close_value > bollinger['Upper Band']:
        bullish_indicators += 1
    elif close_value < bollinger['Lower Band']:
        bearish_indicators += 1

    # EMA cross
    if ema_27 > ema_56 and ema_27 > ema_108:
        bullish_indicators += 1
    elif ema_27 < ema_56 or ema_27 < ema_108:
        bearish_indicators += 1

    if bullish_indicators >= 4:
        return 'call'
    elif bearish_indicators >= 4:
        return 'put'
    else:
        return 'neutral'

def forecast_from_indicator(ticker, indicator_function):
    """
    Calculate the forecast using a specific indicator's method.
    """
    if indicator_function == bollinger_bands:
        current_value = indicator_function(ticker)["SMA"].iloc[-1]
    else:
        current_value = indicator_function(ticker).iloc[-1]

    data = get_data(ticker, period='5m')
    last_value = data['Close'].iloc[-1]

    # Forecast is a projection of the difference
    return last_value + (current_value - last_value)

def combined_forecast(ticker):
    """
    Calculate the combined forecast based on sentiment and various forecasting methods.

    Parameters:
    ticker (str): Stock ticker symbol.

    Returns:
    float: Combined forecasted price.
    """
    sentiment = decide_trade(ticker)
    
    # Define indicator functions and their initial weights
    indicators = [price_rate_of_change, moving_average, vwap, relative_strength_index, bollinger_bands]
    weights = [0.3, 0.2, 0.3, 0.15, 0.05]  # Initial weights (can be adjusted based on sentiment)
    
    forecasts = []
    for i, indicator_function in enumerate(indicators):
        forecast_value = forecast_from_indicator(ticker, indicator_function)
        forecasts.append(forecast_value)
        if (sentiment == 'call' and forecast_value > get_data(ticker, period='5m')['Close'].iloc[-1]) or \
           (sentiment == 'put' and forecast_value < get_data(ticker, period='5m')['Close'].iloc[-1]):
            weights[i] += 0.1  # Increase weight by 10% if the forecast aligns with sentiment
    
    # Normalize the weights to ensure they sum up to 1
    total_weight = sum(weights)
    normalized_weights = [w/total_weight for w in weights]
    
    # Combine forecasts based on weights
    weighted_forecast_sum = sum([forecasts[i] * normalized_weights[i] for i in range(len(forecasts))])
    
    # Add in the linear regression and SMA forecast
    lr_forecast = linear_regression_forecast(ticker)
    sma_forecast = simple_moving_forecast(ticker)
    
    combined_forecast = (weighted_forecast_sum + lr_forecast + sma_forecast) / 3

    # Apply light exponential smoothing
    alpha = 0.1  # A small value for light smoothing
    data = get_data(ticker, period='5m')
    last_close = data['Close'].iloc[-1]
    smoothed_forecast = alpha * combined_forecast + (1 - alpha) * last_close

    return smoothed_forecast

def ideal_0DTE_contract(ticker, sentiment, expected_move):
    """
    Select the ideal 0DTE option contract based on sentiment and expected price move.

    Parameters:
    - ticker (str): Stock ticker symbol.
    - sentiment (str): Either 'call' or 'put'.
    - expected_move (float): Expected price move in the underlying.

    Returns:
    str: The ideal 0DTE contract name.
    """
    # Ensure sentiment is valid
    if sentiment not in ['call', 'put']:
        raise ValueError("Sentiment should be either 'call' or 'put'.")

    # Get the current price of the underlying
    underlying_price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[0]

    # Determine the expected price after the move
    target_price = underlying_price + expected_move if sentiment == 'call' else underlying_price - expected_move
    if sentiment == 'call':
        sentiment == 'c'
    elif sentiment == 'put':
        sentiment == 'p'
    # Get the options chain
    option_chain = yo.get_chain_greeks(stock_ticker=ticker, dividend_yield=0, option_type=sentiment)

    # Add columns for Expiry and Option Type using get_expiry and extract_option_type functions
    option_chain['Expiry'] = option_chain['Symbol'].apply(get_expiry)
    option_chain['OptionType'] = option_chain['Symbol'].apply(extract_option_type)

    # Filter based on 0DTE and sentiment
    expiry_date = datetime.today().strftime('%Y-%m-%d')
    filtered_options = option_chain[
        (option_chain['Expiry'] == expiry_date) &
        (option_chain['OptionType'].str.lower() == sentiment)
    ]

    if filtered_options.empty:
        raise ValueError("No suitable options found for the given criteria.")

    # Sort based on closeness to target price and then by liquidity (volume and open interest)
    filtered_options['Distance'] = abs(filtered_options['Strike'] - target_price)
    sorted_options = filtered_options.sort_values(by=['Distance', 'Volume', 'Open Interest'], ascending=[True, False, False])

    # Return the first (most suitable) option's ticker
    return sorted_options['Symbol'].iloc[0]

# TODO Zero DTE Gambler
def zero_dte_options(ticker):
    #sentiment = decide_trade(ticker)
    sentiment = 'call'
    expected_price = combined_forecast(ticker)
    
    if sentiment == 'call':
        return ideal_0DTE_contract(ticker, sentiment, expected_price)
    elif sentiment == 'put':
        return ideal_0DTE_contract(ticker, sentiment, expected_price)
    else:
        return "Neutral"

# TODO Zero DTE Casino
def setup_iron_condor(ticker, dividend_yield=0, risk_free_rate=None, distance=1, width=5):
    # Get the current price of the underlying asset
    underlying_price = get_underlying_price(ticker)
    
    # Get option chains for calls and puts
    calls = yo.get_chain_greeks(stock_ticker=ticker, dividend_yield=dividend_yield, option_type='c', risk_free_rate=risk_free_rate)
    puts = yo.get_chain_greeks(stock_ticker=ticker, dividend_yield=dividend_yield, option_type='p', risk_free_rate=risk_free_rate)
    
    # Filter out-of-the-money options based on underlying price
    otm_calls = calls[calls['Strike'] > underlying_price]
    otm_puts = puts[puts['Strike'] < underlying_price]
    
    # Select the call and put options to sell (closest to the money but still OTM)
    sell_call = otm_calls.iloc[distance - 1]
    sell_put = otm_puts.iloc[distance - 1]
    
    # Select the call and put options to buy (further out-of-the-money)
    buy_call = otm_calls.iloc[distance - 1 + width]
    buy_put = otm_puts.iloc[distance - 1 - width]
    
    # Calculate net premium received
    net_premium = (sell_call['Bid'] + sell_put['Bid'] - buy_call['Ask'] - buy_put['Ask'])
    
    # Calculate max potential loss (Width of the strikes minus net premium received)
    max_loss = width - net_premium
    
    return {
        "Sell Call": sell_call['Symbol'],
        "Buy Call": buy_call['Symbol'],
        "Sell Put": sell_put['Symbol'],
        "Buy Put": buy_put['Symbol'],
        "Net Premium": net_premium,
        "Max Loss": max_loss
    }

if __name__ == '__main__':    
    '''
    print("Expiry of contract BA231027C00200000:", get_expiry('BA231027C00200000'))
    print("Black-Scholes option price:", black_scholes(100, 95, 1, 0.05, 0.2, 'call'))
    print("Time to maturity of contract BA231027C00200000:", time_to_maturity('BA231027C00200000'))
    print("Strike price of contract BA231027C00200000:", strike_price('BA231027C00200000'))
    print("Ticker from contract BA231027C00200000:", get_ticker_from_contract('BA231027C00200000'))
    print("Underlying price of contract BA231027C00200000:", get_underlying_price('BA231027C00200000'))
    print("Implied volatility of contract BA231027C00200000:", get_implied_volatility('BA231027C00200000'))
    print("Option type of contract BA231027C00200000:", extract_option_type('BA231027C00200000'))
    print("Current risk-free rate:", get_risk_free_rate())
    print("Historical volatility of contract BA231027C00200000:", get_historical_volatility('BA231027C00200000'))
    print("Ideal contract price for BA231027C00200000:", ideal_contract_price('BA231027C00200000'))
    print("Ideal contract for BA with strike 200 and expiry '2023-10-26':", get_ideal_contract('BA', 200, '2023-10-26'))
    print("Profitability range for contract BA231027C00200000 between 170 and 190:", profitability_range('BA231027C00200000', 170, 190))
    print("Undervalued contracts for BA:", under_valued_contracts('BA'))
    print("Overvalued contracts for BA:", over_valued_contracts('BA'))
    
    # For monte_carlo_simulation
    sample_paths = monte_carlo_simulation(100, 0.05, 0.2, 1, 1000, 1/252)
    print("First 5 simulated paths from monte_carlo_simulation:", sample_paths[:5])

    # For monte_carlo_option_price
    mc_price = monte_carlo_option_price(100, 95, 1, 0.05, 0.2, 'call')
    print("Option price using Monte Carlo simulation:", mc_price)

    # For mle_gbm
    mu, sigma = mle_gbm('BA')
    print(f"MLE GBM parameters for BA - Mu: {mu}, Sigma: {sigma}")

    # For estimate_jump_parameters
    lam, mu_j, delta_j = estimate_jump_parameters('BA')
    print(f"Jump parameters for BA - Lambda: {lam}, Mu: {mu_j}, Delta: {delta_j}")

    # For jump_diffusion_simulation
    sample_jump_paths = jump_diffusion_simulation(100, 0.05, 0.2, 1, lam, mu_j, delta_j, 1000, 1/252)
    print("First 5 simulated paths from jump_diffusion_simulation:", sample_jump_paths[:5])

    # For jump_diffusion_option_price
    jd_price = jump_diffusion_option_price(100, 95, 1, 0.05, 0.2, lam, mu_j, delta_j)
    print("Option price using Jump Diffusion simulation:", jd_price)
    
    # For Black-Scholes model
    option_price_bs = price_my_option('BA231027C00200000', 'black_scholes')
    print(f"Option price for BA231027C00200000 using Black-Scholes model: {option_price_bs}")

    # For Monte Carlo simulation
    option_price_mc = price_my_option('BA231027C00200000', 'monte_carlo')
    print(f"Option price for BA231027C00200000 using Monte Carlo simulation: {option_price_mc}")
    
    # For Jump Diffusion simulation
    option_price_jd = price_my_option('BA231027C00200000', 'jump_diffusion')
    print(f"Option price for BA231027C00200000 using Jump Diffusion simulation: {option_price_jd}")
    
    print(avg_contract_price_with_all_models('BA231027C00200000'))
    
    print("EMA for BA:", exponential_moving_average('BA').iloc[-1])
    print("Price Rate of Change for BA:", price_rate_of_change('BA').iloc[-1])
    print("Simple Moving Average for BA:", moving_average('BA').iloc[-1])
    print("VWAP for BA:", vwap('BA').iloc[-1])
    print("Simple Moving Forecast for BA:", simple_moving_forecast('BA'))
    print("Relative Strength Index for BA:", relative_strength_index('BA').iloc[-1])
    print("Bollinger Bands for BA:", bollinger_bands('BA').iloc[-1])
    print("Trade Decision for BA:", decide_trade('BA'))

    
    print("Linear Regression Forecast for BA:", linear_regression_forecast('BA'))

    # Using forecast_from_indicator with 'moving_average' as an example. You can replace with other indicators.

    print("Forecast from moving_average for BA:", forecast_from_indicator('BA', moving_average))
    
    print("Combined Forecast for SPY:", combined_forecast('SPY'))
    '''
    # print(zero_dte_options('SPY'))
    # print(setup_iron_condor('SPY'))