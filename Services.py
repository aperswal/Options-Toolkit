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
from Volatility_Utils import get_implied_volatility, historical_volatility, sabr_volatility, get_historical_volatility_of_contract, derived_implied_volatility, vega, get_ticker_volatility
from Data_Utils import get_option_chain, last_price_contract, get_risk_free_rate, get_ticker_from_contract, get_expiry, get_historical_options_data, get_data, time_to_maturity, strike_price, get_underlying_price, extract_option_type, get_nearest_expiry_and_strike_filtered_options, get_combined_option_chain, get_current_ticker_price
from Pricing_Utils import black_scholes, future_black_scholes_price, black_scholes_vectorized, monte_carlo_simulation, monte_carlo_option_price, mle_gbm, estimate_jump_parameters, jump_diffusion_simulation, jump_diffusion_option_price, price_my_option, ideal_contract_price_black_scholes
import plotly.graph_objects as go
import plotly.express as px
from Technical_Analysis_Utils import calculate_sma, calculate_rsi, calculate_ema, analyze_volume_trends, detect_potential_reversal, get_stock_data_intraday, get_stock_data_daily
from Sentiment_Utils import weighted_reddit_sentiment_analysis, time_aggregated_block_trades, calculate_net_institutional_trading, get_intraday_stock_data
from collections import defaultdict, deque

# Constants
ANNUAL_TRADING_DAYS = 252
RISK_FREE_TICKER = "^IRX"

def max_pain(ticker):
    current_price = get_underlying_price(ticker)

    # Fetch call and put options data
    call_options = yo.get_plain_chain(ticker, 'c')
    put_options = yo.get_plain_chain(ticker, 'p')

    # Extract expiration dates from option contract names
    call_options['Expiration'] = call_options['Symbol'].apply(get_expiry)
    put_options['Expiration'] = put_options['Symbol'].apply(get_expiry)

    # Assuming the DataFrame has 'Strike' and 'Open Interest' columns
    strike_prices = pd.concat([call_options['Strike'], put_options['Strike']]).unique()
    max_pain_strike = 0
    max_pain_value = 0
    total_money_lost = 0

    # Calculate the pain for each strike price
    for strike in strike_prices:
        call_pain = sum((strike - current_price) * call_options[call_options['Strike'] == strike]['Open Interest'].fillna(0))
        put_pain = sum((current_price - strike) * put_options[put_options['Strike'] == strike]['Open Interest'].fillna(0))
        total_pain = call_pain + put_pain

        if total_pain > max_pain_value:
            max_pain_value = total_pain
            max_pain_strike = strike
            total_money_lost = total_pain

    expiration_date = call_options['Expiration'].iloc[0]

    result = {
        "ticker": ticker,
        "max_pain_strike": max_pain_strike,
        "max_pain_value": max_pain_value,
        "total_money_lost": total_money_lost,
        "expiration_date": expiration_date,
        "current_price": current_price
    }
    return result

def visualize_max_pain(max_pain_info, call_options, put_options):
    ticker = max_pain_info['ticker']
    max_pain_strike = max_pain_info['max_pain_strike']
    current_price = max_pain_info['current_price']

    plt.figure(figsize=(10, 6))
    plt.plot(call_options['Strike'], call_options['Open Interest'], label='Calls Open Interest', color='green')
    plt.plot(put_options['Strike'], put_options['Open Interest'], label='Puts Open Interest', color='red')
    plt.axvline(x=max_pain_strike, label='Max Pain Strike', color='blue', linestyle='--')
    plt.axvline(x=current_price, label='Current Price', color='black', linestyle=':')

    plt.title(f"Options Open Interest for {ticker} (Max Pain at {max_pain_strike})")
    plt.xlabel("Strike Price")
    plt.ylabel("Open Interest")
    plt.legend()
    plt.grid(True)
    plt.show()

def over_under_priced_contracts_by_volatility(contract_name):
    contract_name_str = str(contract_name)  # Ensure contract_name is a string
    implied_volatility = get_implied_volatility(contract_name_str) * 100
    historical_volatility = get_historical_volatility_of_contract(contract_name_str) * 100
    iv_premium = abs(implied_volatility - historical_volatility)

    return "underpriced" if iv_premium < (implied_volatility * 0.68) else "overpriced"

def derive_implied_volatility_contract(contract_name):
    S = get_underlying_price(contract_name)  # Underlying price
    K = strike_price(contract_name)  # Strike price
    T = time_to_maturity(contract_name) / 365 # Time to maturity in years
    r = get_risk_free_rate() / 100  # Convert interest rate to decimal form
    market_price = last_price_contract(contract_name)  # Market price
    option_type = 'call' if extract_option_type(contract_name) == 'C' else 'put'  # Option type
    implied_volatility = derived_implied_volatility(S, K, T, r, market_price, option_type)
    
    return implied_volatility

def max_profit_contract(ticker, expected_price, expected_date, days_after_target=3, dividend_yield=0, risk_free_rate=None):
    # Get current and expected prices
    current_price = get_underlying_price(ticker)
    # Determine the option type
    option_type = 'c' if expected_price > current_price else 'p'
    # Define price range for filtering
    upper_limit = expected_price * 1.03
    lower_limit = expected_price * 0.97
    # Get combined option chain data
    cutoff_date = pd.to_datetime(expected_date) + pd.Timedelta(days=days_after_target)
    combined_option_chain = get_combined_option_chain(ticker, dividend_yield, option_type, expected_date, cutoff_date.strftime('%Y-%m-%d'), risk_free_rate)
    # Add a new column for 'Type' by applying the extract_option_type function
    combined_option_chain['Type'] = combined_option_chain['Symbol'].apply(extract_option_type)

    # Filter based on strike price range
    combined_option_chain = combined_option_chain[(combined_option_chain['Strike'] <= upper_limit) & (combined_option_chain['Strike'] >= lower_limit)]

    if combined_option_chain.empty:
        print("No suitable options found after filtering by strike price.")
        return None

    # Pre-calculate common values to avoid repeated function calls
    T = (pd.to_datetime(get_expiry(combined_option_chain['Symbol'].iloc[0])) - pd.to_datetime(expected_date)).days / 365
    r = get_risk_free_rate() / 100  # Convert interest rate to decimal form

    # Vectorized calculations for future Black-Scholes prices
    combined_option_chain['Future_BS_Price'] = combined_option_chain.apply(
        lambda row: black_scholes(expected_price, row['Strike'], T, r, row['Impl. Volatility'], row['Type']),
        axis=1
    )

    # Calculate potential profit percentage vectorized
    combined_option_chain['Potential_Profit_Percentage'] = ((combined_option_chain['Future_BS_Price'] - combined_option_chain['Last Price']) / combined_option_chain['Last Price']) * 100

    # Select the ideal contract with the highest potential profit percentage
    ideal_contract = combined_option_chain.loc[combined_option_chain['Potential_Profit_Percentage'].idxmax()]

    print("Ideal contract:")
    print(ideal_contract)

    return ideal_contract

def avg_contract_price_with_all_models(contract_name):
    monte_carlo = price_my_option(contract_name, 'monte_carlo')
    jump_diffusion = price_my_option(contract_name, 'jump_diffusion')
    black_scholes = price_my_option(contract_name, 'black_scholes')
    price = (monte_carlo + jump_diffusion + black_scholes) / 3
    return price

def profitability_range(contract_name, expected_low_price, expected_high_price):
    profitability = {}
    S = get_underlying_price(contract_name)  # Current price of the underlying asset
    K = strike_price(contract_name)  # Strike price
    T = time_to_maturity(contract_name) / 365  # Time to maturity in years
    r = get_risk_free_rate() / 100  # Risk-free rate
    sigma = get_implied_volatility(contract_name)  # Implied volatility
    option_type = 'call' if extract_option_type(contract_name) == 'C' else 'put'  # Option type

    # Calculate the Black-Scholes price for the current underlying price
    current_bs_price = black_scholes(S, K, T, r, sigma, option_type)

    # Get the actual last traded price of the option
    actual_last_price = last_price_contract(contract_name)

    # Calculate the percentage difference
    percentage_diff = (actual_last_price - current_bs_price) / current_bs_price

    # Get the date range from today until the expiration of the contract
    start_date = datetime.today().date()
    expiry_date = datetime.strptime(get_expiry(contract_name), '%Y-%m-%d').date()
    date_range = pd.bdate_range(start=start_date, end=expiry_date, freq='C', holidays=holidays.US())

    # Adjust the expected price range to be centered around the current price
    mid_price = np.round(S * 2) / 2  # Round to the nearest 0.5
    low_price = mid_price - (expected_high_price - expected_low_price) / 2
    high_price = mid_price + (expected_high_price - expected_low_price) / 2

    # For each price in the range, compute the adjusted price of the option
    for price in np.arange(low_price, high_price + 0.5, 0.5):
        profitability[price] = []
        for date in date_range:
            if isinstance(date, pd.Timestamp):
                date = date.to_pydatetime().date()
            T = (expiry_date - date).days / 365  # Recalculate time to maturity
            bs_price = black_scholes(price, K, T, r, sigma, option_type)
            percentage_change = (bs_price - current_bs_price) / current_bs_price  # Calculate the percentage change
            adjusted_price = actual_last_price * (1 + percentage_change)  # Apply the percentage change to the actual last traded price
            profitability[price].append(adjusted_price)

    # Convert to DataFrame
    profitability_df = pd.DataFrame(profitability, index=date_range)

    # Fill missing values by interpolation
    for date in date_range:
        for price in np.arange(low_price + 0.5, high_price, 0.5):
            if np.isnan(profitability_df.loc[date, price]):
                profitability_df.loc[date, price] = (profitability_df.loc[date, price - 0.5] + profitability_df.loc[date, price + 0.5]) / 2

    return profitability_df

def profitability_heatmap(contract_name, profitability_range):
    # Get the current price of the contract
    current_price = last_price_contract(contract_name)

    # Convert the profitability range to a DataFrame if it's not already one
    if not isinstance(profitability_range, pd.DataFrame):
        df = pd.DataFrame(profitability_range)
    else:
        df = profitability_range

    # Ensure the index is recognized as dates
    df.index = pd.to_datetime(df.index).strftime('%m-%d-%Y')

    # Normalize the data around the current price for color mapping
    normalized_df = df.subtract(current_price)

    # Calculate percentage change for hover information
    percentage_change = normalized_df.divide(current_price).multiply(100)

    # Find the maximum and minimum percentage changes
    max_change = percentage_change.max().max()
    min_change = percentage_change.min().min()

    # Create a symmetric color scale around zero
    color_scale = px.colors.diverging.RdYlGn  # Use a diverging color scale from green to red

    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=normalized_df.values,
        x=normalized_df.columns,
        y=normalized_df.index,
        colorscale=color_scale,
        hoverongaps=False,
        text=df.applymap(lambda x: f"${x:.2f}"),  # Displaying the price on the boxes
        hovertext=percentage_change.applymap(lambda x: f"{x:.2f}%"),  # Hover shows percentage change
        texttemplate="%{text}",  # Use text for display
        hovertemplate="<b>Price:</b> %{x}<br><b>Date:</b> %{y}<br><b>Change:</b> %{hovertext}<extra></extra>",
        colorbar=dict(
            tickvals=[min_change, 0, max_change],
            ticktext=['-100%', '0%', '100%'],
            title='Change'
        )
    ))

    contract_type = extract_option_type(contract_name)
    if contract_type == "C":
        contract_type = "Call"
    else:
        contract_type = "Put"
    ticker = get_ticker_from_contract(contract_name)
    strike = strike_price(contract_name)

    # Update layout for grid and other properties
    fig.update_layout(
        title=f'Profitability Heatmap for a {contract_type} contract on {ticker} at strike price {strike}<br>Range: {df.columns.min()} - {df.columns.max()}',
        xaxis_title='Price Range',
        yaxis_title='Date',
        xaxis=dict(showgrid=True, dtick=1),  # Show grid and set dtick for x-axis
        yaxis=dict(autorange='reversed', showgrid=True),  # Reverse y-axis for chronological order and show grid
        plot_bgcolor='rgba(0,0,0,0)',  # Set the background color to transparent
        paper_bgcolor='rgba(0,0,0,0)',  # Set the paper background color to transparent
        font=dict(size=12, color='black')  # Set the font size and color
    )

    # Show the plot
    fig.show()

def evaluate_contracts(tickers):
    underpriced = {'Call': [], 'Put': []}
    overpriced = {'Call': [], 'Put': []}

    for ticker in tickers:
        filtered_options = get_nearest_expiry_and_strike_filtered_options(ticker)
        
        for _, contract_data in filtered_options.iterrows():
            contract_name = contract_data['Symbol']  # Extract the contract name
            option_type = extract_option_type(contract_name).lower()  # Extract the option type (c or p)

            pricing_status = over_under_priced_contracts_by_volatility(contract_name)

            if pricing_status == 'underpriced':
                underpriced['Call' if option_type == 'c' else 'Put'].append(contract_name)
            elif pricing_status == 'overpriced':
                overpriced['Call' if option_type == 'c' else 'Put'].append(contract_name)

    return pd.DataFrame({'Underpriced': underpriced, 'Overpriced': overpriced})

def market_mispriced_contracts_finder():
    tickers = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'GOOGL', 'NFLX', 'BA']  
    contract_analysis = evaluate_contracts(tickers)
    return contract_analysis

def predict_market_direction(ticker, time_frame='daily', time_period=20):
    """
    Predict the market direction based on technical analysis for a given ticker.
    Chooses between intraday and daily data based on the time_frame.

    Parameters:
    ticker (str): The stock ticker symbol.
    time_frame (str): Time frame for the analysis ('1min', '5min', '15min', '30min', '60min', 'daily').
    time_period (int): Time period to consider for moving averages and RSI.
    """
    # Define valid intraday intervals
    valid_intraday_intervals = ['1min', '5min', '15min', '30min', '60min']
    
    # Check if the time_frame is for intraday or daily data
    if time_frame in valid_intraday_intervals:
        # Use intraday data
        stock_data = get_stock_data_intraday(ticker, time_frame)
        interval = time_frame  # Interval is the same as time_frame for intraday data
    else:
        # Use daily data
        stock_data = get_stock_data_daily(ticker)
        interval = 'daily'  # For daily data, the interval is always 'daily'

    # Max pain analysis
    max_pain_info = max_pain(ticker)
    max_pain_strike = max_pain_info['max_pain_strike']
    current_price = get_current_ticker_price(ticker)

    # Fetch the last value from the SMA, EMA, and RSI data series
    sma_last_value = calculate_sma(ticker, interval=interval, time_period=time_period)['SMA'].iloc[-1]
    ema_last_value = calculate_ema(ticker, interval=interval, time_period=time_period)['EMA'].iloc[-1]
    rsi_last_value = calculate_rsi(ticker, interval=interval, time_period=time_period)['RSI'].iloc[-1]

    bullish_signals = 0
    bearish_signals = 0

    # Comparing current price with SMA and EMA
    if current_price > sma_last_value:
        bullish_signals += 1
    else:
        bearish_signals += 1

    if current_price > ema_last_value:
        bullish_signals += 1.5
    else:
        bearish_signals += 1.5

    # RSI analysis
    if rsi_last_value > 70:
        bearish_signals += 0.5
    elif rsi_last_value < 30:
        bullish_signals += 0.5

    # Max pain analysis
    if max_pain_strike > current_price:
        bullish_signals += 2
    elif max_pain_strike < current_price:
        bearish_signals += 2

    # Volume trend analysis
    volume_trend_analysis = analyze_volume_trends(stock_data)
    if "upward trend" in volume_trend_analysis:
        bullish_signals += 2
    elif "downward trend" in volume_trend_analysis:
        bearish_signals += 2

    # Potential reversal analysis
    reversal_data = detect_potential_reversal(stock_data)
    if reversal_data['potential_reversal'].iloc[-1]:
        # A potential reversal can indicate either bullish or bearish, depending on the current trend
        if bullish_signals > bearish_signals:
            bearish_signals += 1  # Adds weight to bearish if currently bullish
        else:
            bullish_signals += 1  # Adds weight to bullish if currently bearish

    # Determining the market direction based on the signals
    if bullish_signals > bearish_signals:
        market_direction = "Bullish"
    elif bearish_signals > bullish_signals:
        market_direction = "Bearish"
    else:
        market_direction = "Neutral"

    return market_direction

def comprehensive_stock_analysis_with_prediction(ticker, prediction_timeframe=30):
    print(f"Starting analysis for {ticker}...")
    net_institutional_trading = defaultdict(deque)

    print("Performing technical analysis...")
    technical_analysis_data = {}
    for interval in ['1min', '5min']:
        sma_data = calculate_sma(ticker, interval, 20, 'close')
        ema_data = calculate_ema(ticker, interval, 20, 'close')
        rsi_data = calculate_rsi(ticker, interval, 14, 'close')
        technical_analysis_data[interval] = {'SMA': sma_data, 'EMA': ema_data, 'RSI': rsi_data}
        print(f"Technical analysis for {interval} interval complete.")

    print("Performing Reddit sentiment analysis...")
    subreddits = ['wallstreetbets', 'stocks', 'investing']
    sentiment_scores = [weighted_reddit_sentiment_analysis(subreddit, ticker) for subreddit in subreddits]
    average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
    print("Reddit sentiment analysis complete.")

    print("Performing max pain analysis...")
    max_pain_info = max_pain(ticker)
    print("Max pain analysis complete.")

    print("Analyzing net institutional trading...")
    stock_data = get_intraday_stock_data(ticker)
    block_trades = time_aggregated_block_trades(stock_data, '5min', 10000)
    current_date = datetime.now().date()
    calculate_net_institutional_trading(block_trades, current_date, ticker, net_institutional_trading)
    net_institutional_trading_value = net_institutional_trading[ticker][-1] if ticker in net_institutional_trading else (current_date, 0)
    print("Net institutional trading analysis complete.")

    market_direction = predict_market_direction(ticker, '5min', 10)

    current_price = get_current_ticker_price(ticker)
    predicted_price_change = current_price * 0.01 * (1 if market_direction == "Bullish" else -1)
    predicted_price_change += predicted_price_change * average_sentiment
    predicted_price = current_price + predicted_price_change
    probability = np.clip(50 + (average_sentiment * 10) + (10 if market_direction == "Bullish" else -10), 0, 100)

    print("Assembling the report...")
    report = {
        'ticker': ticker,
        'technical_analysis': technical_analysis_data,
        'reddit_sentiment': average_sentiment,
        'max_pain': max_pain_info,
        'net_institutional_trading': net_institutional_trading_value,
        'prediction': {
            'predicted_price': predicted_price,
            'probability': probability,
            'timeframe_minutes': prediction_timeframe
        }
    }

    print("Analysis complete.")
    return report

if __name__ == '__main__':
    print(comprehensive_stock_analysis_with_prediction('AAPL'))