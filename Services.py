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
from Data_Utils import get_option_chain, last_price_contract, get_risk_free_rate, get_ticker_from_contract, get_expiry, get_historical_options_data, get_data, time_to_maturity, strike_price, get_underlying_price, extract_option_type, get_nearest_expiry_and_strike_filtered_options, get_combined_option_chain
from Pricing_Utils import black_scholes, future_black_scholes_price, black_scholes_vectorized, monte_carlo_simulation, monte_carlo_option_price, mle_gbm, estimate_jump_parameters, jump_diffusion_simulation, jump_diffusion_option_price, price_my_option, ideal_contract_price_black_scholes
from Sentiment_Utils import visualize_net_institutional_trading_5_days, visualize_net_institutional_trading_today, calculate_net_institutional_trading, weighted_volume_sentiment_analysis, detect_volume_anomalies, highlight_key_info, time_aggregated_block_trades, get_intraday_stock_data, aggregate_subreddit_sentiment, weighted_reddit_sentiment_analysis, alpha_extract_and_calculate_sentiment, alpha_get_top_gainers_losers, alpha_get_news_sentiment
import plotly.graph_objects as go
import plotly.express as px

# Constants
ANNUAL_TRADING_DAYS = 252
RISK_FREE_TICKER = "^IRX"


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

if __name__ == '__main__':    
    print("Hello World")