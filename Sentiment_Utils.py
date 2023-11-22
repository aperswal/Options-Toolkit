import Services as ou
from alpha_vantage.timeseries import TimeSeries
import json
import requests
from datetime import datetime
import re
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
from Pricing_Utils import black_scholes, future_black_scholes_price, black_scholes_vectorized, monte_carlo_simulation, monte_carlo_option_price, mle_gbm, estimate_jump_parameters, jump_diffusion_simulation, jump_diffusion_option_price, price_my_option, ideal_contract_price_black_scholes
import praw
from textblob import TextBlob
import tweepy
from collections import deque, defaultdict

def alpha_load_api_key():
    with open('alpha_secret.json') as file:
        data = json.load(file)
        return data['key']
    
def reddit_load_api_key():
    with open('reddit_secret.json') as file:
        data = json.load(file)
        return data['client_id'], data['client_secret'], data['user_agent']

def load_twitter_api_keys():
    with open('twitter_secret.json') as file:
        data = json.load(file)
        return data['client_id'], data['client_secret'], data['app_id'], data['access_token'], data['access_token_secret']
    
alpha_api_key = alpha_load_api_key()
reddit_api_key = reddit_load_api_key()
twitter_api_key = load_twitter_api_keys()
net_institutional_trading = defaultdict(deque)


def alpha_get_news_sentiment(tickers=None, topics=None, time_from=None, time_to=None, sort='LATEST', limit=50):
    api_key = alpha_api_key
    url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT'
    params = {
        'apikey': api_key,
        'tickers': tickers,
        'topics': topics,
        'time_from': time_from,
        'time_to': time_to,
        'sort': sort,
        'limit': limit
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()  # Return the JSON data as a dictionary
    else:
        return f"Error fetching data: {response.status_code}"

def alpha_get_top_gainers_losers():
    api_key = alpha_api_key
    url = 'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS'
    params = {
        'apikey': api_key
    }
    response = requests.get(url, params=params)
    return response.json()
    
def alpha_extract_and_calculate_sentiment(ticker, response_dict):
    total_weighted_score = 0
    total_weight = 0

    # Access the news items from the 'feed' key of the response dictionary
    news_items = response_dict.get('feed', [])

    for item in news_items:
        ticker_sentiments = item.get("ticker_sentiment", [])

        for ticker_sentiment in ticker_sentiments:
            if ticker_sentiment["ticker"] == ticker:
                sentiment_score = float(ticker_sentiment["ticker_sentiment_score"])
                relevance_score = float(ticker_sentiment["relevance_score"])
                weight = relevance_score  # Use relevance score as weight

                total_weighted_score += sentiment_score * weight
                total_weight += weight

    if total_weight == 0:
        return "No relevant news found for the ticker."

    overall_score = total_weighted_score / total_weight

    # Determine sentiment label based on overall score
    sentiment_label = "Neutral"
    if overall_score <= -0.35:
        sentiment_label = "Bearish"
    elif -0.35 < overall_score <= -0.15:
        sentiment_label = "Somewhat-Bearish"
    elif 0.15 <= overall_score < 0.35:
        sentiment_label = "Somewhat-Bullish"
    elif overall_score >= 0.35:
        sentiment_label = "Bullish"

    return {"overall_sentiment_score": overall_score, "overall_sentiment_label": sentiment_label}

def weighted_reddit_sentiment_analysis(subreddit, ticker):
    client_id, client_secret, user_agent = reddit_load_api_key()
    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

    subreddit = reddit.subreddit(subreddit)

    total_weighted_sentiment = 0
    total_weight = 0
    six_months_ago = datetime.utcnow().timestamp() - (6 * 30 * 24 * 60 * 60)  # 6 months in seconds
    ticker = ticker.upper()  # Ensuring the ticker is in uppercase for consistency

    for post in subreddit.search(f"title:{ticker}", sort='new', time_filter='year', limit=10):
        if post.created_utc >= six_months_ago:
            post_age = (datetime.utcnow() - datetime.utcfromtimestamp(post.created_utc)).total_seconds()
            weight = (post.score + 1) / (post_age + 1) * (len(post.comments) + 1)
            analysis = TextBlob(post.title)
            sentiment_score = analysis.sentiment.polarity

            total_weighted_sentiment += sentiment_score * weight
            total_weight += weight

    average_weighted_sentiment = total_weighted_sentiment / total_weight if total_weight > 0 else 0
    return average_weighted_sentiment

def aggregate_subreddit_sentiment(subreddits, ticker):
    aggregated_sentiment = 0
    for subreddit in subreddits:
        sentiment = weighted_reddit_sentiment_analysis(subreddit, ticker)
        aggregated_sentiment += sentiment

    overall_sentiment = aggregated_sentiment / len(subreddits) if subreddits else 0
    sentiment_label = "Neutral"
    if overall_sentiment <= -0.35:
        sentiment_label = "Bearish"
    elif -0.35 < overall_sentiment <= -0.15:
        sentiment_label = "Somewhat-Bearish"
    elif 0.15 <= overall_sentiment < 0.35:
        sentiment_label = "Somewhat-Bullish"
    elif overall_sentiment >= 0.35:
        sentiment_label = "Bullish"

    return {"overall_sentiment_score": overall_sentiment, "overall_sentiment_label": sentiment_label}

'''
def twitter_sentiment_analysis(ticker):
    client_id, client_secret, app_id, access_token, access_token_secret = load_twitter_api_keys()
    auth = tweepy.OAuthHandler(client_id, client_secret)
    auth.set_access_token(access_token, access_token_secret)  # Add your Access Token Secret here
    api = tweepy.API(auth)

    ticker_variations = f"{ticker} OR ${ticker} OR #{ticker}"
    tweets = api.search_tweets(q=ticker_variations, count=10, result_type='mixed')
    total_sentiment = 0

    for tweet in tweets:
        analysis = TextBlob(tweet.text)
        total_sentiment += analysis.sentiment.polarity

    average_sentiment = total_sentiment / len(tweets) if tweets else 0

    sentiment_label = "Neutral"
    if average_sentiment <= -0.35:
        sentiment_label = "Bearish"
    elif -0.35 < average_sentiment <= -0.15:
        sentiment_label = "Somewhat-Bearish"
    elif 0.15 <= average_sentiment < 0.35:
        sentiment_label = "Somewhat-Bullish"
    elif average_sentiment >= 0.35:
        sentiment_label = "Bullish"

    return {"overall_sentiment_score": average_sentiment, "overall_sentiment_label": sentiment_label}

def aggregate_twitter_sentiment(accounts, ticker):
    aggregated_sentiment = 0
    for account in accounts:
        sentiment_data = twitter_sentiment_analysis(f"from:{account} {ticker}")
        aggregated_sentiment += sentiment_data['overall_sentiment_score']

    overall_sentiment = aggregated_sentiment / len(accounts) if accounts else 0

    # Labeling the sentiment
    sentiment_label = "Neutral"
    if overall_sentiment <= -0.35:
        sentiment_label = "Bearish"
    elif -0.35 < overall_sentiment <= -0.15:
        sentiment_label = "Somewhat-Bearish"
    elif 0.15 <= overall_sentiment < 0.35:
        sentiment_label = "Somewhat-Bullish"
    elif overall_sentiment >= 0.35:
        sentiment_label = "Bullish"

    return {"overall_sentiment_score": overall_sentiment, "overall_sentiment_label": sentiment_label}

def sentiment_analysis_for_stock_trading_twitter(ticker):
    accounts = ['@TradingThomas3', '@traders_kings']
    stock_sentiment = aggregate_twitter_sentiment(accounts, ticker)
    return stock_sentiment
'''

def get_intraday_stock_data(symbol):
    ts = TimeSeries(key=alpha_api_key, output_format='pandas')
    data, _ = ts.get_intraday(symbol=symbol, interval='5min', outputsize='full')
    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']  # Rename columns
    return data

def time_aggregated_block_trades(data, time_window='1min', block_size=10000):
    # Resampling data to the desired time frame and summing volumes
    aggregated_data = data.resample(time_window).agg({'Open': 'first', 
                                                      'Close': 'last', 
                                                      'Volume': 'sum'})
    
    # Identifying large volume trades and their price impact
    block_trades = aggregated_data[aggregated_data['Volume'] >= block_size]
    block_trades['Price Impact'] = block_trades['Close'] - block_trades['Open']
    
    return block_trades

def detect_volume_anomalies(data, std_factor=3):
    # Using standard deviation to find significant deviations in volume
    avg_volume = data['Volume'].mean()
    std_volume = data['Volume'].std()
    threshold = avg_volume + std_factor * std_volume

    volume_anomalies = data[(data['Volume'] > threshold) | (data['Volume'] < avg_volume - std_factor * std_volume)]
    return volume_anomalies

def highlight_key_info(data):
    # Extracting and returning key details
    return data[['Open', 'Close', 'Volume']]

def weighted_volume_sentiment_analysis(data):
    # Assigning weights based on volume and price impact
    data['Weight'] = data['Volume'] * abs(data['Close'] - data['Open'])
    data['Sentiment'] = data.apply(lambda row: 1 if row['Close'] > row['Open'] else -1, axis=1)
    weighted_sentiment = sum(data['Weight'] * data['Sentiment']) / sum(data['Weight'])

    # Categorizing the sentiment
    if weighted_sentiment > 0:
        return "Bullish"
    elif weighted_sentiment < 0:
        return "Bearish"
    else:
        return "Neutral"


def calculate_net_institutional_trading(block_trades, date, ticker):
    # Filter the block trades for the given date
    block_trades = block_trades[block_trades.index.date == date]
    
    # Calculate the total bought and sold for each day
    block_trades['Net'] = block_trades['Close'] - block_trades['Open']
    block_trades['Institutional Trading'] = block_trades['Net'] * block_trades['Volume']
    
    # Calculate the net institutional trading for each day
    net = block_trades['Institutional Trading'].sum()
    
    # Add the net institutional trading for the day to the dictionary
    net_institutional_trading[ticker].append((date, net))

def visualize_net_institutional_trading_today():
    # Create a bar graph of the net institutional trading for each ticker for today
    tickers, trading = zip(*[(ticker, net_institutional_trading[ticker][-1][1]) for ticker in symbols])
    colors = ['g' if x > 0 else 'r' for x in trading]
    plt.bar(tickers, trading, color=colors)
    plt.xlabel('Ticker')
    plt.ylabel('Net Institutional Trading')
    plt.title('Net Institutional Trading for Today')
    plt.show()

def visualize_net_institutional_trading_5_days():
    # Create a bar graph of the net institutional trading for each ticker for the most recent 5 days
    tickers, trading = zip(*[(ticker, sum([x[1] for x in net_institutional_trading[ticker]])) for ticker in symbols])
    colors = ['g' if x > 0 else 'r' for x in trading]
    plt.bar(tickers, trading, color=colors)
    plt.xlabel('Ticker')
    plt.ylabel('Net Institutional Trading')
    plt.title('Net Institutional Trading for the Most Recent 5 Days')
    plt.show()


symbols = ['SPY', 'MSFT', 'AAPL', 'AMZN', 'NVDA', 'GOOGL', 'META', 'GOOG', 'BRK-B', 'TSLA', 'UNH']

if __name__ == '__main__':
    for ticker in symbols:
        intraday_data = get_intraday_stock_data(ticker)

        # Ensure the index is a datetime for resampling
        intraday_data.index = pd.to_datetime(intraday_data.index)
        
        # Get the block trades
        block_trades = time_aggregated_block_trades(intraday_data, '1min')
        
        # Calculate the net institutional trading
        calculate_net_institutional_trading(block_trades, pd.to_datetime('today').date(), ticker)
    
    # Visualize the net institutional trading for today
    visualize_net_institutional_trading_today()
    
    # Visualize the net institutional trading for the most recent 5 days
    visualize_net_institutional_trading_5_days()