import Services as ou
from alpha_vantage.timeseries import TimeSeries
import json
import requests
from datetime import datetime
# Standard Libraries
import math
import re
from datetime import datetime, timedelta

# Third-party Libraries
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.signal import argrelextrema
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import praw
from textblob import TextBlob
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
    ticker_variations = [ticker.upper(), f"${ticker.upper()}", f"#{ticker.upper()}"]
    search_query = '|'.join(ticker_variations)  # Creates a regex pattern like 'SPY|$SPY|#SPY'

    for post in subreddit.search(f"title:({search_query})", sort='new', time_filter='year', limit=10):
        if post.created_utc >= six_months_ago:
            post_age = (datetime.utcnow() - datetime.utcfromtimestamp(post.created_utc)).total_seconds()
            post_weight = (post.score + 1) / (post_age + 1) * (len(post.comments) + 1)
            post_analysis = TextBlob(post.title)
            post_sentiment_score = post_analysis.sentiment.polarity

            total_weighted_sentiment += post_sentiment_score * post_weight
            total_weight += post_weight

            # Analyze top 10 comments
            post.comments.replace_more(limit=0)  # Load the comments
            for comment in post.comments[:10]:  # Take top 10 comments
                comment_analysis = TextBlob(comment.body)
                comment_sentiment_score = comment_analysis.sentiment.polarity
                comment_weight = post_weight * 0.5  # Assuming comment weight is half of post weight

                total_weighted_sentiment += comment_sentiment_score * comment_weight
                total_weight += comment_weight

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
    subreddits = ['wallstreetbets', 'daytrading', 'options', 'stocks']
    ticker = 'UPST'
    print(aggregate_subreddit_sentiment(subreddits, ticker))

    