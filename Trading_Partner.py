import math
import re
from datetime import datetime, timedelta
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
import plotly.express as px
import json
import openai

from Volatility_Utils import *
from Data_Utils import *
from Pricing_Utils import *
from Services import *
from Sentiment_Utils import *
from Technical_Analysis_Utils import *

def openai_api_key():
    with open('openai_secret.json') as file:
        data = json.load(file)
        return data['key']

class OptionsModule:

    def evaluate_option_strategy(self, strategy_details):
        ticker = strategy_details.get("ticker")
        expected_price = strategy_details.get("expected_price")
        expected_date = strategy_details.get("expected_date")

        ideal_contract = max_profit_contract(ticker, expected_price, expected_date)

        return f"The ideal contract for your strategy: {ideal_contract}"

class TechnicalAnalysisModule:

    def analyze_technical_indicators(self, ticker):
        market_direction = predict_market_direction(ticker)
        sma_data = calculate_sma(ticker)
        ema_data = calculate_ema(ticker)
        rsi_data = calculate_rsi(ticker)

        analysis = f"Market Direction: {market_direction}\nSMA: {sma_data}\nEMA: {ema_data}\nRSI: {rsi_data}"
        return analysis
    

class SentimentAnalysisModule:

    def analyze_market_sentiment(self, ticker):
        news_sentiment = alpha_get_news_sentiment(tickers=ticker)
        news_analysis = f"News Sentiment for {ticker}: {news_sentiment}"

        reddit_sentiment = weighted_reddit_sentiment_analysis('wallstreetbets', ticker)
        reddit_analysis = f"Reddit Sentiment for {ticker}: {reddit_sentiment}"

        return news_analysis + "\n" + reddit_analysis

class TradingPartner:
    def __init__(self):
        self.options_module = OptionsModule()
        self.technical_analysis_module = TechnicalAnalysisModule()
        self.sentiment_analysis_module = SentimentAnalysisModule()
        self.gpt_key = openai_api_key()
        self.openai_client = openai.OpenAI(api_key=self.gpt_key)

    def interpret_query(self, query):
        try:
            chat_completion = self.openai_client.chat.completions.create(
                messages=[{"role": "user", "content": query}],
                model="gpt-3.5-turbo"
            )
            interpreted_query = chat_completion.choices[0].message.content.strip()
            return interpreted_query
        except Exception as e:
            print(f"Error in interpreting query: {e}")
            return "error"
    
    def handle_query(self, query):
        interpreted_query = self.interpret_query(query)
        if interpreted_query == "error":
            return "I'm sorry, I couldn't understand your query."

        if "bullish" in interpreted_query.lower() or "bearish" in interpreted_query.lower():
            ticker = self.extract_ticker(query)
            time_frame = self.extract_time_frame(query)
            if not ticker or not time_frame:
                return "Please specify the stock ticker and time frame for the analysis."
            trend_analysis = self.technical_analysis_module.analyze_market_trend(ticker, time_frame)
            return trend_analysis
        
        if "options" in interpreted_query.lower():
            ticker = self.extract_ticker(query)
            expected_price, expected_date = self.extract_strategy_details(query)
            if not ticker or expected_price is None or expected_date is None:
                return "Please provide complete information for the options contract."
            strategy_details = {
                "ticker": ticker,
                "expected_price": expected_price,
                "expected_date": expected_date
            }
            return self.options_module.evaluate_option_strategy(strategy_details)
        elif "technical" in interpreted_query.lower():
            ticker = self.extract_ticker(query)
            if not ticker:
                return "Please specify the stock ticker for technical analysis."
            return self.technical_analysis_module.analyze_technical_indicators(ticker)
        elif "sentiment" in interpreted_query.lower():
            ticker = self.extract_ticker(query)
            if not ticker:
                return "Please specify the stock ticker for sentiment analysis."
            return self.sentiment_analysis_module.analyze_market_sentiment(ticker)
        else:
            return "I'm not sure how to respond to that. Can you please provide more details?"
    
    def extract_ticker(self, query):
        match = re.search(r'\b[A-Z]{1,4}\b', query)
        return match.group(0) if match else "Unknown"

    def extract_strategy_details(self, query):
        # Regex pattern for extracting price and date
        price_pattern = r'\$\d+\.?\d*'
        date_pattern = r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'

        # Find price and date in the query
        price_match = re.search(price_pattern, query)
        date_match = re.search(date_pattern, query)

        if price_match and date_match:
            # Extract and convert price and date
            price = float(price_match.group(0).replace('$', ''))
            date_str = date_match.group(0)
            try:
                date = datetime.strptime(date_str, '%m/%d/%Y').strftime('%Y-%m-%d')
            except ValueError:
                return None, None  # Invalid date format
            return price, date
        else:
            return None, None
        
if __name__ == '__main__':
    trading_partner = TradingPartner()
    while True:
        user_query = input("How can I assist you with your trading? (type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        response = trading_partner.handle_query(user_query)
        print("Trading Partner:", response)