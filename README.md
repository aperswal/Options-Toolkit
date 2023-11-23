# Options Contract Pricing Project

## Overview
The Options Contract Pricing Project is designed to provide advanced tools and analytics for options traders. It integrates market sentiment analysis, options pricing models, and volatility evaluation, offering a comprehensive suite of functionalities.

## Modules and Their Functions

### Data_Utils.py
- `get_option_chain(ticker)`: Retrieves the options chain for a given stock ticker.
- `last_price_contract(contract_name)`: Gets the last traded price of a specific options contract.
- `get_risk_free_rate()`: Fetches the current risk-free interest rate.
- `get_ticker_from_contract(contract_name)`: Extracts the stock ticker from an options contract name.
- `get_expiry(contract_name)`: Determines the expiration date of an options contract.
- `get_historical_options_data(ticker, start_date, end_date)`: Gathers historical options data for a given ticker within a specified date range.
- `get_data(ticker)`: General function to obtain data for a given ticker.
- `time_to_maturity(contract_name)`: Calculates the time to maturity for an options contract.
- `strike_price(contract_name)`: Extracts the strike price from an options contract name.
- `get_underlying_price(contract_name)`: Retrieves the current price of the underlying asset for an options contract.
- `extract_option_type(contract_name)`: Determines whether an options contract is a call or put.
- `get_nearest_expiry_and_strike_filtered_options(ticker)`: Fetches options contracts for a ticker with the nearest expiration and filters them based on certain criteria.
- `get_combined_option_chain(ticker, dividend_yield, option_type, start_date, end_date, risk_free_rate)`: Combines various options data for a comprehensive analysis.

### Pricing_Utils.py
- `black_scholes(S, K, T, r, sigma, option_type)`: Implements the Black-Scholes pricing model for options.
- `future_black_scholes_price(contract_name, future_price)`: Estimates future Black-Scholes price for an options contract.
- `black_scholes_vectorized(...)`: Vectorized version of the Black-Scholes formula for batch processing.
- `monte_carlo_simulation(...)`: Conducts a Monte Carlo simulation for stock price paths.
- `monte_carlo_option_price(...)`: Estimates the price of an option using Monte Carlo simulations.
- `mle_gbm(ticker)`: Performs Maximum Likelihood Estimation for Geometric Brownian Motion parameters.
- `estimate_jump_parameters(ticker)`: Estimates parameters for the Jump Diffusion model.
- `jump_diffusion_simulation(...)`: Simulates stock price paths using the Merton Jump Diffusion model.
- `jump_diffusion_option_price(...)`: Calculates option price using Jump Diffusion simulations.
- `price_my_option(contract_name, model)`: Prices an option using specified pricing models.

### Volatility_Utils.py
- `get_implied_volatility(contract_name)`: Calculates the implied volatility for an options contract.
- `historical_volatility(ticker)`: Computes historical volatility for a given stock ticker.
- `sabr_volatility(...)`: Implements the SABR volatility model.
- `get_historical_volatility_of_contract(contract_name)`: Fetches historical volatility specifically for an options contract.
- `derived_implied_volatility(...)`: Derives the implied volatility based on market data.
- `vega(...)`: Calculates the vega of an option.
- `get_ticker_volatility(ticker)`: Retrieves volatility metrics for a specific stock ticker.

### Sentiment_Utils.py
- `visualize_net_institutional_trading_5_days()`: Visualizes net institutional trading over the past five days.
- `visualize_net_institutional_trading_today()`: Displays net institutional trading for the current day.
- `calculate_net_institutional_trading(...)`: Calculates net institutional trading based on block trades.
- `weighted_volume_sentiment_analysis(...)`: Analyzes sentiment based on trading volume and price impact.
- `detect_volume_anomalies(...)`: Identifies significant deviations in trading volume.
- `highlight_key_info(...)`: Extracts and highlights key information from data.
- `weighted_reddit_sentiment_analysis(subreddit, ticker)`: Performs sentiment analysis based on Reddit posts and comments.
- `aggregate_subreddit_sentiment(...)`: Aggregates sentiment scores from multiple subreddits.
- `alpha_extract_and_calculate_sentiment(...)`: Extracts and calculates sentiment from Alphavantage news feeds.
- `alpha_get_top_gainers_losers()`: Retrieves top gainers and losers from the stock market.
- `alpha_get_news_sentiment(...)`: Fetches news sentiment for specified tickers or topics.

### Services.py
- `over_under_priced_contracts_by_volatility(contract_name)`: Determines if contracts are overpriced or underpriced based on volatility.
- `derive_implied_volatility_contract(contract_name)`: Derives implied volatility for a specific contract.
- `max_profit_contract(...)`: Identifies the contract with the maximum potential profit based on future expectations.
- `avg_contract_price_with_all_models(contract_name)`: Averages the contract price using different pricing models.
- `profitability_range(...)`: Calculates profitability ranges for contracts within expected price fluctuations.
- `profitability_heatmap(...)`: Generates a heatmap visualization for contract profitability.
- `evaluate_contracts(tickers)`: Evaluates contracts for a list of tickers to determine pricing status.
- `market_mispriced_contracts_finder()`: Identifies mispriced contracts in the market.
