# Option Analysis Tool

This tool offers a collection of financial functions and utilities to analyze options, their prices, and other associated attributes.

## Dependencies

The following standard libraries are used:
- `math`
- `re`
- `datetime`

And the third-party libraries include:
- `numpy`
- `pandas`
- `scipy.stats`
- `yfinance`
- `yoptions`
- `sklearn.linear_model`

## Key Functions

1. Contract Details

- get_expiry(contract): Fetch the expiry date of a given contract.
- time_to_maturity(contract): Determine the time left for a contract's expiration.
- strike_price(contract): Extract the strike price from a contract.
- get_ticker_from_contract(contract): Extract the ticker symbol from a contract.
- extract_option_type(contract): Identify if a contract is a call or put.

2. Pricing Models

- black_scholes(...): Compute the Black-Scholes option price.
- monte_carlo_option_price(...): Calculate option price using Monte Carlo simulation.
- jump_diffusion_option_price(...): Compute option price with the Jump Diffusion model.
- price_my_option(contract, model_type): Price an option using the specified model.

3. Simulations

- monte_carlo_simulation(...): Perform a Monte Carlo simulation for option pricing.
- jump_diffusion_simulation(...): Execute a Jump Diffusion simulation for option movements.

4. Estimations and Volatility

- get_underlying_price(contract): Get the current price of the underlying asset of a contract.
- get_implied_volatility(contract): Extract the implied volatility for a given contract.
- get_historical_volatility(contract): Fetch the historical volatility for a specified contract.
- mle_gbm(ticker): Maximum Likelihood Estimation for Geometric Brownian Motion parameters.
- estimate_jump_parameters(ticker): Estimate the jump parameters for a given ticker.

5. Forecasts and Analysis

- exponential_moving_average(ticker): Calculate the EMA for a ticker.
- price_rate_of_change(ticker): Compute the price rate of change.
- moving_average(ticker): Determine the simple moving average.
- vwap(ticker): Compute the Volume Weighted Average Price.
- simple_moving_forecast(ticker): Generate a forecast based on simple moving average.
- relative_strength_index(ticker): Calculate the RSI for a ticker.
- bollinger_bands(ticker): Fetch the Bollinger Bands for a given ticker.
- decide_trade(ticker): Provide a trade decision based on multiple indicators.
- linear_regression_forecast(ticker): Forecast future prices using linear regression.
- forecast_from_indicator(ticker, indicator_function): Generate forecasts based on a chosen indicator function.
- combined_forecast(ticker): Create a forecast by combining multiple methods.

6. Contract Analysis

- ideal_contract_price(contract): Calculate the ideal price for a given contract.
- get_ideal_contract(ticker, strike, expiry): Get the ideal contract details for the given parameters.
- profitability_range(contract, low, high): Analyze the profitability range for a contract within a price range.
- under_valued_contracts(ticker): Identify undervalued contracts for a ticker.
- over_valued_contracts(ticker): Identify overvalued contracts for a ticker.
