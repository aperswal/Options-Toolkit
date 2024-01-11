import Services as s
import json
import pandas as pd
import Sentiment_Utils as su

def get_max_pain_of_a_ticker_for_next_day(ticker):
    data = s.max_pain_for_next_day(ticker)
    json_data = json.dumps(data, indent=4)  
    return json_data

def maximum_profit_contract_for_strike_price_on_date(ticker, strike_price, date):
    data = s.max_profit_contract(ticker, strike_price, date)
    json_data = data.to_json(indent=4)
    return json_data

def determine_contract_relative_value_by_volatility(contract_name):
    data = s.over_under_priced_contracts_by_volatility(contract_name)
    json_data = {contract_name: data}
    return json.dumps(json_data, indent=4)

def news_sentiment_analysis(ticker):
    response_dict = su.alpha_get_news_sentiment(ticker)
    data = su.alpha_extract_and_calculate_sentiment(ticker, response_dict)
    json_data = json.dumps(data, indent=4)
    return json_data

def contract_price_at_projected_underlying_price(contract_name, price):
    bs_price_at_target = s.calculate_contract_price_at_target(contract_name, price)
    
    result = {
        "Contract Name": contract_name,
        "Projected Price": price,
        "Contract Price at Projected Price": bs_price_at_target
    }
    
    json_data = json.dumps(result, indent=4)
    return json_data

def block_trades(ticker):
    ticker_data = su.get_intraday_stock_data(ticker)
    data = su.time_aggregated_block_trades(ticker_data)
    json_data = data.to_json(indent=4, orient='records', date_format='iso')
    return json_data

def volume_anomalies(ticker):
    ticker_data = su.get_intraday_stock_data(ticker)
    data = su.detect_volume_anomalies(ticker_data)
    json_data = data.to_json(indent=4, orient='records', date_format='iso')
    return json_data

def main():

    print(get_max_pain_of_a_ticker_for_next_day('AAPL'))
    print(maximum_profit_contract_for_strike_price_on_date('AAPL', 200, '2024-01-15'))
    print(determine_contract_relative_value_by_volatility('AAPL240112C00185000'))
    print(news_sentiment_analysis('AAPL'))
    print(block_trades('AAPL'))
    print(volume_anomalies('AAPL'))
    print(contract_price_at_projected_underlying_price('AAPL240119C00187500', 178))

if __name__ == '__main__':
    main()
