import json
import requests
import pandas as pd
import matplotlib.pyplot as plt

def alpha_load_api_key():
    with open('alpha_secret.json') as file:
        return json.load(file)['key']

def alpha_vantage_request(function, symbol, api_key, **kwargs):
    base_url = "https://www.alphavantage.co/query"
    params = {"function": function, "symbol": symbol, "apikey": api_key, **kwargs}
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data: {response.status_code}")
        return None

def present_data(data, title="Financial Data"):
    if isinstance(data, dict):
        print(f"\n{title}:")
        for key, value in data.items():
            print(f"{key}: {value}")
    elif isinstance(data, pd.DataFrame):
        print(f"\n{title}:")
        print(data.to_string(index=False))

def plot_data(data, title="Financial Plot"):
    if isinstance(data, pd.Series):
        data.plot(kind='bar')
        plt.title(title)
        plt.show()
    elif isinstance(data, pd.DataFrame):
        data.plot(kind='line')
        plt.title(title)
        plt.show()

def fetch_financial_data(ticker, api_key, report_type):
    base_url = "https://www.alphavantage.co/query"
    function_map = {
        'income_statement': 'INCOME_STATEMENT',
        'balance_sheet': 'BALANCE_SHEET',
        'cash_flow': 'CASH_FLOW'
    }
    params = {
        "function": function_map[report_type],
        "symbol": ticker,
        "apikey": api_key
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching {report_type} data: {response.status_code}")
        return None

def three_statement_model(ticker, api_key):
    income_statement = fetch_financial_data(ticker, api_key, 'income_statement')
    balance_sheet = fetch_financial_data(ticker, api_key, 'balance_sheet')
    cash_flow = fetch_financial_data(ticker, api_key, 'cash_flow')

    if not income_statement or not balance_sheet or not cash_flow:
        return "Error fetching financial data"

    # Process and integrate data from the three statements
    # Example (simplified):
    net_income = income_statement['annualReports'][0]['netIncome']
    total_assets = balance_sheet['annualReports'][0]['totalAssets']
    total_liabilities = balance_sheet['annualReports'][0]['totalLiabilities']
    operating_cash_flow = cash_flow['annualReports'][0]['operatingCashflow']

    financial_health = {
        'Net Income': net_income,
        'Total Assets': total_assets,
        'Total Liabilities': total_liabilities,
        'Operating Cash Flow': operating_cash_flow
    }

    return financial_health

def forecast_book_value(ticker, api_key, growth_rate, years):
    balance_sheet = fetch_financial_data(ticker, api_key, 'balance_sheet')
    if not balance_sheet:
        return "Error fetching balance sheet data"

    current_book_value = balance_sheet['annualReports'][0]['totalShareholderEquity']

    forecast = {}
    for year in range(1, years + 1):
        future_value = current_book_value * ((1 + growth_rate) ** year)
        forecast[f'Year {year}'] = future_value

    return forecast

def get_company_overview(symbol, api_key):
    return alpha_vantage_request("OVERVIEW", symbol, api_key)

def get_earnings(symbol, api_key):
    return alpha_vantage_request("EARNINGS", symbol, api_key)

def get_balance_sheet(symbol, api_key):
    return alpha_vantage_request("BALANCE_SHEET", symbol, api_key)

def get_cash_flow(symbol, api_key):
    return alpha_vantage_request("CASH_FLOW", symbol, api_key)

def debt_analysis(total_debt, interest_expense, operating_income):
    interest_coverage_ratio = operating_income / interest_expense if interest_expense != 0 else float('inf')
    return {
        "Interest Coverage Ratio": interest_coverage_ratio
    }

def automatic_debt_analysis(ticker, api_key):
    income_statement = alpha_vantage_request("INCOME_STATEMENT", ticker, api_key)
    balance_sheet = alpha_vantage_request("BALANCE_SHEET", ticker, api_key)

    total_debt = balance_sheet['TotalDebt']  # Adjust key as per API response structure
    interest_expense = income_statement['InterestExpense']  # Adjust key as per API response structure
    operating_income = income_statement['OperatingIncome']  # Adjust key as per API response structure

    return debt_analysis(total_debt, interest_expense, operating_income)

def discounted_cash_flow(free_cash_flows, discount_rate, forecast_years):
    dcf_valuation = 0
    for year in range(1, forecast_years + 1):
        dcf_valuation += free_cash_flows[year - 1] / ((1 + discount_rate) ** year)
    return dcf_valuation

def leveraged_buyout(equity_contribution, debt_amount, exit_multiple, ebitda, depreciation, amortization, capex, working_capital, interest_rate):
    initial_investment = equity_contribution + debt_amount
    net_income = ebitda - depreciation - amortization - capex - working_capital
    annual_cash_flow = net_income - (debt_amount * interest_rate)
    exit_value = ebitda * exit_multiple
    total_cash_return = exit_value + (annual_cash_flow * 5)  # Assuming a 5-year holding period
    irr = (total_cash_return / initial_investment) ** (1/5) - 1
    cash_on_cash_multiple = total_cash_return / equity_contribution

    return {
        "IRR": irr,
        "Cash-on-Cash Multiple": cash_on_cash_multiple
    }

def return_on_equity(net_income, shareholder_equity):
    return net_income / shareholder_equity if shareholder_equity != 0 else float('inf')

def automatic_roe(ticker, api_key):
    balance_sheet = get_balance_sheet(ticker, api_key)
    net_income = balance_sheet['NetIncome']
    shareholder_equity = balance_sheet['TotalShareholderEquity']
    return return_on_equity(net_income, shareholder_equity)

def price_earnings_ratio(market_price, earnings_per_share):
    return market_price / earnings_per_share if earnings_per_share != 0 else float('inf')

def automatic_pe_ratio(ticker, api_key):
    overview = get_company_overview(ticker, api_key)
    market_price = overview['MarketPrice']  # Assuming MarketPrice is a key in the overview data
    earnings_per_share = overview['EPS']
    return price_earnings_ratio(market_price, earnings_per_share)

def current_ratio(current_assets, current_liabilities):
    return current_assets / current_liabilities if current_liabilities != 0 else float('inf')

def automatic_current_ratio(ticker, api_key):
    balance_sheet = get_balance_sheet(ticker, api_key)
    current_assets = balance_sheet['TotalCurrentAssets']
    current_liabilities = balance_sheet['TotalCurrentLiabilities']
    return current_ratio(current_assets, current_liabilities)

def asset_turnover_ratio(sales, total_assets):
    return sales / total_assets if total_assets != 0 else float('inf')

def automatic_asset_turnover(ticker, api_key):
    overview = get_company_overview(ticker, api_key)
    sales = overview['Revenue']  # Assuming Revenue is a key in the overview data
    total_assets = overview['TotalAssets']
    return asset_turnover_ratio(sales, total_assets)

# Load Alpha Vantage API Key
alpha_api_key = alpha_load_api_key()

# Example usage
if __name__ == '__main__':
    symbol = "IBM"  # Replace with desired stock symbol
    overview = get_company_overview(symbol, alpha_api_key)
    earnings = get_earnings(symbol, alpha_api_key)
    balance_sheet = get_balance_sheet(symbol, alpha_api_key)
    cash_flow = get_cash_flow(symbol, alpha_api_key)

    print("Company Overview:", overview)
    print("Earnings Data:", earnings)
    print("Balance Sheet Data:", balance_sheet)
    print("Cash Flow Data:", cash_flow)
    
    
