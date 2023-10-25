import unittest
from Utils import *

class TestUtils(unittest.TestCase):

    def test_get_expiry(self):
        result = get_expiry('BA231027C00200000')
        self.assertIsNotNone(result)

    def test_black_scholes(self):
        result = black_scholes(100, 95, 1, 0.05, 0.2, 'call')
        self.assertIsNotNone(result)

    def test_time_to_maturity(self):
        result = time_to_maturity('BA231027C00200000')
        self.assertIsNotNone(result)

    def test_strike_price(self):
        result = strike_price('BA231027C00200000')
        self.assertIsNotNone(result)

    def test_get_ticker_from_contract(self):
        result = get_ticker_from_contract('BA231027C00200000')
        self.assertIsNotNone(result)
        
    def test_get_underlying_price(self):
        result = get_underlying_price('BA231027C00200000')
        self.assertIsNotNone(result)
        
    def test_get_implied_volatility(self):
        result = get_implied_volatility('BA231027C00200000')
        self.assertIsNotNone(result)
        
    def test_get_risk_free_rate(self):
        result = get_risk_free_rate()
        self.assertIsNotNone(result)
        
    def test_get_historical_volatility(self):
        result = get_historical_volatility('BA231027C00200000')
        self.assertIsNotNone(result)
        
    def test_price_my_option(self):
        result = price_my_option('BA231027C00200000', 'black_scholes')
        self.assertIsNotNone(result)
    
    def test_avg_contract_price_with_all_models(self):
        result = avg_contract_price_with_all_models('BA231027C00200000')
        self.assertIsNotNone(result)
        
    def test_ideal_contract_price(self):
        result = ideal_contract_price('BA231027C00200000')
        self.assertIsNotNone(result)
        
    def test_under_valued_contracts(self):
        result = under_valued_contracts('BA231027C00200000')
        self.assertIsNotNone(result)
        
    def test_over_valued_contracts(self):
        result = over_valued_contracts('BA231027C00200000')
        self.assertIsNotNone(result)
        
    def test_monte_carlo_simulation(self):
        result = monte_carlo_simulation(100, 0.05, 0.2, 1, 1000, 1/252)
        self.assertIsNotNone(result)
        
    def test_monte_carlo_option_price(self):
        result = monte_carlo_option_price(100, 95, 1, 0.05, 0.2, 'call')
        self.assertIsNotNone(result)
    
    def test_mle_gbm(self):
        result = mle_gbm('BA231027C00200000')
        self.assertIsNotNone(result)
        
    def test_estimate_jump_parameters(self):
        result = estimate_jump_parameters('BA231027C00200000')
        self.assertIsNotNone(result)
        
    def test_jump_diffusion_simulation(self):
        result = jump_diffusion_simulation(100, 0.05, 0.2, 1, 0.1, 0.1, 0.1, 1000, 1/252)
        self.assertIsNotNone(result)
        
    def test_jump_diffusion_option_price(self):
        result = jump_diffusion_option_price(100, 95, 1, 0.05, 0.2, 0.1, 0.1, 0.1)
        self.assertIsNotNone(result)
        
    def test_exponential_moving_average(self):
        result = exponential_moving_average('BA231027C00200000')
        self.assertIsNotNone(result)
        
    def test_price_rate_of_change(self):
        result = price_rate_of_change('BA231027C00200000')
        self.assertIsNotNone(result)
    
        
if __name__ == '__main__':
    unittest.main()