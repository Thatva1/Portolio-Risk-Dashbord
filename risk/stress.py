import pandas as pd
import numpy as np

def run_historical_scenario(returns: pd.DataFrame, weights: np.ndarray, scenario: str) -> float:
    """Run historical scenario replay."""
    scenarios = {
        'GFC_2008': ('2008-01-01', '2008-12-31'),
        'COVID_2020': ('2020-02-19', '2020-03-23'),
        'DOTCOM_2000': ('2000-03-10', '2002-10-09')
    }
    
    if scenario not in scenarios:
        raise ValueError(f"Scenario {scenario} not defined.")
        
    start_date, end_date = scenarios[scenario]
    
    # Filter returns for the scenario period
    mask = (returns.index >= start_date) & (returns.index <= end_date)
    scenario_returns = returns.loc[mask]
    
    if scenario_returns.empty:
        return np.nan
        
    portfolio_returns = scenario_returns.dot(weights)
    
    # Return cumulative return over the scenario
    cum_return = (1 + portfolio_returns).prod() - 1
    return cum_return

def custom_shock(returns: pd.DataFrame, weights: np.ndarray, shocks: dict) -> float:
    """
    Apply custom percentage shocks to individual assets.
    shocks format: {'AAPL': -0.10, 'MSFT': -0.05}
    """
    shocked_returns = np.zeros(len(weights))
    
    for i, col in enumerate(returns.columns):
        if col in shocks:
            shocked_returns[i] = shocks[col]
            
    portfolio_shock = np.dot(shocked_returns, weights)
    return portfolio_shock
