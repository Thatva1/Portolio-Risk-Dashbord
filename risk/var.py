import numpy as np
import pandas as pd
from scipy.stats import norm, t

def historical_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculate Historical VaR for a 1D series of returns."""
    return np.percentile(returns, 100 * (1 - confidence_level))

def parametric_var_normal(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculate Parametric VaR using Normal distribution."""
    mu = np.mean(returns)
    sigma = np.std(returns)
    return norm.ppf(1 - confidence_level, mu, sigma)

def parametric_var_t(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculate Parametric VaR using Student-t distribution."""
    params = t.fit(returns)
    return t.ppf(1 - confidence_level, *params)

def monte_carlo_var(returns: pd.Series, confidence_level: float = 0.95, sims: int = 10000) -> float:
    """Calculate Monte Carlo VaR."""
    mu = np.mean(returns)
    sigma = np.std(returns)
    np.random.seed(42) # for reproducibility
    simulated_returns = np.random.normal(mu, sigma, sims)
    return np.percentile(simulated_returns, 100 * (1 - confidence_level))

def portfolio_historical_var(portfolio_returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculate Historical VaR for portfolio returns."""
    return historical_var(portfolio_returns, confidence_level)
