import numpy as np
import pandas as pd
from scipy.stats import norm

def compute_component_var(returns: pd.DataFrame, weights: np.ndarray, confidence_level: float = 0.95) -> pd.Series:
    """
    Compute Component VaR for each asset in the portfolio.
    Using Parametric (Normal) assumption for covariance matrix.
    """
    cov_matrix = returns.cov()
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Marginal VaR
    z_score = norm.ppf(1 - confidence_level)
    marginal_var = z_score * np.dot(cov_matrix, weights) / portfolio_volatility
    
    # Component VaR = Weight * Marginal VaR
    component_var = weights * marginal_var
    
    return pd.Series(component_var, index=returns.columns)

def compute_marginal_var(returns: pd.DataFrame, weights: np.ndarray, confidence_level: float = 0.95) -> pd.Series:
    """Compute Marginal VaR for each asset."""
    cov_matrix = returns.cov()
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    z_score = norm.ppf(1 - confidence_level)
    marginal_var = z_score * np.dot(cov_matrix, weights) / portfolio_volatility
    return pd.Series(marginal_var, index=returns.columns)
