import numpy as np
import pandas as pd
from .var import historical_var

def historical_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculate Historical Conditional Value at Risk (Expected Shortfall)."""
    var_threshold = historical_var(returns, confidence_level)
    tail_returns = returns[returns <= var_threshold]
    if len(tail_returns) == 0:
        return var_threshold
    return np.mean(tail_returns)

def portfolio_historical_cvar(portfolio_returns: pd.Series, confidence_level: float = 0.95) -> float:
    return historical_cvar(portfolio_returns, confidence_level)

def factor_cvar_decomposition(portfolio_returns: pd.Series, factor_returns: pd.DataFrame, confidence_level: float = 0.95) -> pd.Series:
    """
    Decompose CVaR into risk factor contributions using the tail returns.
    """
    var_threshold = historical_var(portfolio_returns, confidence_level)
    tail_indices = portfolio_returns <= var_threshold
    
    if not tail_indices.any():
        return pd.Series(0, index=factor_returns.columns)
        
    port_tail = portfolio_returns[tail_indices]
    factor_tail = factor_returns.loc[tail_indices.index]
    
    import statsmodels.api as sm
    X = sm.add_constant(factor_tail)
    model = sm.OLS(port_tail, X).fit()
    
    betas = model.params.drop('const', errors='ignore')
    
    # Factor CVaR contribution = Beta_i * E[Factor_i | Portfolio <= VaR]
    factor_means_in_tail = factor_tail.mean()
    contributions = betas * factor_means_in_tail
    
    return contributions

