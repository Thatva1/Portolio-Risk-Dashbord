import pandas as pd
import numpy as np

def calculate_cagr(returns: pd.Series) -> float:
    """Calculate Compound Annual Growth Rate."""
    if len(returns) == 0:
        return np.nan
    # Assuming daily returns
    days = len(returns)
    cum_return = (1 + returns).prod()
    cagr = cum_return ** (252 / days) - 1
    return cagr

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate annualized Sharpe Ratio."""
    excess_returns = returns - risk_free_rate / 252
    if excess_returns.std() == 0:
        return np.nan
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate annualized Sortino Ratio."""
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < 0]
    if downside_returns.std() == 0:
        return np.nan
    return np.sqrt(252) * excess_returns.mean() / downside_returns.std()

def calculate_max_drawdown(returns: pd.Series) -> tuple:
    """Calculate Maximum Drawdown and the underwater series."""
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    return drawdowns.min(), drawdowns

def calculate_annualized_volatility(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(252)
