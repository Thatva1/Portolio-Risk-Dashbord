import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, norm

def calculate_skewness(returns: pd.Series) -> float:
    """Calculate skewness of returns."""
    return skew(returns.dropna())

def calculate_kurtosis(returns: pd.Series) -> float:
    """Calculate excess kurtosis of returns."""
    return kurtosis(returns.dropna(), fisher=True)

def cornish_fisher_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculate Cornish-Fisher adjusted VaR taking into account skewness and kurtosis."""
    z = norm.ppf(1 - confidence_level)
    s = calculate_skewness(returns)
    k = calculate_kurtosis(returns)
    
    z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3*z) * k / 24 - (2*z**3 - 5*z) * (s**2) / 36
    
    mu = returns.mean()
    sigma = returns.std()
    
    return mu + z_cf * sigma
