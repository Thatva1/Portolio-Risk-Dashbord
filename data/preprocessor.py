import pandas as pd
import numpy as np

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from prices."""
    # log(P_t / P_{t-1})
    returns = np.log(prices / prices.shift(1))
    return returns.dropna()

def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily returns from prices."""
    returns = prices.pct_change()
    return returns.dropna()

def align_datasets(df1: pd.DataFrame, df2: pd.DataFrame, method='inner') -> tuple:
    """Align two dataframes on their index."""
    aligned = pd.concat([df1, df2], axis=1, join=method).dropna()
    return aligned[df1.columns], aligned[df2.columns]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove NaNs and forward fill missing prices if necessary."""
    # Forward fill then backward fill
    df = df.ffill().bfill()
    return df
