import pandas as pd
import numpy as np
from analytics.factors import run_factor_regression

def test_run_factor_regression():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100)
    port_returns = pd.Series(np.random.normal(0, 0.01, 100), index=dates)
    factor_returns = pd.DataFrame(np.random.normal(0, 0.01, (100, 3)), index=dates, columns=['Mkt-RF', 'SMB', 'HML'])
    
    results = run_factor_regression(port_returns, factor_returns)
    assert 'alpha' in results
    assert 'betas' in results
    assert 'r_squared' in results
