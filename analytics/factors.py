import pandas as pd
import statsmodels.api as sm

def run_factor_regression(portfolio_returns: pd.Series, factor_returns: pd.DataFrame) -> dict:
    """Run OLS regression of portfolio returns against Fama-French factors."""
    # Align data
    aligned_port, aligned_factors = portfolio_returns.align(factor_returns, join='inner')
    
    # Add constant for alpha
    X = sm.add_constant(aligned_factors)
    y = aligned_port
    
    model = sm.OLS(y, X).fit()
    
    results = {
        'alpha': model.params.get('const', 0),
        'betas': model.params.drop('const', errors='ignore').to_dict(),
        'r_squared': model.rsquared,
        'p_values': model.pvalues.to_dict()
    }
    return results

def rolling_factor_regression(portfolio_returns: pd.Series, factor_returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Calculate rolling betas."""
    aligned_port, aligned_factors = portfolio_returns.align(factor_returns, join='inner')
    
    betas = []
    indices = []
    
    for i in range(window, len(aligned_port)):
        y_win = aligned_port.iloc[i-window:i]
        X_win = sm.add_constant(aligned_factors.iloc[i-window:i])
        model = sm.OLS(y_win, X_win).fit()
        beta_dict = model.params.drop('const', errors='ignore').to_dict()
        betas.append(beta_dict)
        indices.append(aligned_port.index[i-1])
        
    return pd.DataFrame(betas, index=indices)
