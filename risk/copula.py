import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal, t

def fit_and_simulate_copula(returns: pd.DataFrame, copula_type: str = 'gaussian', sims: int = 10000) -> pd.DataFrame:
    """
    Fits a copula to the joint distribution of returns and simulates new paths using scipy.
    """
    dim = returns.shape[1]
    
    if copula_type == 'gaussian':
        # 1. Transform marginals to uniform using normal CDF for simplicity
        uniforms = norm.cdf(returns.values, loc=returns.mean().values, scale=returns.std().values)
        
        # 2. Transform uniforms to standard normal to estimate correlation matrix
        # Clip to avoid inf
        uniforms = np.clip(uniforms, 1e-10, 1 - 1e-10)
        normals = norm.ppf(uniforms)
        corr_matrix = np.corrcoef(normals, rowvar=False)
        
        # 3. Simulate from multivariate normal with estimated correlation
        mv_norm = multivariate_normal(mean=np.zeros(dim), cov=corr_matrix)
        sim_normals = mv_norm.rvs(size=sims)
        if sims == 1:
            sim_normals = sim_normals.reshape(1, -1)
            
        # 4. Transform back to uniforms
        sim_uniforms = norm.cdf(sim_normals)
        
    elif copula_type == 'student':
        # Simple student-t copula approximation
        df_deg = 4
        uniforms = t.cdf(returns.values, df=df_deg, loc=returns.mean().values, scale=returns.std().values)
        uniforms = np.clip(uniforms, 1e-10, 1 - 1e-10)
        normals = norm.ppf(uniforms)
        corr_matrix = np.corrcoef(normals, rowvar=False)
        
        mv_norm = multivariate_normal(mean=np.zeros(dim), cov=corr_matrix)
        Y = mv_norm.rvs(size=sims)
        if sims == 1:
            Y = Y.reshape(1, -1)
        V = np.random.chisquare(df_deg, size=(sims, 1))
        sim_t = Y / np.sqrt(V / df_deg)
        
        sim_uniforms = t.cdf(sim_t, df=df_deg)
        
    else:
        raise ValueError("copula_type must be 'gaussian' or 'student'")
        
    # Transform back to original marginal distributions using normal inverse CDF
    simulated_returns = np.zeros_like(sim_uniforms)
    for i, col in enumerate(returns.columns):
        mu = returns[col].mean()
        sigma = returns[col].std()
        sim_uniforms[:, i] = np.clip(sim_uniforms[:, i], 1e-10, 1 - 1e-10)
        simulated_returns[:, i] = norm.ppf(sim_uniforms[:, i], loc=mu, scale=sigma)
        
    return pd.DataFrame(simulated_returns, columns=returns.columns)

def copula_var(returns: pd.DataFrame, weights: np.ndarray, confidence_level: float = 0.95, copula_type: str = 'gaussian', sims: int = 10000) -> float:
    """Calculate VaR using copula-based Monte Carlo simulation."""
    sim_returns = fit_and_simulate_copula(returns, copula_type, sims)
    port_sim_returns = sim_returns.dot(weights)
    return float(np.percentile(port_sim_returns, 100 * (1 - confidence_level)))
