import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
import scipy.spatial.distance as ssd

def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate pairwise correlation matrix."""
    return returns.corr()

def hierarchically_cluster_correlation(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """Reorder correlation matrix using hierarchical clustering."""
    dist = 1 - corr_matrix.values
    dist = np.clip(dist, 0, 2)
    dist_array = ssd.squareform(dist)
    
    Z = linkage(dist_array, 'ward')
    ordering = leaves_list(Z)
    
    cols = corr_matrix.columns[ordering]
    return corr_matrix.loc[cols, cols]

def calculate_diversification_ratio(returns: pd.DataFrame, weights: np.ndarray) -> float:
    """Calculate Diversification Ratio."""
    vols = returns.std()
    weighted_avg_vol = np.dot(weights, vols)
    
    cov_matrix = returns.cov()
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    if portfolio_vol == 0:
        return np.nan
        
    return weighted_avg_vol / portfolio_vol
