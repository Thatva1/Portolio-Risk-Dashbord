import numpy as np
import pandas as pd
from typing import Dict, List

def calculate_liquidity_penalty(volumes: pd.DataFrame, position_sizes: np.ndarray, base_spreads: Dict[str, float] = None) -> pd.Series:
    """
    Calculate a simple liquidity penalty (L-VaR adjustment).
    This assumes penalty = (0.5 * spread) + (Position Size / ADV) * impact_factor
    """
    if base_spreads is None:
        base_spreads = {}
        
    adv = volumes.mean()
    
    penalties = []
    for i, col in enumerate(volumes.columns):
        pos_size = position_sizes[i]
        avg_vol = adv[col]
        spread = base_spreads.get(col, 0.001) # Default 10 bps spread
        
        # Simple market impact model
        if avg_vol > 0:
            impact = 0.1 * np.sqrt(pos_size / avg_vol)
        else:
            impact = 0.0
            
        penalty = (0.5 * spread) + impact
        penalties.append(penalty)
        
    return pd.Series(penalties, index=volumes.columns)

def liquidity_adjusted_var(base_var: float, liquidity_penalties: pd.Series, weights: np.ndarray) -> float:
    """Add liquidity penalty to base VaR. Note: base_var is typically a negative number representing loss."""
    # Since var is negative, we subtract the penalty to make it more negative (larger loss)
    portfolio_penalty = np.dot(liquidity_penalties.values, weights)
    return base_var - portfolio_penalty
