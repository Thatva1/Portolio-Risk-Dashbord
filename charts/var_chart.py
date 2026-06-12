import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_var_breaches(returns: pd.Series, var_threshold: float) -> go.Figure:
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=returns.index, y=returns.values, mode='lines', name='Returns', line=dict(color='blue', width=1)))
    fig.add_hline(y=var_threshold, line_dash="dash", line_color="red", annotation_text="VaR Threshold")
    
    breaches = returns[returns < var_threshold]
    fig.add_trace(go.Scatter(x=breaches.index, y=breaches.values, mode='markers', name='Breaches', marker=dict(color='red', size=8)))
    
    fig.update_layout(title="VaR Breach Timeline", xaxis_title="Date", yaxis_title="Daily Return")
    return fig
