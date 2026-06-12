import plotly.graph_objects as go
import pandas as pd

def plot_rolling_betas(rolling_betas: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col in rolling_betas.columns:
        fig.add_trace(go.Scatter(x=rolling_betas.index, y=rolling_betas[col], mode='lines', name=col))
        
    fig.update_layout(title="Rolling Factor Betas", xaxis_title="Date", yaxis_title="Beta")
    return fig
