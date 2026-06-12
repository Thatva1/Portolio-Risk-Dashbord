import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np

def plot_return_distribution(returns: pd.Series) -> go.Figure:
    returns = returns.dropna()
    fig = ff.create_distplot([returns.values], ['Returns'], bin_size=0.005, show_rug=False)
    fig.update_layout(title="Daily Return Distribution", xaxis_title="Return", yaxis_title="Density")
    return fig
