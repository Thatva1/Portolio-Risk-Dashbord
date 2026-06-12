import plotly.express as px
import pandas as pd
import numpy as np

def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    fig = px.imshow(corr_matrix, 
                    text_auto=".2f", 
                    color_continuous_scale='RdBu_r', 
                    zmin=-1, zmax=1,
                    title="Correlation Heatmap")
    return fig
