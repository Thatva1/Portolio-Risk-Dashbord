import plotly.graph_objects as go
import pandas as pd

def plot_equity_curve(returns: pd.Series, benchmark_returns: pd.Series = None) -> go.Figure:
    fig = go.Figure()
    
    port_cum = (1 + returns).cumprod()
    fig.add_trace(go.Scatter(x=port_cum.index, y=port_cum.values, mode='lines', name='Portfolio'))
    
    if benchmark_returns is not None:
        bench_cum = (1 + benchmark_returns).cumprod()
        fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum.values, mode='lines', name='Benchmark'))
        
    fig.update_layout(title="Equity Curve", xaxis_title="Date", yaxis_title="Cumulative Return")
    return fig

def plot_underwater(drawdowns: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drawdowns.index, y=drawdowns.values, fill='tozeroy', name='Drawdown', line=dict(color='red')))
    fig.update_layout(title="Underwater Chart (Drawdowns)", xaxis_title="Date", yaxis_title="Drawdown", yaxis_tickformat='.1%')
    return fig
