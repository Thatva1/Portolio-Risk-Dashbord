import streamlit as st
import pandas as pd
import numpy as np
import yaml
from data.fetcher import fetch_prices, fetch_fama_french, fetch_intraday_prices
from data.preprocessor import compute_log_returns, align_datasets
from data.ibkr_sync import fetch_live_portfolio
from risk.var import historical_var, parametric_var_normal, parametric_var_t, monte_carlo_var, portfolio_historical_var
from risk.cvar import historical_cvar, portfolio_historical_cvar, factor_cvar_decomposition
from risk.component_var import compute_component_var
from risk.copula import copula_var
from risk.liquidity import calculate_liquidity_penalty, liquidity_adjusted_var
from analytics.metrics import calculate_cagr, calculate_sharpe_ratio, calculate_max_drawdown, calculate_annualized_volatility
from analytics.factors import run_factor_regression
from analytics.correlation import calculate_correlation_matrix
from charts.equity_curve import plot_equity_curve, plot_underwater
from charts.var_chart import plot_var_breaches
from charts.heatmap import plot_correlation_heatmap
from charts.distribution import plot_return_distribution

st.set_page_config(page_title="Portfolio Risk Dashboard", layout="wide")

@st.cache_data
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()

st.title("Portfolio Risk Dashboard (VaR / CVaR) - V2")

# Sidebar
st.sidebar.header("Portfolio Configuration")
use_ibkr = st.sidebar.checkbox("Sync with Live IBKR Portfolio")

tickers_input = st.sidebar.text_input("Tickers (comma separated)", value=", ".join(config['portfolio']['tickers']))
weights_input = st.sidebar.text_input("Weights (comma separated)", value=", ".join(map(str, config['portfolio']['weights'])))
benchmark = st.sidebar.text_input("Benchmark", value=config['portfolio']['benchmark'])
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime(config['portfolio']['start_date']))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime(config['portfolio']['end_date']))

if use_ibkr:
    with st.spinner("Connecting to IBKR..."):
        ib_tickers, ib_weights = fetch_live_portfolio()
        if ib_tickers is not None and len(ib_tickers) > 0:
            st.sidebar.success("Successfully synced with IBKR!")
            tickers = ib_tickers
            weights = ib_weights
        else:
            st.sidebar.error("Failed to sync or no positions found. Using config.")
            tickers = [t.strip() for t in tickers_input.split(',')]
            weights = np.array([float(w.strip()) for w in weights_input.split(',')])
else:
    tickers = [t.strip() for t in tickers_input.split(',')]
    weights = np.array([float(w.strip()) for w in weights_input.split(',')])

if len(tickers) != len(weights):
    st.error("Number of tickers must match number of weights.")
    st.stop()

# Load Data
with st.spinner("Fetching data..."):
    prices = fetch_prices(tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    bench_prices = fetch_prices([benchmark], start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

returns = compute_log_returns(prices)
bench_returns = compute_log_returns(bench_prices)[benchmark]

# Align
returns, bench_returns = align_datasets(returns, bench_returns)
portfolio_returns = returns.dot(weights)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Risk Engine", "Factor Analysis", "Correlation", "Advanced Risk (V2)"])

with tab1:
    st.subheader("Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CAGR", f"{calculate_cagr(portfolio_returns)*100:.2f}%")
    col2.metric("Ann. Volatility", f"{calculate_annualized_volatility(portfolio_returns)*100:.2f}%")
    col3.metric("Sharpe Ratio", f"{calculate_sharpe_ratio(portfolio_returns):.2f}")
    max_dd, drawdowns = calculate_max_drawdown(portfolio_returns)
    col4.metric("Max Drawdown", f"{max_dd*100:.2f}%")
    
    st.plotly_chart(plot_equity_curve(portfolio_returns, bench_returns), use_container_width=True)
    st.plotly_chart(plot_underwater(drawdowns), use_container_width=True)

with tab2:
    st.subheader("Value at Risk & Expected Shortfall")
    conf_level = st.selectbox("Confidence Level", config['risk']['confidence_levels'], key='var_conf')
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Historical VaR", f"{portfolio_historical_var(portfolio_returns, conf_level)*100:.2f}%")
    col2.metric("Parametric VaR (Normal)", f"{parametric_var_normal(portfolio_returns, conf_level)*100:.2f}%")
    col3.metric("Monte Carlo VaR", f"{monte_carlo_var(portfolio_returns, conf_level, config['risk']['monte_carlo_sims'])*100:.2f}%")
    
    st.metric("Expected Shortfall (CVaR)", f"{portfolio_historical_cvar(portfolio_returns, conf_level)*100:.2f}%")
    
    st.plotly_chart(plot_var_breaches(portfolio_returns, portfolio_historical_var(portfolio_returns, conf_level)), use_container_width=True)
    
    st.subheader("Component VaR")
    cvar = compute_component_var(returns, weights, conf_level)
    st.bar_chart(cvar)
    
    st.subheader("Return Distribution")
    st.plotly_chart(plot_return_distribution(portfolio_returns), use_container_width=True)

with tab3:
    st.subheader("Factor Analysis")
    model = config['factors']['model']
    with st.spinner(f"Fetching {model} factors..."):
        factors = fetch_fama_french(model, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        port_simple = np.exp(portfolio_returns) - 1
        results = run_factor_regression(port_simple, factors)
        
    st.write(f"R-squared: {results['r_squared']:.4f}")
    st.write(f"Alpha: {results['alpha']:.4f}")
    st.write("Betas:")
    st.json(results['betas'])

with tab4:
    st.subheader("Correlation Matrix")
    corr = calculate_correlation_matrix(returns)
    st.plotly_chart(plot_correlation_heatmap(corr), use_container_width=True)

with tab5:
    st.subheader("Advanced Risk Models (V2)")
    conf_level_v2 = st.selectbox("Confidence Level", config['risk']['confidence_levels'], key='var_conf_v2')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Intraday VaR (1-Hour)")
        with st.spinner("Fetching intraday data..."):
            intra_prices = fetch_intraday_prices(tickers, period="60d", interval="1h")
            intra_returns = compute_log_returns(intra_prices)
            port_intra_returns = intra_returns.dot(weights)
            
            intra_var = portfolio_historical_var(port_intra_returns, conf_level_v2)
            st.metric(f"1-Hour Historical VaR ({conf_level_v2*100}%)", f"{intra_var*100:.2f}%")
            
        st.markdown("#### Liquidity-Adjusted VaR (L-VaR)")
        with st.spinner("Calculating liquidity penalties..."):
            st.info("Approximating liquidity penalty using 10bps spread cost (for demo).")
            base_var = portfolio_historical_var(portfolio_returns, conf_level_v2)
            penalty = np.dot(np.array([0.0005]*len(weights)), weights) # 5bps half-spread average
            l_var = base_var - penalty
            st.metric("Base Daily VaR", f"{base_var*100:.2f}%")
            st.metric("Liquidity-Adjusted VaR", f"{l_var*100:.2f}%", delta=f"{-penalty*100:.2f}%", delta_color="inverse")
                
    with col2:
        st.markdown("#### Copula-Based Monte Carlo VaR")
        copula_type = st.selectbox("Copula Structure", ["gaussian", "student"])
        with st.spinner(f"Fitting {copula_type} copula & simulating..."):
            c_var = copula_var(returns, weights, conf_level_v2, copula_type=copula_type, sims=2000)
            st.metric(f"Copula VaR ({conf_level_v2*100}%)", f"{c_var*100:.2f}%")
            
        st.markdown("#### CVaR Factor Decomposition")
        model = config['factors']['model']
        with st.spinner(f"Decomposing CVaR into {model} factors..."):
            factors = fetch_fama_french(model, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            port_simple = np.exp(portfolio_returns) - 1
            p_ret, f_ret = align_datasets(port_simple, factors)
            decomp = factor_cvar_decomposition(p_ret, f_ret, conf_level_v2)
            st.bar_chart(decomp)

