# Portfolio Risk Dashboard (VaR / CVaR)

A comprehensive Python-based dashboard for calculating and visualizing Value at Risk (VaR), Conditional Value at Risk (CVaR), and factor exposures for custom portfolios. The dashboard is built with **Streamlit** and relies on an advanced mathematical risk engine leveraging `scipy`, `pandas`, and `numpy`.

## Features

### 🔹 V1: Core Risk Engine & Analytics
- **VaR/CVaR Models:** Historical, Parametric (Normal & Student-t), and Monte Carlo VaR simulations.
- **Component VaR:** Analyzes the marginal risk contribution of individual portfolio assets.
- **Stress Testing:** Historical scenario replays (e.g., GFC 2008, COVID-2020) and custom shock matrices.
- **Factor Analysis:** OLS linear regressions against Fama-French 3-factor and 5-factor models to determine alpha and beta sensitivities.
- **Diversification Metrics:** Hierarchically clustered correlation heatmaps and diversification ratios.
- **Performance Tracking:** CAGR, Sharpe/Sortino ratios, and underwater drawdown charts.

### 🚀 V2: Advanced Risk & Integration
- **Intraday VaR:** 1-hour rolling window VaR using intraday data.
- **Copula-Based Dependency Modeling:** Replaces simple linear correlation with multivariate Gaussian and Student-t Copulas to better model tail dependence during extreme market events.
- **Liquidity-Adjusted VaR (L-VaR):** Incorporates execution costs (spreads) and market impact into the VaR metric.
- **CVaR Factor Decomposition:** Uses Euler allocation to decompose Expected Shortfall into contributions from specific Fama-French factors.
- **Live IBKR Integration:** Sync live portfolio positions, weights, and average costs directly from Interactive Brokers (TWS / IB Gateway) using `ib_insync`.

---

## 🛠️ Architecture Pipeline

```text
Data Layer → Risk Engine → Factor Model → Correlation & Stress → Dashboard (Streamlit)
```

1. **Data Layer (`data/`)**: Price ingestion via `yfinance`, factor returns from Ken French data library, and `parquet` caching via `pyarrow`. Also includes live IBKR syncing.
2. **Risk Engine (`risk/`)**: Core mathematical computations for Historical, Parametric, Copula-based, and Liquidity-Adjusted VaR.
3. **Analytics (`analytics/`)**: Fama-French exposures, rolling betas, performance metrics, and correlation clustering.
4. **Charting (`charts/`)**: Interactive `plotly` UI components (equity curves, heatmaps, VaR breach timelines, etc.).

---

## 💻 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Thatva1/Portolio-Risk-Dashbord.git
   cd "Portolio-Risk-Dashbord"
   ```

2. **Install dependencies:**
   Ensure you have Python 3.9+ installed.
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Portfolio:**
   Edit the `config.yaml` file to set your default tickers, weights, benchmark, and confidence levels.

4. **Run the Dashboard:**
   ```bash
   streamlit run app.py
   ```

## 📈 Interactive Brokers Live Sync
To use the live portfolio synchronization feature:
1. Open **Trader Workstation (TWS)** or **IB Gateway**.
2. Navigate to **API Settings** and enable `Enable ActiveX and Socket Clients`.
3. Ensure the port matches the default (`7497` for paper trading or `7496` for live trading).
4. Check the **Sync with Live IBKR Portfolio** box in the dashboard's sidebar!

---

*Developed for Portfolio Risk Analysis and Advanced Financial Modeling.*
