# Portfolio Risk Dashboard (VaR / CVaR) — Technical Reference

---

## Architecture Pipeline

```
Data Layer → Risk Engine → Factor Model → Correlation & Stress → Dashboard (Streamlit)
```

---

## Tech Stack by Layer

### 1. Data Layer
*Price ingestion, returns calculation & caching*

| Library | Purpose | Type |
|---|---|---|
| `yfinance` | OHLCV price history for all portfolio tickers | core |
| `pandas-datareader` | Fama-French 3F & 5F factors from Ken French's library | core |
| `pyarrow` | Parquet caching layer — avoids re-downloading on every run | core |
| `pandas` | Log returns calculation, alignment, resampling | core |
| `pydantic` | Portfolio config validation (tickers, weights, benchmark) | core |

---

### 2. Risk Engine
*VaR, CVaR and component risk calculations*

| Library | Purpose | Type |
|---|---|---|
| `numpy` | Return arrays, covariance matrices, VaR/CVaR math | core |
| `scipy.stats` | Normal + Student-t distribution fitting (parametric VaR) | core |
| `pandas` | Rolling windows for historical simulation VaR | core |
| `numpy.random` | Monte Carlo simulation — 10,000 portfolio return paths | core |
| `scipy.optimize` | Portfolio variance minimisation for component VaR | core |

---

### 3. Factor Model
*Fama-French exposure, rolling betas & alpha decomposition*

| Library | Purpose | Type |
|---|---|---|
| `statsmodels` | FF3 / FF5 OLS regression + rolling window betas | core |
| `pandas-datareader` | Download FF factor returns directly (SMB, HML, RMW, CMA) | core |
| `numpy` | Beta decomposition, R-squared, factor variance attribution | core |
| `scikit-learn` | PCA for principal risk factor extraction | optional |

---

### 4. Correlation & Stress Testing
*Diversification analysis and scenario replay*

| Library | Purpose | Type |
|---|---|---|
| `numpy` | Pairwise correlation matrix + rolling correlation | core |
| `scipy.cluster.hierarchy` | Hierarchical clustering for heatmap ordering | core |
| `pandas` | Historical scenario slicing (GFC, COVID, dot-com) | core |
| `numpy.random` | Custom Monte Carlo stress path generation | core |

---

### 5. Visualisation & Dashboard
*Charts and interactive Streamlit UI*

| Library | Purpose | Type |
|---|---|---|
| `plotly` | Interactive equity curve, VaR cone, correlation heatmap | core |
| `plotly.graph_objects` | Custom VaR breach timeline + return distribution overlay | core |
| `seaborn` | Styled correlation heatmap + return distribution KDE | core |
| `matplotlib` | Static chart exports for README screenshots | core |
| `Streamlit` | Full web dashboard — weights input, date range, export | core |
| `streamlit-extras` | Enhanced Streamlit components (metric cards, tabs) | optional |

---

## Features Checklist

### VaR & CVaR Engine
- [x] Historical VaR — rolling 252-day window at 95% and 99%
- [x] Parametric VaR — normal distribution assumption
- [x] Parametric VaR — Student-t distribution (fat tails)
- [x] Monte Carlo VaR — 10,000 simulated return paths
- [x] CVaR / Expected Shortfall at 95% and 99%
- [x] Component VaR — per-asset risk contribution (%)
- [x] Marginal VaR — incremental risk of adding one unit of each asset
- [x] VaR breach timeline chart (days limit was exceeded)

### Portfolio Analytics
- [x] CAGR, Sharpe, Sortino, Calmar, Omega ratio
- [x] Maximum drawdown + underwater chart
- [x] Rolling 252-day and 60-day Sharpe + Sortino
- [x] Daily P&L distribution (histogram + KDE overlay)
- [x] Best and worst day / month / year summary
- [x] Annualised volatility (total + downside)

### Factor Analysis
- [x] Fama-French 3-factor exposure (market, SMB, HML)
- [x] Fama-French 5-factor exposure (adds RMW, CMA)
- [x] Rolling 60-day and 252-day betas for each factor
- [x] Factor contribution to portfolio variance (%)
- [x] CAPM alpha + beta vs chosen benchmark
- [x] R-squared (% of variance explained by factor model)

### Correlation & Diversification
- [x] Full pairwise correlation heatmap (hierarchically clustered)
- [x] Rolling 60-day pairwise correlation between assets
- [x] Diversification ratio (weighted avg vol / portfolio vol)
- [x] Effective number of bets (entropy-based concentration)
- [x] Correlation regime detection (low vs high correlation periods)

### Stress Testing & Scenarios
- [x] Historical scenario replay — GFC 2008, COVID 2020, dot-com 2000
- [x] Custom shock input (define % shocks to individual assets)
- [x] Monte Carlo drawdown probability cone (forward-looking)
- [x] Tail risk: skewness, excess kurtosis, Cornish-Fisher VaR adjustment

### Dashboard UI
- [x] Portfolio weight input — manual sliders or CSV upload
- [x] Ticker autocomplete with name resolution
- [x] Date range selector (backtest window)
- [x] Confidence level toggle (95% / 99%)
- [x] Benchmark selector (SPY, ACWI, ^FTSE, custom)
- [x] Export full report to PDF / CSV

### Advanced — v2 Roadmap
- [ ] Live portfolio integration via IBKR API
- [ ] Intraday VaR (1-hour rolling window)
- [ ] Liquidity-adjusted VaR (bid-ask spread + volume impact)
- [ ] Copula-based dependency modelling (replacing correlation)
- [ ] Expected shortfall decomposition by risk factor

---

## Repo Structure

```
portfolio-risk-dashboard/
├── data/
│   ├── fetcher.py          # yfinance + FF factor downloader
│   ├── cache.py            # parquet read/write caching
│   └── preprocessor.py     # log returns, alignment, outlier removal
├── risk/
│   ├── var.py              # Historical, Parametric, Monte Carlo VaR
│   ├── cvar.py             # CVaR / Expected Shortfall
│   ├── component_var.py    # per-asset risk contribution
│   └── stress.py           # historical scenarios + custom shocks
├── analytics/
│   ├── metrics.py          # Sharpe, Sortino, Calmar, drawdown stats
│   ├── factors.py          # FF3/FF5 OLS regression + rolling betas
│   ├── correlation.py      # pairwise matrix, diversification ratio
│   └── distribution.py     # skewness, kurtosis, Cornish-Fisher adj
├── charts/
│   ├── equity_curve.py     # portfolio NAV + drawdown waterfall
│   ├── var_chart.py        # VaR breach timeline + Monte Carlo cone
│   ├── heatmap.py          # clustered correlation heatmap
│   ├── factor_chart.py     # rolling betas + factor variance bar
│   └── distribution.py     # return histogram + KDE + normal overlay
├── tests/
│   ├── test_var.py         # VaR calculation unit tests
│   └── test_factors.py     # FF regression output tests
├── app.py                  # Streamlit dashboard entry point
├── config.yaml             # tickers, weights, benchmark, date range
├── requirements.txt        # pinned dependencies
└── README.md               # screenshots, methodology, example output
```

---

## requirements.txt

```
yfinance>=0.2.40
pandas>=2.2.0
pandas-datareader>=0.10.0
pyarrow>=15.0.0
numpy>=1.26.0
scipy>=1.12.0
statsmodels>=0.14.0
scikit-learn>=1.4.0
plotly>=5.20.0
matplotlib>=3.8.0
seaborn>=0.13.0
streamlit>=1.32.0
pydantic>=2.6.0
pyyaml>=6.0.1
pytest>=8.0.0
```

---

## config.yaml (example)

```yaml
portfolio:
  tickers: ["AAPL", "MSFT", "JPM", "GLD", "TLT"]
  weights: [0.25, 0.25, 0.20, 0.15, 0.15]
  benchmark: "SPY"
  start_date: "2018-01-01"
  end_date: "2024-12-31"

risk:
  confidence_levels: [0.95, 0.99]
  monte_carlo_sims: 10000
  rolling_window_days: 252

factors:
  model: "FF5"  # FF3 or FF5
  rolling_window_days: [60, 252]
```

---

*Generated for Thatva Gowda | MSc Finance, Bayes Business School*
