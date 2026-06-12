import yfinance as yf
import pandas_datareader.data as web
import pandas as pd
from typing import List
from .cache import save_to_cache, load_from_cache

def fetch_prices(tickers: List[str], start_date: str, end_date: str, use_cache: bool = True) -> pd.DataFrame:
    """Fetch adjusted close prices for a list of tickers."""
    cache_key = f"prices_{start_date}_{end_date}_{'_'.join(sorted(tickers))}"
    if use_cache:
        cached = load_from_cache(cache_key)
        if cached is not None:
            return cached
            
    print(f"Downloading data for {tickers}...")
    # Add auto_adjust=False to avoid FutureWarnings and download Adj Close
    df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
    
    # Deal with multi-index columns if multiple tickers
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Adj Close']
    elif 'Adj Close' in df.columns:
        df = df[['Adj Close']]
        df.columns = [tickers[0]]
        
    # Handle single ticker case
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
        
    if use_cache:
        save_to_cache(df, cache_key)
        
    return df

def fetch_fama_french(model: str = "FF5", start_date: str = "2018-01-01", end_date: str = "2024-12-31", use_cache: bool = True) -> pd.DataFrame:
    """Fetch Fama-French factors from Ken French data library."""
    dataset_name = "F-F_Research_Data_5_Factors_2x3_daily" if model == "FF5" else "F-F_Research_Data_Factors_daily"
    cache_key = f"ff_factors_{model}_{start_date}_{end_date}"
    
    if use_cache:
        cached = load_from_cache(cache_key)
        if cached is not None:
            return cached
            
    print(f"Downloading Fama-French {model} factors...")
    factors = web.DataReader(dataset_name, "famafrench", start=start_date, end=end_date)[0]
    
    # Fama-French data is typically in percentages, convert to decimals
    factors = factors / 100.0
    
    if use_cache:
        save_to_cache(factors, cache_key)
        
    return factors

def fetch_intraday_prices(tickers: List[str], period: str = "60d", interval: str = "1h", use_cache: bool = True) -> pd.DataFrame:
    """Fetch intraday prices for a list of tickers."""
    cache_key = f"intraday_{period}_{interval}_{'_'.join(sorted(tickers))}"
    if use_cache:
        cached = load_from_cache(cache_key)
        if cached is not None:
            return cached
            
    print(f"Downloading intraday data for {tickers}...")
    df = yf.download(tickers, period=period, interval=interval, auto_adjust=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
    elif 'Adj Close' in df.columns:
        df = df[['Adj Close']]
        df.columns = [tickers[0]]
    elif 'Close' in df.columns:
        df = df[['Close']]
        df.columns = [tickers[0]]
        
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
        
    if use_cache:
        save_to_cache(df, cache_key)
        
    return df

