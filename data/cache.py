import pandas as pd
import os

CACHE_DIR = "cache"

def ensure_cache_dir():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def save_to_cache(df: pd.DataFrame, filename: str):
    ensure_cache_dir()
    path = os.path.join(CACHE_DIR, f"{filename}.parquet")
    df.columns = df.columns.astype(str) # ensure columns are strings for parquet
    df.to_parquet(path)

def load_from_cache(filename: str) -> pd.DataFrame:
    path = os.path.join(CACHE_DIR, f"{filename}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None
