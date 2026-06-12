import pandas as pd
from ib_insync import IB, util
import nest_asyncio

# Required for ib_insync in jupyter/streamlit environments
nest_asyncio.apply()

def fetch_live_portfolio(host='127.0.0.1', port=7497, client_id=1):
    """
    Connects to Interactive Brokers TWS/Gateway and fetches live portfolio positions.
    Returns: (tickers, weights)
    """
    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id)
        positions = ib.positions()
        
        if not positions:
            ib.disconnect()
            return [], []
            
        data = []
        for pos in positions:
            if pos.contract.secType == 'STK':
                mv = pos.position * pos.avgCost 
                data.append({'Ticker': pos.contract.symbol, 'Value': mv})
                
        df = pd.DataFrame(data)
        if df.empty:
            ib.disconnect()
            return [], []
            
        total_value = df['Value'].sum()
        df['Weight'] = df['Value'] / total_value
        
        tickers = df['Ticker'].tolist()
        weights = df['Weight'].values
        
        ib.disconnect()
        return tickers, weights
        
    except Exception as e:
        print(f"Failed to connect to IBKR: {e}")
        return None, None
