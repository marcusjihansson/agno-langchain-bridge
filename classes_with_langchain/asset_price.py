from alpaca.data.timeframe import TimeFrame
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.requests import StockBarsRequest
import datetime
from datetime import timedelta, datetime
from zoneinfo import ZoneInfo
import pandas as pd
import pprint 
from typing import Dict, Union

# Safely access API keys using .get()
import os 

alpaca_key = os.getenv('ALPACA_KEY')
alpaca_secret = os.getenv('ALPACA_SECRET')


class AssetPriceFetcher:
    def __init__(self, alpaca_key: str, alpaca_secret: str):
        """
        Initialize the StockAnalyzer with Alpaca API credentials.
        """
        self.alpaca_key = alpaca_key
        self.alpaca_secret = alpaca_secret

    def fetch_ohlcv(self, symbol: str, days: int = 30, timeframe: TimeFrame = TimeFrame.Day, limit: int = None) -> Union[pd.DataFrame, str]:
        """
        Fetch historical stock bars from Alpaca API.
        """
        try:
            # Create Alpaca client
            client = StockHistoricalDataClient(self.alpaca_key, self.alpaca_secret)

            # Set timezone to New York
            now = datetime.now(ZoneInfo("America/New_York"))

            # Prepare request parameters
            if limit is None:
                limit = days

            req = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=timeframe,
                start=now - timedelta(days=days),
                limit=limit
            )

            # Fetch stock bars
            bars = client.get_stock_bars(req)
            df = bars.df

            # Ensure VWAP is included
            if "vwap" not in df.columns:
                return "Error: VWAP data not available in fetched data."

            # Reset index and parse timestamp as datetime
            df = df.reset_index()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)  # Set timestamp as the index

            return df  # Return as a DataFrame for further processing

        except Exception as e:
            return f"Error fetching OHLCV data: {str(e)}"

    def fetch_ohlcv_crypto(self, symbol: str, days: int = 30, timeframe: TimeFrame = TimeFrame.Day, limit: int = None) -> Union[pd.DataFrame, str]:
        """
        Fetch historical stock bars from Alpaca API.
        """
        try:
            # Create Alpaca client
            client = CryptoHistoricalDataClient(self.alpaca_key, self.alpaca_secret)

            # Set timezone to New York
            now = datetime.now(ZoneInfo("America/New_York"))

            # Prepare request parameters
            if limit is None:
                limit = days

            req = CryptoBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=timeframe,
                start=now - timedelta(days=days),
                limit=limit
            )

            # Fetch stock bars
            bars = client.get_crypto_bars(req)
            df = bars.df

            # Ensure VWAP is included
            if "vwap" not in df.columns:
                return "Error: VWAP data not available in fetched data."

            # Reset index and parse timestamp as datetime
            df = df.reset_index()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)  # Set timestamp as the index

            return df  # Return as a DataFrame for further processing

        except Exception as e:
            return f"Error fetching OHLCV data: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Initialize the StockAnalyzer instance
    analyzer = AssetPriceFetcher(alpaca_key, alpaca_secret)
    
    # Analyze a company by its ticker symbol
    symbol = "ETH/USD"
    asset_price = analyzer.fetch_ohlcv_crypto(symbol)
    
    
    # Print the analysis result
    print(f"Asset Price for {symbol}:")

    pprint.pprint(asset_price)