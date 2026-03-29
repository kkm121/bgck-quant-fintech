"""
Microservice: NSE Data Fetcher
Architecture Component: Ingestion Layer
Mathematical/Theoretical Purpose:
Standardizes temporal chronologies into volume-synchronized OHLCV matrices. 
Updated to handle Yahoo Finance's 730-day limit for hourly data and cache locking.
"""
import pandas as pd
import yfinance as yf
from typing import List
import datetime

class NSEDataFetcher:
    def __init__(self, tickers: List[str], start_date: str = None, end_date: str = None, interval: str = "1h"):
        self.tickers = tickers
        self.interval = interval
        
        # If no dates provided, use the last 59 days for 1h data (safe limit)
        if start_date is None:
            end = datetime.datetime.now()
            start = end - datetime.timedelta(days=59)
            self.start_date = start.strftime('%Y-%m-%d')
            self.end_date = end.strftime('%Y-%m-%d')
        else:
            self.start_date = start_date
            self.end_date = end_date

    def fetch_data(self) -> pd.DataFrame:
        formatted_tickers = [f"{ticker}.NS" if not ticker.endswith(".NS") else ticker for ticker in self.tickers]
        
        # We use period instead of start/end if we want the most recent reliable data
        # Or we keep start/end but ensure the interval is compatible
        raw_data = yf.download(
            tickers=formatted_tickers,
            start=self.start_date,
            end=self.end_date,
            interval=self.interval,
            group_by='ticker',
            auto_adjust=True,
            threads=False # Set to False to prevent 'database is locked' errors
        )
        
        if raw_data.empty or (len(formatted_tickers) > 1 and raw_data.dropna(how='all').empty):
            raise ValueError(f"No data returned for {self.tickers}. Ensure the date range is within the last 730 days for 1h data.")
            
        return self._clean_and_structure(raw_data, formatted_tickers)

    def _clean_and_structure(self, df: pd.DataFrame, formatted_tickers: List[str]) -> pd.DataFrame:
        if len(self.tickers) == 1 and not isinstance(df.columns, pd.MultiIndex):
            df.columns = pd.MultiIndex.from_product([formatted_tickers, df.columns])
        
        # Drop tickers that failed to download to prevent downstream index errors
        df = df.dropna(axis=1, how='all')
        df = df.ffill().bfill()
        return df

    def extract_close_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        close_df = pd.DataFrame()
        tickers_found = df.columns.levels[0]
        for ticker in tickers_found:
            if 'Close' in df[ticker]:
                close_df[ticker] = df[ticker]['Close']
        return close_df

    def extract_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        volume_df = pd.DataFrame()
        tickers_found = df.columns.levels[0]
        for ticker in tickers_found:
            if 'Volume' in df[ticker]:
                volume_df[ticker] = df[ticker]['Volume']
        return volume_df