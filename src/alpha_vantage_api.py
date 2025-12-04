"""
Alpha Vantage API Client
========================

This module provides a clean interface to the Alpha Vantage stock market API.
We use this to fetch real-time OHLCV data for generating candlestick charts.

API Documentation: https://www.alphavantage.co/documentation/

Rate Limits (Free Tier):
    - 5 API calls per minute
    - 25 API calls per day

OHLCV Data Format:
    - Open: Price at market open
    - High: Highest price during the period
    - Low: Lowest price during the period
    - Close: Price at market close
    - Volume: Number of shares traded

Note: For production use, consider caching responses to avoid hitting
rate limits during development and testing.
"""

import requests
import pandas as pd
import time
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from datetime import datetime, timedelta

from src.config import ALPHA_VANTAGE_API_KEY, ALPHA_VANTAGE_BASE_URL


class AlphaVantageAPI:
    """
    Client for fetching stock market data from Alpha Vantage.
    
    This class handles API communication, error handling, and data parsing.
    Results are returned as pandas DataFrames for easy manipulation.
    
    Usage:
        api = AlphaVantageAPI(api_key="your_key")
        df, error = api.get_daily_data("AAPL")
        
        if error:
            print(f"Error: {error}")
        else:
            print(df.head())
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the API client.
        
        Args:
            api_key: Alpha Vantage API key. Falls back to environment
                    variable if not provided.
        """
        self.api_key = api_key or ALPHA_VANTAGE_API_KEY
        self.base_url = ALPHA_VANTAGE_BASE_URL
        self._last_call_time = 0
        
        if not self.api_key:
            print("WARNING: No API key configured.")
    
    def _rate_limit(self):
        """
        Enforce rate limiting between API calls.
        
        Alpha Vantage allows 5 calls per minute on free tier.
        We wait at least 12 seconds between calls to be safe.
        """
        min_interval = 12  # seconds
        elapsed = time.time() - self._last_call_time
        
        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            time.sleep(sleep_time)
        
        self._last_call_time = time.time()
    
    def _make_request(self, params: Dict) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Make HTTP request to the API with error handling.
        
        Args:
            params: Query parameters for the request
            
        Returns:
            tuple: (response_data, error_message)
        """
        # Enforce rate limiting
        self._rate_limit()
        
        # Add API key to parameters
        params["apikey"] = self.api_key
        
        try:
            response = requests.get(
                self.base_url,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            # Check for API-level errors
            if "Error Message" in data:
                return None, data["Error Message"]
            
            if "Note" in data:
                # Rate limit exceeded
                return None, "API rate limit exceeded. Please wait."
            
            if "Information" in data:
                return None, data["Information"]
            
            return data, None
            
        except requests.exceptions.Timeout:
            return None, "Request timed out. Try again later."
        except requests.exceptions.ConnectionError:
            return None, "Network connection error."
        except requests.exceptions.RequestException as e:
            return None, f"Request failed: {str(e)}"
        except ValueError:
            return None, "Invalid JSON response from API."
    
    def get_daily_data(
        self,
        symbol: str,
        outputsize: str = "compact"
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Fetch daily OHLCV data for a stock symbol.
        
        This is the primary method for getting data to generate
        candlestick charts. Returns daily data sorted chronologically.
        
        Args:
            symbol: Stock ticker (e.g., 'AAPL', 'GOOGL', 'MSFT')
            outputsize: 'compact' = last 100 days, 'full' = 20+ years
        
        Returns:
            tuple: (DataFrame, error_message)
            
            DataFrame columns: Open, High, Low, Close, Volume
            Index: DatetimeIndex (sorted ascending)
        """
        
        # --- CACHING LOGIC START ---
        cache_dir = Path("data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{symbol}_{outputsize}.json"
        
        data = None
        error = None

        # Check if cache exists
        if cache_file.exists():
            print(f"Loading {symbol} from cache...")
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading cache: {e}")
                data = None
        
        # If no data from cache, fetch from API
        if data is None:
            print(f"Fetching {symbol} from API...")
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol.upper().strip(),
                "outputsize": outputsize
            }
            
            data, error = self._make_request(params)
            
            if error:
                return None, error
            
            # Save to cache if valid data
            ts_key = "Time Series (Daily)"
            if ts_key in data:
                with open(cache_file, 'w') as f:
                    json.dump(data, f)
        # --- CACHING LOGIC END ---

        # Parse the time series data
        ts_key = "Time Series (Daily)"
        if ts_key not in data:
            return None, f"No daily data found for symbol: {symbol}"
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(data[ts_key], orient="index")
        
        # Clean column names (remove the "1. ", "2. " prefixes)
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        
        # Sort chronologically (oldest first)
        df = df.sort_index()
        
        print(f"Retrieved {len(df)} days of data for {symbol}")
        return df, None
    
    def get_intraday_data(
        self,
        symbol: str,
        interval: str = "5min"
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Fetch intraday OHLCV data for a stock symbol.
        
        Intraday data provides finer granularity for short-term analysis.
        Note: Extended hours data may be included.
        
        Args:
            symbol: Stock ticker
            interval: Time interval - '1min', '5min', '15min', '30min', '60min'
        
        Returns:
            tuple: (DataFrame, error_message)
        """
        valid_intervals = ["1min", "5min", "15min", "30min", "60min"]
        if interval not in valid_intervals:
            return None, f"Invalid interval. Use one of: {valid_intervals}"
        
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol.upper().strip(),
            "interval": interval,
            "outputsize": "compact"
        }
        
        data, error = self._make_request(params)
        
        if error:
            return None, error
        
        ts_key = f"Time Series ({interval})"
        if ts_key not in data:
            return None, f"No intraday data found for symbol: {symbol}"
        
        df = pd.DataFrame.from_dict(data[ts_key], orient="index")
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df.index = pd.to_datetime(df.index)
        df.index.name = "Datetime"
        df = df.sort_index()
        
        return df, None
    
    def search_symbol(self, keywords: str) -> List[Dict]:
        """
        Search for stock symbols by company name or keyword.
        
        Useful when you have a company name but need the ticker symbol.
        
        Args:
            keywords: Search query (e.g., "Apple", "Microsoft")
        
        Returns:
            list: List of matching symbols with metadata
        """
        params = {
            "function": "SYMBOL_SEARCH",
            "keywords": keywords
        }
        
        data, error = self._make_request(params)
        
        if error or "bestMatches" not in data:
            return []
        
        results = []
        for match in data["bestMatches"]:
            results.append({
                "symbol": match.get("1. symbol", ""),
                "name": match.get("2. name", ""),
                "type": match.get("3. type", ""),
                "region": match.get("4. region", ""),
                "currency": match.get("8. currency", "USD")
            })
        
        return results
    
    def get_quote(self, symbol: str) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Get the current/latest quote for a stock.
        
        Args:
            symbol: Stock ticker
            
        Returns:
            tuple: (quote_dict, error_message)
        """
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol.upper().strip()
        }
        
        data, error = self._make_request(params)
        
        if error:
            return None, error
        
        if "Global Quote" not in data or not data["Global Quote"]:
            return None, f"No quote available for: {symbol}"
        
        q = data["Global Quote"]
        
        return {
            "symbol": q.get("01. symbol", ""),
            "open": float(q.get("02. open", 0)),
            "high": float(q.get("03. high", 0)),
            "low": float(q.get("04. low", 0)),
            "price": float(q.get("05. price", 0)),
            "volume": int(q.get("06. volume", 0)),
            "latest_trading_day": q.get("07. latest trading day", ""),
            "previous_close": float(q.get("08. previous close", 0)),
            "change": float(q.get("09. change", 0)),
            "change_percent": q.get("10. change percent", "0%")
        }, None


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def fetch_stock_data(
    symbol: str,
    api_key: str = None,
    days: int = 100
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Simple function to fetch stock data without creating a class instance.
    
    Args:
        symbol: Stock ticker symbol
        api_key: Optional API key (uses environment variable if not provided)
        days: Number of days of data (100 for compact, more for full)
    
    Returns:
        tuple: (DataFrame, error_message)
    """
    api = AlphaVantageAPI(api_key=api_key)
    outputsize = "compact" if days <= 100 else "full"
    return api.get_daily_data(symbol, outputsize=outputsize)


# =============================================================================
# DEMO / TESTING
# =============================================================================

if __name__ == "__main__":
    print("Alpha Vantage API Module Test")
    print("=" * 40)
    
    api = AlphaVantageAPI()
    
    if not api.api_key:
        print("\nNo API key found. Set ALPHA_VANTAGE_API_KEY in .env file")
        print("Skipping live API tests.")
    else:
        print("\nSearching for 'Apple'...")
        results = api.search_symbol("Apple")
        for r in results[:3]:
            print(f"  {r['symbol']}: {r['name']}")
        
        print("\nFetching AAPL data...")
        df, error = api.get_daily_data("AAPL")
        
        if error:
            print(f"Error: {error}")
        else:
            print(f"\nData shape: {df.shape}")
            print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
            print("\nLast 5 days:")
            print(df.tail())