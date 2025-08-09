import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')

class DataFetcher:
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.returns_data = None
        self.mean_returns = None
        self.cov_matrix = None
        self.tickers = []
        
        # Crypto mappings for Yahoo Finance
        self.CRYPTO_MAP = {
            'BTC': 'BTC-USD',
            'ETH': 'ETH-USD',
            'DOGE': 'DOGE-USD',
            'ADA': 'ADA-USD',
            'SOL': 'SOL-USD',
            'XRP': 'XRP-USD',
            'DOT': 'DOT-USD',
            'AVAX': 'AVAX-USD',
            'LTC': 'LTC-USD',
            'LINK': 'LINK-USD',
            'MATIC': 'MATIC-USD'
        }
    
    @st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
    def fetch_historical_prices(_self, tickers: List[str], period: str = "1y") -> pd.DataFrame:
        """
        Fetch historical price data for given tickers
        
        Args:
            tickers: List of ticker symbols
            period: Time period for historical data (1y, 2y, 5y, max)
            
        Returns:
            DataFrame with adjusted closing prices
        """
        try:
            # Clean and validate tickers
            clean_tickers = []
            for ticker in tickers:
                # Remove any whitespace and convert to uppercase
                clean_ticker = str(ticker).strip().upper()
                
                # Handle crypto tickers
                if clean_ticker in _self.CRYPTO_MAP:
                    clean_ticker = _self.CRYPTO_MAP[clean_ticker]
                elif '.' in clean_ticker:
                    clean_ticker = clean_ticker.replace('.', '-')
                
                if clean_ticker and clean_ticker not in ['NAN', 'NONE', '']:
                    clean_tickers.append(clean_ticker)
            
            if not clean_tickers:
                st.error("No valid tickers provided")
                return pd.DataFrame()
                
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Fetch data using yfinance
            data_dict = {}
            failed_tickers = []
            
            for i, ticker in enumerate(clean_tickers):
                try:
                    status_text.text(f"Fetching {ticker}... ({i+1}/{len(clean_tickers)})")
                    
                    # Download data for individual ticker
                    ticker_obj = yf.Ticker(ticker)
                    # Handle different period types
                    if period == "max":
                        hist_data = ticker_obj.history(period="max", auto_adjust=True)
                    else:
                        hist_data = ticker_obj.history(period=period, auto_adjust=True)
                    
                    if not hist_data.empty and len(hist_data) > 30:  # Minimum 30 days of data
                        data_dict[ticker] = hist_data['Close']
                    else:
                        failed_tickers.append(ticker)
                        
                except Exception as e:
                    st.warning(f"Failed to fetch data for {ticker}: {str(e)}")
                    failed_tickers.append(ticker)
                
                progress_bar.progress((i + 1) / len(clean_tickers))
            
            progress_bar.empty()
            status_text.empty()
            
            if not data_dict:
                st.error("No historical data could be fetched for any ticker")
                return pd.DataFrame()
            
            # Combine data into single DataFrame
            price_data = pd.DataFrame(data_dict)
            
            # Remove tickers with insufficient data
            min_data_points = max(252, len(price_data) * 0.8)  # At least 80% of data or 252 days
            valid_tickers = []
            for ticker in price_data.columns:
                if price_data[ticker].count() >= min_data_points:
                    valid_tickers.append(ticker)
                else:
                    failed_tickers.append(ticker)
            
            if failed_tickers:
                st.warning(f"Excluded tickers due to insufficient data: {', '.join(failed_tickers)}")
            
            if not valid_tickers:
                st.error("No tickers have sufficient historical data")
                return pd.DataFrame()
                
            price_data = price_data[valid_tickers]
            
            # Forward fill missing values
            price_data = price_data.fillna(method='ffill').dropna()
            
            st.success(f"Successfully fetched data for {len(valid_tickers)} tickers with {len(price_data)} days of data")
            
            return price_data
            
        except Exception as e:
            st.error(f"Failed to fetch data: {str(e)}")
            return pd.DataFrame()
    
    def calculate_returns(_self, price_data: pd.DataFrame, return_type: str = "daily") -> pd.DataFrame:
        """
        Calculate returns from price data
        
        Args:
            price_data: DataFrame with price data
            return_type: Type of returns ('daily', 'weekly', 'monthly')
            
        Returns:
            DataFrame with returns
        """
        try:
            if return_type == "daily":
                returns = price_data.pct_change().dropna()
            elif return_type == "weekly":
                returns = price_data.resample('W').last().pct_change().dropna()
            elif return_type == "monthly":
                returns = price_data.resample('M').last().pct_change().dropna()
            else:
                returns = price_data.pct_change().dropna()
            
            # Remove extreme outliers (returns > 50% or < -50%)
            returns = returns.clip(-0.5, 0.5)
            
            return returns
            
        except Exception as e:
            st.error(f"Error calculating returns: {str(e)}")
            return pd.DataFrame()
    
    def prepare_historical_values(_self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare historical values data in the format needed for visualizations
        
        Args:
            price_data: DataFrame with historical prices
            
        Returns:
            DataFrame with historical values in long format
        """
        try:
            if price_data.empty:
                return pd.DataFrame()
            
            # Prepare historical values data
            historical_values = []
            
            # Create portfolio total data
            portfolio_total = price_data.sum(axis=1).reset_index()
            portfolio_total.columns = ['date', 'market_value']
            portfolio_total['symbol'] = 'PORTFOLIO_TOTAL'
            
            # Create asset-level data
            for symbol in price_data.columns:
                asset_data = price_data[[symbol]].reset_index()
                asset_data.columns = ['date', 'market_value']
                asset_data['symbol'] = symbol
                historical_values.append(asset_data)
            
            # Add portfolio total
            historical_values.append(portfolio_total)
            
            # Combine and return
            return pd.concat(historical_values, ignore_index=True)
            
        except Exception as e:
            st.error(f"Error preparing historical data: {str(e)}")
            return pd.DataFrame()
    
    def fetch_historical_data(_self, tickers: List[str], period: str = "1y") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fetch all historical data needed for both optimization and visualization
        
        Returns:
            Tuple of (price_data, returns_data, historical_values)
        """
        price_data = _self.fetch_historical_prices(tickers, period)
        if price_data.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        returns_data = _self.calculate_returns(price_data)
        historical_values = _self.prepare_historical_values(price_data)
        
        return price_data, returns_data, historical_values