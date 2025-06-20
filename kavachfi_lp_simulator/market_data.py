"""
Market data utilities for fetching real cryptocurrency prices.
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Tuple

class AssetType(Enum):
    BTC = "Bitcoin"
    ETH = "Ethereum"
    SOL = "Solana"

class MarketDataFetcher:
    """Handles fetching of real market data for cryptocurrencies."""
    
    TICKERS = {
        AssetType.BTC: "BTC-USD",
        AssetType.ETH: "ETH-USD",
        AssetType.SOL: "SOL-USD"
    }
    
    @classmethod
    def get_historical_data(
        cls, 
        asset_type: AssetType, 
        days: int = 7,
        end_date: Optional[datetime] = None
    ) -> Tuple[pd.DataFrame, bool]:
        """
        Fetch historical price and volume data for the specified asset.
        
        Args:
            asset_type: Type of asset (BTC, ETH, or SOL)
            days: Number of days of historical data to fetch
            end_date: End date for the data (defaults to now)
            
        Returns:
            Tuple of (dataframe with historical data, success boolean)
        """
        if end_date is None:
            end_date = datetime.now()
            
        start_date = end_date - timedelta(days=days)
        
        try:
            ticker = cls.TICKERS[asset_type]
            data = yf.download(
                ticker, 
                start=start_date, 
                end=end_date,
                progress=False
            )
            
            if data.empty:
                return pd.DataFrame(), False
                
            # Resample to hourly data to match simulation frequency
            data = data.resample('1H').ffill()
            
            # Calculate hourly returns for volatility
            data['Returns'] = data['Adj Close'].pct_change()
            
            return data, True
            
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return pd.DataFrame(), False
    
    @classmethod
    def merge_with_simulation(
        cls,
        sim_time_index: pd.DatetimeIndex,
        sim_price_series: np.ndarray,
        sim_volume_series: np.ndarray,
        asset_type: AssetType
    ) -> Dict[str, np.ndarray]:
        """
        Merge real market data with simulation data.
        
        Args:
            sim_time_index: Time index from simulation
            sim_price_series: Simulated price series
            sim_volume_series: Simulated volume series
            asset_type: Type of asset
            
        Returns:
            Dictionary containing both real and simulated data
        """
        # Get real market data
        real_data, success = cls.get_historical_data(
            asset_type=asset_type,
            days=len(sim_time_index) // 24 + 1  # Add buffer day
        )
        
        result = {
            'sim_price': sim_price_series,
            'sim_volume': sim_volume_series,
            'time_index': sim_time_index
        }
        
        if success and not real_data.empty:
            # Align real data with simulation time index
            aligned_real = real_data.reindex(sim_time_index, method='ffill')
            
            # Scale real data to match simulation's price range
            if len(sim_price_series) > 0 and len(aligned_real) > 0:
                # Calculate scaling factor to match initial prices
                if sim_price_series[0] > 0 and aligned_real['Adj Close'].iloc[0] > 0:
                    scale_factor = sim_price_series[0] / aligned_real['Adj Close'].iloc[0]
                    scaled_prices = aligned_real['Adj Close'].values * scale_factor
                    scaled_volumes = aligned_real['Volume'].values * scale_factor
                    
                    result.update({
                        'real_price': scaled_prices[:len(sim_price_series)],
                        'real_volume': scaled_volumes[:len(sim_volume_series)],
                        'has_real_data': True
                    })
                    return result
        
        # Fallback if no real data or scaling failed
        result.update({
            'real_price': np.array([]),
            'real_volume': np.array([]),
            'has_real_data': False
        })
        return result
