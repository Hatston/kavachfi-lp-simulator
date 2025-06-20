"""
Fee calculation engine for LP returns.
"""
import numpy as np
from typing import Dict, Tuple

class FeeCalculator:
    """
    Calculates LP fees and returns based on market conditions and parameters.
    """
    
    def __init__(self, lp_pool_size: float, fee_rate: float, spread: float, insurance_fee_rate: float):
        """
        Initialize the FeeCalculator with LP pool and fee parameters.
        
        Args:
            lp_pool_size: Total size of the LP pool in USD
            fee_rate: Trading fee rate (e.g., 0.001 for 0.1%)
            spread: Base spread for spread revenue calculation
            insurance_fee_rate: Insurance fee rate on trading volume
        """
        self.lp_pool_size = lp_pool_size
        self.fee_rate = fee_rate
        self.spread = spread
        self.insurance_fee_rate = insurance_fee_rate
        
        # Initialize result attributes
        self.trading_fees = None
        self.spread_revenue = None
        self.insurance_fees = None
        self.total_revenue = None
        self.cumulative_fees = None
        self.apr_series = None
        
    def calculate_trading_fees(
        self,
        volume_series: np.ndarray,
        liquidation_events: np.ndarray,
        trading_fee_rate: float
    ) -> np.ndarray:
        """
        Calculate trading fees from regular volume and liquidation events.
        
        Args:
            volume_series: Array of trading volumes
            liquidation_events: Array of liquidation event sizes
            trading_fee_rate: Trading fee rate (e.g., 0.001 for 0.1%)
            
        Returns:
            np.ndarray: Trading fees per period
        """
        # Calculate fees from regular trading volume
        regular_fees = volume_series * trading_fee_rate
        
        # Calculate fees from liquidation events (assume same fee rate)
        liquidation_fees = liquidation_events * trading_fee_rate
        
        return regular_fees + liquidation_fees
    
    def calculate_spread_revenue(
        self,
        volume_series: np.ndarray,
        price_series: np.ndarray,
        base_spread: float = 0.0005,
        vol_to_spread_impact: float = 0.1
    ) -> np.ndarray:
        """
        Calculate spread revenue based on volume and volatility.
        
        Args:
            volume_series: Array of trading volumes
            price_series: Array of prices
            base_spread: Base spread (as a fraction of price)
            vol_to_spread_impact: How much volatility affects the spread
            
        Returns:
            np.ndarray: Spread revenue per period
        """
        # Calculate price returns as a proxy for volatility
        returns = np.diff(price_series) / price_series[:-1]
        # Pad with first value to match length
        returns = np.concatenate(([returns[0]], returns))
        
        # Calculate dynamic spread: wider in volatile markets
        dynamic_spread = base_spread * (1 + vol_to_spread_impact * np.abs(returns) / base_spread)
        
        # Spread revenue is volume * spread
        return volume_series * dynamic_spread
    
    def calculate_apr(
        self,
        revenue_series: np.ndarray,
        lp_pool_size: float,
        periods_per_year: int = 24 * 365  # Hourly data
    ) -> np.ndarray:
        """
        Calculate APR series based on revenue and LP pool size.
        
        Args:
            revenue_series: Array of revenue per period
            lp_pool_size: Total size of the LP pool
            periods_per_year: Number of periods in a year
            
        Returns:
            np.ndarray: APR series (annualized)
        """
        # Calculate period return (as a decimal)
        period_return = revenue_series / lp_pool_size
        
        # Annualize the return
        return period_return * periods_per_year
    
    def calculate_fees(
        self,
        price_series: np.ndarray,
        volume_series: np.ndarray,
        liquidation_events: np.ndarray,
        periods_per_year: int = 24 * 365
    ) -> Dict[str, np.ndarray]:
        """
        Calculate all fees and returns using instance parameters.
        
        Args:
            price_series: Array of prices
            volume_series: Array of trading volumes
            liquidation_events: Array of liquidation event sizes
            periods_per_year: Number of periods in a year
            
        Returns:
            dict: Dictionary containing all calculated metrics
        """
        # Calculate trading fees
        trading_fees = self.calculate_trading_fees(
            volume_series=volume_series,
            liquidation_events=liquidation_events,
            trading_fee_rate=self.fee_rate
        )
        
        # Calculate spread revenue
        spread_revenue = self.calculate_spread_revenue(
            volume_series=volume_series,
            price_series=price_series,
            base_spread=self.spread,
            vol_to_spread_impact=0.1  # Default value
        )
        
        # Calculate insurance fees
        self.insurance_fees = volume_series * self.insurance_fee_rate
        
        # Calculate total revenue (trading fees + spread revenue - insurance fees)
        total_revenue = trading_fees + spread_revenue - self.insurance_fees
        
        # Calculate cumulative fees
        cumulative_fees = np.cumsum(total_revenue)
        
        # Calculate APR series
        self.apr_series = self.calculate_apr(
            revenue_series=total_revenue,
            lp_pool_size=self.lp_pool_size,
            periods_per_year=periods_per_year
        )
        
        # Store results
        self.trading_fees = trading_fees
        self.spread_revenue = spread_revenue
        self.total_revenue = total_revenue
        self.cumulative_fees = cumulative_fees
        
        return {
            'trading_fees': trading_fees,
            'spread_revenue': spread_revenue,
            'insurance_fees': self.insurance_fees,
            'total_revenue': total_revenue,
            'cumulative_fees': cumulative_fees,
            'apr_series': self.apr_series
        }
        
    def run_calculations(
        self,
        volume_series: np.ndarray,
        price_series: np.ndarray,
        liquidation_events: np.ndarray,
        trading_fee_rate: float,
        lp_pool_size: float,
        base_spread: float = 0.0005,
        vol_to_spread_impact: float = 0.1,
        periods_per_year: int = 24 * 365
    ) -> Dict[str, np.ndarray]:
        """
        Run all fee and return calculations (legacy method).
        
        This is kept for backward compatibility but delegates to calculate_fees.
        """
        # Update instance variables if different from constructor
        self.fee_rate = trading_fee_rate
        self.lp_pool_size = lp_pool_size
        self.spread = base_spread
        
        return self.calculate_fees(
            price_series=price_series,
            volume_series=volume_series,
            liquidation_events=liquidation_events,
            periods_per_year=periods_per_year
        )
