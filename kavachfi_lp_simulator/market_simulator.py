"""
Market simulation engine for generating realistic price and volume data for multiple assets.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any

class AssetType(Enum):
    BTC = "Bitcoin"
    ETH = "Ethereum"
    SOL = "Solana"

@dataclass
class AssetParameters:
    """Parameters specific to each asset type."""
    base_vol: float  # Base daily volatility
    price_range: Tuple[float, float]  # Typical price range (min, max)
    liquidation_threshold: float  # Price move % that triggers liquidations
    volume_mult: float  # Volume multiplier for this asset
    vol_cluster_strength: float  # How much volatility clusters (0-1)
    volume_vol: float  # Volatility of volume (0-1)
    volume_correlation: float  # How volume correlates with volatility (0-1)
    base_liquidation_size: float  # Base size of liquidation events
    liquidation_prob: float  # Base probability of liquidation (0-1)

# Asset-specific parameters
ASSET_PARAMS = {
    AssetType.BTC: AssetParameters(
        base_vol=0.12,
        price_range=(20000, 70000),
        liquidation_threshold=0.07,  # 7% moves trigger liquidations
        volume_mult=1.0,
        vol_cluster_strength=0.7,
        volume_vol=0.5,
        volume_correlation=0.6,
        base_liquidation_size=50000,
        liquidation_prob=0.05
    ),
    AssetType.ETH: AssetParameters(
        base_vol=0.18,
        price_range=(1500, 4500),
        liquidation_threshold=0.05,  # 5% moves trigger liquidations
        volume_mult=1.2,
        vol_cluster_strength=0.8,  # Higher persistence than BTC
        volume_vol=0.6,
        volume_correlation=0.7,
        base_liquidation_size=30000,
        liquidation_prob=0.07
    ),
    AssetType.SOL: AssetParameters(
        base_vol=0.25,
        price_range=(20, 200),
        liquidation_threshold=0.10,  # 10% moves trigger liquidations
        volume_mult=1.5,
        vol_cluster_strength=0.9,
        volume_vol=0.8,
        volume_correlation=0.8,
        base_liquidation_size=10000,
        liquidation_prob=0.1
    )
}

# ETH-specific scenarios
ETH_SCENARIOS = {
    "ETH Shanghai Upgrade": {
        "volatility_multiplier": 1.5,
        "volume_multiplier": 1.8,
        "liquidation_impact": 1.2,
        "description": "Simulates market conditions around the Shanghai upgrade"
    },
    "ETH Bear Market 2022": {
        "volatility_multiplier": 2.0,
        "volume_multiplier": 1.5,
        "liquidation_impact": 1.5,
        "description": "Simulates the high volatility and liquidations of the 2022 bear market"
    },
    "ETH DeFi Summer": {
        "volatility_multiplier": 1.3,
        "volume_multiplier": 2.5,
        "liquidation_impact": 0.8,
        "description": "Simulates high volume, lower volatility conditions of DeFi Summer"
    },
    "ETH Merge Event": {
        "volatility_multiplier": 1.7,
        "volume_multiplier": 2.0,
        "liquidation_impact": 1.3,
        "description": "Simulates the market around the Ethereum Merge event"
    },
    "ETH Consolidation": {
        "volatility_multiplier": 0.7,
        "volume_multiplier": 0.8,
        "liquidation_impact": 0.6,
        "description": "Simulates low volatility, low volume consolidation"
    }
}

class MarketSimulator:
    """
    Simulates market conditions including price movements, trading volume, and liquidation events
    for different crypto assets.
    """
    
    def __init__(self):
        self.price_series = None
        self.volume_series = None
        self.liquidation_events = None
        self.asset_type: Optional[AssetType] = None
        self.asset_params: Optional[AssetParameters] = None
        self.current_scenario: Optional[Dict[str, Any]] = None
        
    def generate_price_series(
        self,
        periods: int,
        initial_price: float = 50000.0,
        base_vol: float = 0.01,
        vol_cluster_strength: float = 0.7,
        seed: int = None
    ) -> np.ndarray:
        """
        Generate a price series with volatility clustering.
        
        Args:
            periods: Number of periods to simulate
            initial_price: Starting price
            base_vol: Base volatility (standard deviation of returns)
            vol_cluster_strength: Strength of volatility clustering (0-1)
            seed: Random seed for reproducibility
            
        Returns:
            np.ndarray: Simulated price series
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Generate random returns with time-varying volatility
        returns = np.zeros(periods)
        vol = np.ones(periods) * base_vol
        
        for t in range(1, periods):
            # Update volatility based on previous period's return
            vol[t] = base_vol * (1 + vol_cluster_strength * np.abs(returns[t-1]) / base_vol)
            # Generate return with current volatility
            returns[t] = np.random.normal(0, vol[t])
            
        # Convert returns to price series
        price_series = initial_price * (1 + returns).cumprod()
        return price_series
    
    def generate_volume_series(
        self,
        periods: int,
        base_volume: float,
        price_series: np.ndarray,
        volume_vol: float = 0.5,
        correlation_strength: float = 0.6,
        seed: int = None
    ) -> np.ndarray:
        """
        Generate trading volume series correlated with price volatility.
        
        Args:
            periods: Number of periods to simulate
            base_volume: Base trading volume
            price_series: Generated price series
            volume_vol: Volume volatility
            correlation_strength: Strength of volume-volatility correlation (0-1)
            seed: Random seed for reproducibility
            
        Returns:
            np.ndarray: Simulated volume series
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Calculate price returns and their absolute values (as volatility proxy)
        returns = np.diff(price_series) / price_series[:-1]
        vol_proxy = np.abs(returns)
        # Pad with first value to match length
        vol_proxy = np.concatenate(([vol_proxy[0]], vol_proxy))
        
        # Generate base volume with some randomness
        base_vol_series = base_volume * np.exp(np.random.normal(0, volume_vol, periods))
        
        # Correlate volume with volatility
        volume_series = base_vol_series * (1 + correlation_strength * (vol_proxy / vol_proxy.mean() - 1))
        
        # Ensure volume is positive
        return np.maximum(volume_series, 0.1 * base_volume)
    
    def simulate_liquidation_events(
        self,
        price_series: np.ndarray,
        base_liquidation_size: float,
        liquidation_prob: float = 0.05,
        min_price_move: float = 0.03,
        seed: int = None
    ) -> np.ndarray:
        """
        Simulate liquidation events based on large price moves.
        
        Args:
            price_series: Generated price series
            base_liquidation_size: Base size of liquidation events
            liquidation_prob: Base probability of liquidation per period
            min_price_move: Minimum price move to trigger liquidation
            seed: Random seed for reproducibility
            
        Returns:
            np.ndarray: Array of liquidation sizes (0 for no liquidation)
        """
        if seed is not None:
            np.random.seed(seed)
            
        periods = len(price_series)
        liquidation_sizes = np.zeros(periods)
        
        # Calculate price returns
        returns = np.diff(price_series) / price_series[:-1]
        returns = np.concatenate(([0], returns))  # Pad with 0 for first period
        
        # Identify periods with large price moves
        large_moves = np.abs(returns) > min_price_move
        
        # Generate liquidation events
        for t in range(periods):
            if large_moves[t] and np.random.rand() < liquidation_prob:
                # Larger price moves result in larger liquidations
                size_multiplier = 1 + np.abs(returns[t]) / min_price_move
                liquidation_sizes[t] = base_liquidation_size * size_multiplier * np.random.lognormal(0, 0.5)
                
        return liquidation_sizes
    
    def set_asset(self, asset_type: AssetType) -> None:
        """Set the asset type and load its parameters."""
        self.asset_type = asset_type
        self.asset_params = ASSET_PARAMS[asset_type]
    
    def get_available_eth_scenarios(self) -> List[str]:
        """Return list of available ETH-specific scenarios."""
        return list(ETH_SCENARIOS.keys())
        
    def apply_eth_scenario(self, scenario_name: str) -> None:
        """
        Apply ETH-specific scenario parameters.
        
        Args:
            scenario_name: Name of the scenario to apply
        """
        if self.asset_type != AssetType.ETH or scenario_name not in ETH_SCENARIOS:
            return
            
        # Store the current scenario
        self.current_scenario = ETH_SCENARIOS[scenario_name].copy()
        scenario = self.current_scenario
        
        # Apply scenario multipliers to base parameters
        self.asset_params.base_vol *= scenario["volatility_multiplier"]
        self.asset_params.volume_mult *= scenario["volume_multiplier"]
        self.asset_params.liquidation_prob *= scenario["liquidation_impact"]
        
        # Additional scenario-specific adjustments
        if scenario_name == "ETH Shanghai Upgrade":
            # Higher volume-vol correlation during upgrades
            self.asset_params.volume_correlation = min(0.9, self.asset_params.volume_correlation * 1.2)
        elif scenario_name == "ETH Bear Market 2022":
            # More frequent and severe liquidations
            self.asset_params.liquidation_threshold *= 0.8
            self.asset_params.base_liquidation_size *= 1.3
        elif scenario_name == "ETH DeFi Summer":
            # Higher volume persistence
            self.asset_params.vol_cluster_strength = min(0.9, self.asset_params.vol_cluster_strength * 1.3)
        elif scenario_name == "ETH Merge Event":
            # Increased volatility clustering
            self.asset_params.vol_cluster_strength = min(0.95, self.asset_params.vol_cluster_strength * 1.5)
        elif scenario_name == "ETH Consolidation":
            # Reduced volatility and volume
            self.asset_params.volume_vol *= 0.7
            self.asset_params.liquidation_threshold *= 1.3
            
    def run_simulation(
        self,
        asset_type: AssetType = AssetType.BTC,
        periods: int = 24 * 7,  # 1 week of hourly data
        initial_price: Optional[float] = None,
        base_volume: Optional[float] = None,
        base_vol: Optional[float] = None,
        vol_cluster_strength: Optional[float] = None,
        volume_vol: Optional[float] = None,
        volume_correlation: Optional[float] = None,
        base_liquidation_size: Optional[float] = None,
        liquidation_prob: Optional[float] = None,
        min_price_move: Optional[float] = None,
        eth_scenario: Optional[str] = None,
        seed: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        Run complete market simulation for the specified asset.
        
        Args:
            asset_type: Type of asset to simulate (BTC, ETH, SOL)
            periods: Number of periods to simulate
            initial_price: Initial price of the asset (if None, uses asset's typical range)
            base_volume: Base trading volume (if None, uses asset's default)
            base_vol: Base volatility (if None, uses asset's default)
            vol_cluster_strength: Strength of volatility clustering (if None, uses asset's default)
            volume_vol: Volume volatility (if None, uses asset's default)
            volume_correlation: Correlation between volume and volatility (if None, uses asset's default)
            base_liquidation_size: Base size of liquidation events (if None, uses asset's default)
            liquidation_prob: Base probability of liquidation per period (if None, uses asset's default)
            min_price_move: Minimum price move to trigger liquidation (if None, uses asset's threshold)
            eth_scenario: Optional ETH-specific scenario to apply
            seed: Random seed for reproducibility
            
        Returns:
            dict: Dictionary containing price_series, volume_series, liquidation_events, and metadata
        """
        # Reset current scenario
        self.current_scenario = None
        # Set asset type and load parameters
        self.set_asset(asset_type)
        params = self.asset_params
        
        # Apply ETH scenario if specified
        if eth_scenario and asset_type == AssetType.ETH:
            self.apply_eth_scenario(eth_scenario)
        
        # Use provided parameters or fall back to asset defaults
        initial_price = initial_price or np.mean(params.price_range)
        base_volume = base_volume or (1000000 * params.volume_mult)  # $1M base, adjusted by asset
        base_vol = base_vol or params.base_vol
        vol_cluster_strength = vol_cluster_strength or params.vol_cluster_strength
        volume_vol = volume_vol or params.volume_vol
        volume_correlation = volume_correlation or params.volume_correlation
        base_liquidation_size = base_liquidation_size or params.base_liquidation_size
        liquidation_prob = liquidation_prob or params.liquidation_prob
        min_price_move = min_price_move or params.liquidation_threshold
        
        # Generate price series with volatility clustering
        price_series = self.generate_price_series(
            periods=periods,
            initial_price=initial_price,
            base_vol=base_vol,
            vol_cluster_strength=vol_cluster_strength,
            seed=seed
        )
        
        # Generate volume series correlated with volatility
        volume_series = self.generate_volume_series(
            periods=periods,
            base_volume=base_volume,
            price_series=price_series,
            volume_vol=volume_vol,
            correlation_strength=volume_correlation,
            seed=seed
        )
        
        # Simulate liquidation events
        liquidation_events = self.simulate_liquidation_events(
            price_series=price_series,
            base_liquidation_size=base_liquidation_size,
            liquidation_prob=liquidation_prob,
            min_price_move=min_price_move,
            seed=seed
        )
        
        # Store results
        self.price_series = price_series
        self.volume_series = volume_series
        self.liquidation_events = liquidation_events
        
        return {
            'price_series': price_series,
            'volume_series': volume_series,
            'liquidation_events': liquidation_events,
            'asset_type': asset_type.value,
            'asset_name': asset_type.name,
            'scenario': eth_scenario if eth_scenario and asset_type == AssetType.ETH else None,
            'parameters': {
                'base_vol': base_vol,
                'volume_mult': params.volume_mult,
                'liquidation_threshold': min_price_move,
                'base_volume': base_volume
            }
        }
    
    def get_available_eth_scenarios(self) -> List[str]:
        """Return list of available ETH-specific scenarios."""
        return list(ETH_SCENARIOS.keys())
