"""
Visualization module for the KavachFi LP Fee Simulator.
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from enum import Enum

class AssetType(Enum):
    BTC = "Bitcoin"
    ETH = "Ethereum"
    SOL = "Solana"

class SimulationVisualizer:
    """
    Handles all visualization components for the LP fee simulation.
    """
    
    def __init__(self):
        """Initialize the visualizer with default styles and configurations."""
        # Color schemes for different assets
        self.asset_colors = {
            AssetType.BTC: {
                'primary': '#F7931A',  # Bitcoin Orange
                'secondary': '#FFC107',
                'success': '#4CAF50',
                'danger': '#F44336',
                'warning': '#FF9800',
                'info': '#00ACC1',
                'dark': '#212121',
                'light': '#F5F5F5'
            },
            AssetType.ETH: {
                'primary': '#627EEA',  # Ethereum Blue
                'secondary': '#C2C9FF',
                'success': '#4CAF50',
                'danger': '#F44336',
                'warning': '#FF9800',
                'info': '#00ACC1',
                'dark': '#212121',
                'light': '#F5F5F5'
            },
            AssetType.SOL: {
                'primary': '#00FFA3',  # Solana Green
                'secondary': '#03E1FF',
                'success': '#4CAF50',
                'danger': '#F44336',
                'warning': '#FF9800',
                'info': '#00ACC1',
                'dark': '#212121',
                'light': '#F5F5F5'
            }
        }
        
        # Default colors if asset type is not specified
        self.default_colors = {
            'primary': '#1E88E5',
            'secondary': '#FFC107',
            'success': '#4CAF50',
            'danger': '#F44336',
            'warning': '#FF9800',
            'info': '#00ACC1',
            'dark': '#212121',
            'light': '#F5F5F5'
        }
    
    def _get_colors(self, asset_type: Optional[AssetType] = None) -> Dict[str, str]:
        """Get color scheme for the specified asset type."""
        if asset_type and asset_type in self.asset_colors:
            return self.asset_colors[asset_type]
        return self.default_colors
    
    def create_price_chart(
        self,
        time_index: pd.DatetimeIndex,
        price_series: np.ndarray,
        asset_type: Optional[AssetType] = None,
        title: str = "Price Movement"
    ) -> go.Figure:
        """
        Create a line chart showing price movement over time.
        
        Args:
            time_index: Datetime index for x-axis
            price_series: Array of price data
            asset_type: Optional asset type for color scheme
            title: Chart title
            
        Returns:
            plotly.graph_objects.Figure: The generated figure
        """
        fig = go.Figure()
        
        colors = self._get_colors(asset_type)
        
        fig.add_trace(
            go.Scatter(
                x=time_index,
                y=price_series,
                mode='lines',
                name='Price',
                line=dict(color='#2E86AB')
            )
        )
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Price ($)',
            template='plotly_dark',
            margin=dict(l=20, r=20, t=40, b=20),
            height=300
        )
        
        return fig
    
    def create_volume_chart(
        self,
        time_index: pd.DatetimeIndex,
        volume_series: np.ndarray,
        liquidation_events: Optional[np.ndarray] = None,
        title: str = "Trading Volume & Liquidations"
    ) -> go.Figure:
        """
        Create a bar chart showing trading volume and liquidation events.
        
        Args:
            time_index: Datetime index for x-axis
            volume_series: Array of volume data
            liquidation_events: Optional array of liquidation events
            title: Chart title
            
        Returns:
            plotly.graph_objects.Figure: The generated figure
        """
        fig = go.Figure()
        
        # Add volume bars
        fig.add_trace(
            go.Bar(
                x=time_index,
                y=volume_series,
                name='Trading Volume',
                marker_color='#3BB273',
                opacity=0.7
            )
        )
        
        # Add liquidation events if provided
        if liquidation_events is not None and liquidation_events.sum() > 0:
            # Only show non-zero liquidations
            mask = liquidation_events > 0
            fig.add_trace(
                go.Scatter(
                    x=time_index[mask],
                    y=liquidation_events[mask],
                    mode='markers',
                    name='Liquidation Events',
                    marker=dict(
                        color='#E15554',
                        size=8,
                        line=dict(width=1, color='DarkSlateGrey')
                    )
                )
            )
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Volume ($)',
            barmode='stack',
            template='plotly_dark',
            margin=dict(l=20, r=20, t=40, b=20),
            height=300,
            showlegend=True
        )
        
        return fig
    
    def create_apr_chart(
        self,
        time_index: pd.DatetimeIndex,
        apr_series: np.ndarray,
        title: str = "LP APR Over Time"
    ) -> go.Figure:
        """
        Create a line chart showing LP APR over time.
        
        Args:
            time_index: Datetime index for x-axis
            apr_series: Array of APR values (as decimals)
            title: Chart title
            
        Returns:
            plotly.graph_objects.Figure: The generated figure
        """
        # Convert APR to percentage for display
        apr_percent = apr_series * 100
        
        fig = go.Figure()
        
        # Add APR line
        fig.add_trace(
            go.Scatter(
                x=time_index,
                y=apr_percent,
                mode='lines',
                name='APR',
                line=dict(color='#F7B801')
            )
        )
        
        # Add a horizontal line at the average APR
        avg_apr = apr_percent.mean()
        fig.add_hline(
            y=avg_apr,
            line_dash='dash',
            line_color='#F7B801',
            opacity=0.5,
            annotation_text=f'Avg: {avg_apr:.1f}%',
            annotation_position='bottom right'
        )
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='APR (%)',
            template='plotly_dark',
            margin=dict(l=20, r=20, t=40, b=20),
            height=300
        )
        
        return fig
    
    def create_revenue_breakdown_chart(
        self,
        time_index: pd.DatetimeIndex,
        trading_fees: np.ndarray,
        spread_revenue: np.ndarray,
        title: str = "Revenue Breakdown"
    ) -> go.Figure:
        """
        Create a stacked area chart showing the breakdown of revenue sources.
        
        Args:
            time_index: Datetime index for x-axis
            trading_fees: Array of trading fee revenue
            spread_revenue: Array of spread revenue
            title: Chart title
            
        Returns:
            plotly.graph_objects.Figure: The generated figure
        """
        fig = go.Figure()
        
        # Add trading fees
        fig.add_trace(
            go.Scatter(
                x=time_index,
                y=trading_fees,
                mode='lines',
                name='Trading Fees',
                stackgroup='one',
                line=dict(width=0.5, color='#3BB273')
            )
        )
        
        # Add spread revenue on top
        fig.add_trace(
            go.Scatter(
                x=time_index,
                y=spread_revenue,
                mode='lines',
                name='Spread Revenue',
                stackgroup='one',
                line=dict(width=0.5, color='#E15554')
            )
        )
        
        # Calculate and add total revenue line
        total_revenue = trading_fees + spread_revenue
        fig.add_trace(
            go.Scatter(
                x=time_index,
                y=total_revenue,
                mode='lines',
                name='Total Revenue',
                line=dict(color='#F7B801', width=2)
            )
        )
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Revenue ($)',
            showlegend=True,
            template='plotly_dark',
            margin=dict(l=20, r=20, t=40, b=20),
            height=300
        )
        
        return fig
    
    def create_volatility_vs_revenue_chart(
        self,
        price_series: np.ndarray,
        total_revenue: np.ndarray,
        window: int = 24,  # 24-hour rolling window
        title: str = "Volatility vs. Revenue"
    ) -> go.Figure:
        """
        Create a scatter plot of volatility vs. revenue.
        
        Args:
            price_series: Array of price data
            total_revenue: Array of total revenue data
            window: Window size for volatility calculation
            title: Chart title
            
        Returns:
            plotly.graph_objects.Figure: The generated figure
        """
        # Calculate rolling volatility (annualized)
        returns = np.diff(np.log(price_series))
        rolling_vol = np.zeros_like(price_series)
        rolling_vol[1:] = pd.Series(returns).rolling(window=window-1).std() * np.sqrt(365 * 24)  # Annualized
        
        # Remove NaN values
        mask = ~np.isnan(rolling_vol) & ~np.isnan(total_revenue)
        if mask.sum() == 0:
            # Fallback if no valid data points
            return go.Figure()
            
        rolling_vol = rolling_vol[mask]
        total_revenue = total_revenue[mask]
        
        # Create figure
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=rolling_vol * 100,  # Convert to percentage
                y=total_revenue,
                mode='markers',
                marker=dict(
                    size=8,
                    color=total_revenue,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Revenue ($)')
                ),
                hovertemplate='Volatility: %{x:.1f}%<br>Revenue: $%{y:,.0f}<extra></extra>'
            )
        )
        
        # Add trendline
        if len(rolling_vol) > 1:
            z = np.polyfit(rolling_vol * 100, total_revenue, 1)
            p = np.poly1d(z)
            fig.add_trace(
                go.Scatter(
                    x=np.linspace(rolling_vol.min() * 100, rolling_vol.max() * 100, 100),
                    y=p(np.linspace(rolling_vol.min() * 100, rolling_vol.max() * 100, 100)),
                    mode='lines',
                    line=dict(color='#F7B801', dash='dash'),
                    name='Trend',
                    showlegend=False
                )
            )
        
        fig.update_layout(
            title=title,
            xaxis_title='Annualized Volatility (%)',
            yaxis_title='Revenue ($)',
            template='plotly_dark',
            margin=dict(l=20, r=20, t=40, b=20),
            height=350,
            showlegend=False
        )
        
        return fig
    
    def create_eth_scenario_comparison(
        self,
        scenario_names: List[str],
        scenario_metrics: Dict[str, Dict[str, float]],
        title: str = "ETH Scenario Comparison"
    ) -> go.Figure:
        """
        Create a bar chart comparing key metrics across different ETH scenarios.
        
        Args:
            scenario_names: List of scenario names
            scenario_metrics: Dictionary mapping scenario names to metric dictionaries
            title: Chart title
            
        Returns:
            plotly.graph_objects.Figure: The generated figure
        """
        metrics = list(scenario_metrics[scenario_names[0]].keys())
        colors = self.asset_colors[AssetType.ETH]
        
        fig = go.Figure()
        
        for i, metric in enumerate(metrics):
            values = [scenario_metrics[scn][metric] for scn in scenario_names]
            fig.add_trace(go.Bar(
                name=metric,
                x=scenario_names,
                y=values,
                marker_color=colors[list(colors.keys())[i % len(colors)]]
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Scenario",
            yaxis_title="Value",
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_cumulative_revenue_chart(
        self,
        time_index: pd.DatetimeIndex,
        revenue_series: np.ndarray,
        title: str = "Cumulative LP Revenue"
    ) -> go.Figure:
        """
        Create a line chart showing cumulative LP revenue over time.
        
        Args:
            time_index: Datetime index for x-axis
            revenue_series: Array of cumulative revenue values
            title: Chart title
            
        Returns:
            plotly.graph_objects.Figure: The generated figure
        """
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=time_index,
                y=revenue_series,
                mode='lines',
                name='Cumulative Revenue',
                line=dict(color='#9B5DE5', width=2)
            )
        )
        
        # Add final value annotation
        if len(revenue_series) > 0:
            fig.add_annotation(
                x=time_index[-1],
                y=revenue_series[-1],
                text=f"${revenue_series[-1]:,.0f}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40
            )
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Cumulative Revenue ($)',
            template='plotly_dark',
            margin=dict(l=20, r=20, t=40, b=20),
            height=300
        )
        
        return fig
    
    def create_eth_scenario_comparison(
        self,
        scenario_names: List[str],
        scenario_metrics: Dict[str, Dict[str, float]],
        title: str = "ETH Scenario Comparison"
    ) -> go.Figure:
        """
        Create a bar chart comparing key metrics across different ETH scenarios.
        
        Args:
            scenario_names: List of scenario names
            scenario_metrics: Dictionary mapping scenario names to metric dictionaries
            title: Chart title
            
        Returns:
            plotly.graph_objects.Figure: The generated figure
        """
        # Define metrics to display and their formatting
        metrics = [
            ('APY', 'APY', '{:,.2f}%', 'primary'),
            ('Total Fees', 'Total Fees', '${:,.0f}', 'secondary'),
            ('Liquidation Events', 'Liquidation Events', '{:,.0f}', 'danger'),
            ('Avg. Daily Volume', 'Avg. Daily Volume', '${:,.0f}M', 'info')
        ]
        
        # Create subplots
        fig = make_subplots(
            rows=1, 
            cols=len(metrics),
            subplot_titles=[m[1] for m in metrics],
            horizontal_spacing=0.05
        )
        
        # Get colors for the current asset type (ETH)
        colors = self._get_colors(AssetType.ETH)
        
        # Add a trace for each metric
        for i, (metric_key, metric_name, text_format, color_key) in enumerate(metrics, 1):
            values = [scenario_metrics[name].get(metric_key, 0) for name in scenario_names]
            
            # Format values for display
            text_values = []
            for val in values:
                if 'M' in text_format and val >= 1_000_000:
                    text_values.append(f"${val/1_000_000:,.1f}M")
                elif 'M' in text_format:
                    text_values.append(f"${val/1_000:,.0f}K")
                else:
                    text_values.append(text_format.format(val))
            
            # Add bar trace
            fig.add_trace(
                go.Bar(
                    x=scenario_names,
                    y=values,
                    name=metric_name,
                    text=text_values,
                    textposition='auto',
                    marker_color=colors[color_key],
                    showlegend=False
                ),
                row=1, col=i
            )
            
            # Update y-axis title
            fig.update_yaxes(title_text=metric_name, row=1, col=i)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=16, color=colors['dark'])
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(t=60, b=20, l=20, r=20),
            height=500,
            showlegend=False
        )
        
        return fig

    def create_metric_cards(
        self,
        apr_series: np.ndarray,
        total_revenue: np.ndarray,
        lp_pool_size: float,
        simulation_days: int = 7
    ) -> Dict[str, float]:
        """
        Calculate key metrics for display in the dashboard.
        
        Args:
            apr_series: Array of APR values (as decimals)
            total_revenue: Array of total revenue values
            lp_pool_size: Total size of the LP pool
            simulation_days: Number of days in the simulation
            
        Returns:
            dict: Dictionary of metric names and values
        """
        if len(apr_series) == 0 or len(total_revenue) == 0:
            return {}
            
        # Calculate metrics
        avg_apr = np.mean(apr_series) * 100  # Convert to percentage
        max_apr = np.max(apr_series) * 100
        min_apr = np.min(apr_series) * 100
        total_revenue_sum = np.sum(total_revenue)
        revenue_per_million = (total_revenue_sum / lp_pool_size) * 1_000_000 if lp_pool_size > 0 else 0
        
        # Annualize the revenue if simulation is less than a year
        if simulation_days < 365:
            annualization_factor = 365 / simulation_days
            projected_annual_revenue = total_revenue_sum * annualization_factor
            projected_apr = (projected_annual_revenue / lp_pool_size) * 100 if lp_pool_size > 0 else 0
        else:
            projected_annual_revenue = total_revenue_sum * (365 / simulation_days)
            projected_apr = avg_apr
        
        return {
            'avg_apr': avg_apr,
            'max_apr': max_apr,
            'min_apr': min_apr,
            'total_revenue': total_revenue_sum,
            'revenue_per_million': revenue_per_million,
            'projected_annual_revenue': projected_annual_revenue,
            'projected_apr': projected_apr
        }
