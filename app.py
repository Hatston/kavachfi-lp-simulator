"""
KavachFi LP Fee Simulation Dashboard

This Streamlit application simulates Liquidity Provider (LP) fees and returns
for the KavachFi perpetual DEX with CLOB and native insurance on Rise Chain.
"""
import base64
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta

from kavachfi_lp_simulator.market_simulator import MarketSimulator, AssetType
from kavachfi_lp_simulator.fee_calculator import FeeCalculator
from kavachfi_lp_simulator.visualization import SimulationVisualizer
from kavachfi_lp_simulator.market_data import MarketDataFetcher, AssetType as MarketAssetType

# Set page config
st.set_page_config(
    page_title="KavachFi LP Fee Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load custom CSS
try:
    local_css("style.css")
except:
    pass  # Use default styling if custom CSS is not found

# Initialize session state for simulation results
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None

# Initialize simulation components with default values
market_simulator = MarketSimulator()
# Initialize with default values that will be updated during simulation
fee_calculator = FeeCalculator(
    lp_pool_size=1000000,  # Default 1M pool size
    fee_rate=0.001,        # Default 0.1% fee rate
    spread=0.0005,         # Default 5 bps spread
    insurance_fee_rate=0.0005  # Default 5 bps insurance fee rate
)
visualizer = SimulationVisualizer()

# Sidebar - Simulation Parameters
with st.sidebar:
    st.title("⚙️ Simulation Parameters")
    
    # Asset Selection
    asset_type = st.selectbox(
        "Select Asset",
        options=["BTC", "ETH", "SOL"],
        index=0,
        help="Select the asset to simulate"
    )
    
    # Convert to AssetType enum
    from kavachfi_lp_simulator.market_simulator import AssetType
    asset_type = AssetType[asset_type]
    
    # ETH-specific scenarios
    eth_scenario = None
    if asset_type == AssetType.ETH:
        from kavachfi_lp_simulator.market_simulator import MarketSimulator
        simulator = MarketSimulator()
        eth_scenarios = simulator.get_available_eth_scenarios()
        eth_scenario = st.selectbox(
            "ETH Scenario (Optional)",
            options=["None"] + eth_scenarios,
            index=0,
            help="Select an ETH-specific scenario"
        )
        if eth_scenario == "None":
            eth_scenario = None
    
    # Simulation Presets
    st.subheader("🔧 Market Presets")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🐂 Bull Market"):
            st.session_state.base_volume = 2_000_000
            st.session_state.trading_fee = 0.0008  # 0.08%
            st.session_state.lp_pool_size = 5_000_000
            st.session_state.volatility = 0.015
            st.session_state.liquidation_size = 100_000
    with col2:
        if st.button("🐻 Bear Market"):
            st.session_state.base_volume = 3_000_000
            st.session_state.trading_fee = 0.0012  # 0.12%
            st.session_state.lp_pool_size = 3_000_000
            st.session_state.volatility = 0.025
            st.session_state.liquidation_size = 150_000
    
    col3, col4 = st.columns(2)
    with col3:
        if st.button("🔄 Sideways"):
            st.session_state.base_volume = 1_000_000
            st.session_state.trading_fee = 0.0005  # 0.05%
            st.session_state.lp_pool_size = 10_000_000
            st.session_state.volatility = 0.008
            st.session_state.liquidation_size = 50_000
    with col4:
        if st.button("💥 Market Crash"):
            st.session_state.base_volume = 5_000_000
            st.session_state.trading_fee = 0.002  # 0.2%
            st.session_state.lp_pool_size = 2_000_000
            st.session_state.volatility = 0.05
            st.session_state.liquidation_size = 300_000
    
    st.markdown("---")
    
    # Simulation Parameters
    st.subheader("📊 Market Parameters")
    
    # Initialize session state for parameters if they don't exist
    if 'base_volume' not in st.session_state:
        st.session_state.base_volume = 1_000_000
    if 'trading_fee' not in st.session_state:
        st.session_state.trading_fee = 0.001  # 0.1%
    if 'lp_pool_size' not in st.session_state:
        st.session_state.lp_pool_size = 5_000_000
    if 'volatility' not in st.session_state:
        st.session_state.volatility = 0.01
    if 'liquidation_size' not in st.session_state:
        st.session_state.liquidation_size = 100_000
    
    # Sliders with session state
    st.session_state.base_volume = st.slider(
        "Base Hourly Trading Volume ($)",
        min_value=10_000,
        max_value=10_000_000,
        value=st.session_state.base_volume,
        step=10_000,
        help="Base trading volume in USD per hour"
    )
    
    st.session_state.trading_fee = st.slider(
        "Trading Fee (%)",
        min_value=0.01,
        max_value=0.2,
        value=st.session_state.trading_fee * 100,  # Convert to percentage
        step=0.01,
        format="%.2f%%",
        help="Trading fee as a percentage of trading volume"
    ) / 100  # Convert back to decimal
    
    st.session_state.lp_pool_size = st.slider(
        "LP Pool Size ($)",
        min_value=100_000,
        max_value=50_000_000,
        value=st.session_state.lp_pool_size,
        step=100_000,
        help="Total size of the liquidity pool in USD"
    )
    
    st.session_state.volatility = st.slider(
        "Market Volatility (std dev of returns)",
        min_value=0.001,
        max_value=0.1,
        value=st.session_state.volatility,
        step=0.001,
        format="%.3f",
        help="Base volatility of the market (standard deviation of returns)"
    )
    
    st.session_state.liquidation_size = st.slider(
        "Base Liquidation Size ($)",
        min_value=1_000,
        max_value=500_000,
        value=st.session_state.liquidation_size,
        step=1_000,
        help="Base size of liquidation events in USD"
    )
    
    # Simulation duration
    simulation_days = st.slider(
        "Simulation Duration (days)",
        min_value=1,
        max_value=30,
        value=7,
        step=1,
        help="Number of days to simulate"
    )
    
    # Run simulation button
    if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
        with st.spinner("Running simulation..."):
            # Generate time index
            periods = simulation_days * 24  # Hourly data
            end_time = datetime.now()
            start_time = end_time - timedelta(days=simulation_days)
            time_index = pd.date_range(start=start_time, end=end_time, periods=periods)
            
            # Run simulation
            market_simulator = MarketSimulator()
            market_data = market_simulator.run_simulation(
                asset_type=asset_type,
                periods=periods,
                initial_price=st.session_state.base_volume / 24,  # Convert daily to hourly
                base_volume=st.session_state.base_volume,
                base_vol=st.session_state.volatility,
                vol_cluster_strength=0.5,
                volume_vol=0.5,
                volume_correlation=0.5,
                base_liquidation_size=st.session_state.liquidation_size,
                liquidation_prob=0.5,
                min_price_move=0.01,
                eth_scenario=eth_scenario if asset_type == AssetType.ETH else None
            )
            
            # Update fee calculator with current parameters
            fee_calculator.lp_pool_size = st.session_state.lp_pool_size
            fee_calculator.fee_rate = st.session_state.trading_fee
            fee_calculator.spread = 0.0005
            fee_calculator.insurance_fee_rate = 0.001
            
            # Calculate fees
            fee_data = fee_calculator.calculate_fees(
                price_series=market_data['price_series'],
                volume_series=market_data['volume_series'],
                liquidation_events=market_data['liquidation_events']
            )
            
            # Get real market data for comparison
            market_data_fetcher = MarketDataFetcher()
            market_asset_type = {
                AssetType.BTC: MarketAssetType.BTC,
                AssetType.ETH: MarketAssetType.ETH,
                AssetType.SOL: MarketAssetType.SOL
            }.get(asset_type, MarketAssetType.BTC)
            
            # Merge real and simulated data
            merged_data = market_data_fetcher.merge_with_simulation(
                sim_time_index=time_index,
                sim_price_series=market_data['price_series'],
                sim_volume_series=market_data['volume_series'],
                asset_type=market_asset_type
            )
            
            # Store results in session state
            st.session_state.simulation_results = {
                'time_index': time_index,
                'market_data': {
                    **market_data,
                    'real_price': merged_data.get('real_price', np.array([])),
                    'real_volume': merged_data.get('real_volume', np.array([])),
                    'has_real_data': merged_data.get('has_real_data', False)
                },
                'fee_data': fee_data,
                'simulation_params': {
                    'asset_type': asset_type,
                    'initial_price': st.session_state.base_volume / 24,
                    'base_volume': st.session_state.base_volume,
                    'lp_pool_size': st.session_state.lp_pool_size,
                    'fee_rate': st.session_state.trading_fee,
                    'spread': 0.0005,
                    'insurance_fee_rate': 0.001,
                    'simulation_days': simulation_days,
                    'eth_scenario': eth_scenario if asset_type == AssetType.ETH else None,
                    'has_real_data': merged_data.get('has_real_data', False)
                }
            }
            
            # If this is an ETH simulation with a scenario, run other scenarios for comparison
            if asset_type == AssetType.ETH and eth_scenario:
                st.session_state.eth_scenario_metrics = {}
                all_eth_scenarios = market_simulator.get_available_eth_scenarios()
                
                # Calculate metrics for the current scenario
                total_liquidations = np.sum(market_data['liquidation_events'] > 0)
                total_volume = np.sum(market_data['volume_series'])
                
                # Store current scenario results
                st.session_state.eth_scenario_metrics[eth_scenario] = {
                    'APY': np.mean(fee_data['apr_series']),  # Use average of APR series
                    'Total Fees': np.sum(fee_data['total_revenue']),
                    'Liquidation Events': total_liquidations,
                    'Avg. Daily Volume': total_volume / simulation_days
                }
                
                # Run other scenarios
                progress_bar = st.progress(0)
                total_scenarios = len(all_eth_scenarios)
                
                for i, scenario in enumerate(all_eth_scenarios):
                    if scenario != eth_scenario:  # Skip the one we already ran
                        progress_bar.progress((i + 1) / (total_scenarios), text=f"Running {scenario} scenario...")
                        
                        # Run simulation for this scenario
                        scenario_market_data = market_simulator.run_simulation(
                            asset_type=asset_type,
                            periods=periods,
                            initial_price=st.session_state.base_volume / 24,
                            base_volume=st.session_state.base_volume,
                            base_vol=st.session_state.volatility,
                            vol_cluster_strength=0.5,
                            volume_vol=0.5,
                            volume_correlation=0.5,
                            base_liquidation_size=st.session_state.liquidation_size,
                            liquidation_prob=0.5,
                            min_price_move=0.01,
                            eth_scenario=scenario,
                            seed=int(time.time()) + i + 1  # Different seed for each scenario
                        )
                        
                        # Calculate fees for this scenario
                        scenario_fee_data = fee_calculator.calculate_fees(
                            price_series=scenario_market_data['price_series'],
                            volume_series=scenario_market_data['volume_series'],
                            liquidation_events=scenario_market_data['liquidation_events']
                        )
                        
                        # Store scenario results
                        st.session_state.eth_scenario_metrics[scenario] = {
                            'APY': np.mean(scenario_fee_data['apr_series']),  # Use average of APR series
                            'Total Fees': np.sum(scenario_fee_data['total_revenue']),
                            'Liquidation Events': np.sum(scenario_market_data['liquidation_events'] > 0),
                            'Avg. Daily Volume': np.sum(scenario_market_data['volume_series']) / simulation_days
                        }
                
                progress_bar.empty()

# Main content
st.title("📊 KavachFi LP Fee Simulator")

# Display welcome message or simulation results
if 'simulation_results' not in st.session_state or st.session_state.simulation_results is None:
    # Welcome message
    st.markdown("""
    Welcome to the KavachFi Liquidity Provider Fee Simulator. 
    
    This tool helps you understand potential earnings and risks of providing liquidity on the KavachFi perpetual DEX.
    
    ### How to Use:
    1. Select an asset and adjust simulation parameters in the sidebar
    2. For ETH, choose a specific scenario (optional)
    3. Click 'Run Simulation' to generate results
    4. View the charts and metrics below
    5. Export results using the download button
    """)
    
    st.info("👈 Configure the simulation parameters in the sidebar and click 'Run Simulation' to begin.")
else:
    # Show simulation results
    results = st.session_state.simulation_results
    time_index = results['time_index']
    market_data = results['market_data']
    fee_data = results['fee_data']
    params = results['simulation_params']
    
    # Display asset and scenario info
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### 🔹 **Asset:** {params['asset_type']}")
    with col2:
        scenario = params.get('scenario', 'Custom')
        st.markdown(f"### 📊 **Scenario:** {scenario}")
    st.markdown("---")
    
    # Calculate metrics
    metrics = visualizer.create_metric_cards(
        apr_series=fee_data['apr_series'],
        total_revenue=fee_data['total_revenue'],
        lp_pool_size=params['lp_pool_size'],
        simulation_days=params['simulation_days']
    )
    
    # Display simulation results with asset and scenario info
    asset_name = market_data['asset_name']
    scenario_name = params.get('scenario', 'None')
    st.header(f"📊 {asset_name} Simulation Results" + (f" - {scenario_name}" if scenario_name != 'None' else ""))
    
    # Calculate metrics from the series data
    total_volume = np.sum(market_data['volume_series'])
    total_liquidations = np.sum(market_data['liquidation_events'] > 0)
    avg_apy = np.mean(fee_data['apr_series'])
    
    # Show key metrics with better formatting
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trading Volume", f"${total_volume:,.2f}", 
                 help="Total trading volume in USD over the simulation period")
    with col2:
        st.metric("Total Fees Earned", f"${np.sum(fee_data['total_revenue']):,.2f}",
                 help="Total fees earned by LPs in USD")
    with col3:
        st.metric("Total Liquidations", f"{total_liquidations:,}",
                 help="Number of liquidation events that occurred")
    with col4:
        st.metric("APY Estimate", f"{avg_apy:.2f}%",
                 help="Average Annual Percentage Yield for LPs based on simulation")
    
    # Create tabs for different visualizations
    tab_titles = [
        f"{asset_name} Price & Volume", 
        "Revenue Breakdown", 
        "Volatility Impact", 
        "Cumulative Revenue"
    ]
    
    # Add ETH Scenario Comparison tab if applicable
    if params['asset_type'] == 'ETH' and 'eth_scenario_metrics' in st.session_state:
        tab_titles.insert(1, "ETH Scenario Comparison")
    
    # Create tabs
    tabs = st.tabs(tab_titles)
    
    # Unpack tabs
    if len(tabs) == 5:  # With ETH Scenario Comparison
        tab1, tab_eth, tab2, tab3, tab4 = tabs
        
        # Add ETH scenario comparison in its own tab
        with tab_eth:
            st.subheader("ETH Scenario Comparison")
            st.markdown("""
                Compare how different ETH market scenarios would have performed 
                with the current simulation parameters. Each scenario applies different 
                market conditions based on historical ETH market behavior.
            """)
            
            # Add scenario descriptions
            scenario_descriptions = {
                "ETH Shanghai Upgrade": "Simulates market conditions around the Shanghai upgrade",
                "ETH Bear Market 2022": "Simulates the high volatility and liquidations of the 2022 bear market",
                "ETH DeFi Summer": "Simulates high volume, lower volatility conditions of DeFi Summer",
                "ETH Merge Event": "Simulates the market around the Ethereum Merge event",
                "ETH Consolidation": "Simulates low volatility, low volume consolidation"
            }
            
            # Show scenario descriptions in expanders
            with st.expander("ℹ️ Scenario Descriptions"):
                for scenario, desc in scenario_descriptions.items():
                    st.markdown(f"**{scenario}**: {desc}")
            
            # Show the scenario comparison chart
            scenario_fig = visualizer.create_eth_scenario_comparison(
                scenario_names=list(st.session_state.eth_scenario_metrics.keys()),
                scenario_metrics=st.session_state.eth_scenario_metrics,
                title="ETH Scenario Performance Comparison"
            )
            st.plotly_chart(scenario_fig, use_container_width=True)
            
            # Add some insights based on the comparison
            st.markdown("### Key Insights")
            
            # Find best and worst performing scenarios by APY
            scenarios = st.session_state.eth_scenario_metrics
            if scenarios:
                best_scenario = max(scenarios.items(), key=lambda x: x[1]['APY'])
                worst_scenario = min(scenarios.items(), key=lambda x: x[1]['APY'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Highest APY Scenario", 
                             f"{best_scenario[0]}", 
                             f"{best_scenario[1]['APY']:.2f}%")
                with col2:
                    st.metric("Lowest APY Scenario", 
                             f"{worst_scenario[0]}", 
                             f"{worst_scenario[1]['APY']:.2f}%")
                
                # Add some analysis
                st.markdown("""
                    **Note:** The scenarios show how different market conditions can impact LP returns. 
                    Higher volatility and trading volume typically lead to higher fees but may also 
                    increase the risk of liquidations and impermanent loss.
                """)
    else:  # Without ETH Scenario Comparison
        tab1, tab2, tab3, tab4 = tabs
    
    with tab1:
        # First row: Price and Volume
        col1, col2 = st.columns(2)
        
        with col1:
            # Convert asset name to AssetType enum
            asset_enum = {
                'BTC': AssetType.BTC,
                'ETH': AssetType.ETH,
                'SOL': AssetType.SOL
            }.get(asset_name, None)
            
            # Price chart
            price_fig = visualizer.create_price_chart(
                time_index=time_index,
                price_series=market_data['price_series'],
                asset_type=asset_enum,
                title=f"{asset_name} Price Movement",
                real_price_series=market_data.get('real_price')
            )
            st.plotly_chart(price_fig, use_container_width=True)
            
            # Add note about real data if available
            if results['simulation_params'].get('has_real_data', False):
                st.caption("ℹ️ Real market data is shown as a dotted line for comparison")
        
        with col2:
            # Volume chart
            volume_fig = visualizer.create_volume_chart(
                time_index=time_index,
                volume_series=market_data['volume_series'],
                liquidation_events=market_data['liquidation_events'],
                title=f"{asset_name} Trading Volume & Liquidations",
                real_volume_series=market_data.get('real_volume'),
                asset_type=asset_enum
            )
            st.plotly_chart(volume_fig, use_container_width=True)
            
            # Add note about real data if available
            if results['simulation_params'].get('has_real_data', False):
                st.caption("ℹ️ Real market volume is shown as light gray bars (6h average)")
        
        # Second row: APR and Revenue
        col3, col4 = st.columns(2)
        
        with col3:
            # APR chart
            apr_fig = visualizer.create_apr_chart(
                time_index=time_index,
                apr_series=fee_data['apr_series'],
                title="LP APR Over Time"
            )
            st.plotly_chart(apr_fig, use_container_width=True)
        
        with col4:
            # Revenue breakdown chart
            revenue_fig = visualizer.create_revenue_breakdown_chart(
                time_index=time_index,
                trading_fees=fee_data['trading_fees'],
                spread_revenue=fee_data['spread_revenue'],
                title="Revenue Breakdown"
            )
            st.plotly_chart(revenue_fig, use_container_width=True)
    
    with tab2:
        # Revenue analysis
        st.subheader("Revenue Analysis")
        
        # Cumulative revenue
        cum_rev_fig = visualizer.create_cumulative_revenue_chart(
            time_index=time_index,
            revenue_series=fee_data['cumulative_fees'],
            title="Cumulative LP Revenue"
        )
        st.plotly_chart(cum_rev_fig, use_container_width=True)
        
        # Revenue composition
        st.subheader("Revenue Composition")
        
        # Calculate total revenue by source
        total_trading_fees = np.sum(fee_data['trading_fees'])
        total_spread_revenue = np.sum(fee_data['spread_revenue'])
        total_revenue = total_trading_fees + total_spread_revenue
        
        # Create pie chart
        fig = go.Figure()
        
        fig.add_trace(go.Pie(
            labels=['Trading Fees', 'Spread Revenue'],
            values=[total_trading_fees, total_spread_revenue],
            hole=0.5,
            marker_colors=['#3BB273', '#E15554'],
            textinfo='percent+value',
            texttemplate='%{label}<br>$%{value:,.0f} (%{percent})',
            hoverinfo='label+percent+value',
            textposition='inside'
        ))
        
        fig.update_layout(
            title="Revenue Sources",
            showlegend=False,
            template='plotly_dark',
            margin=dict(t=50, b=20, l=20, r=20),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Volatility impact analysis
        st.subheader("Volatility Impact on Revenue")
        
        # Volatility vs Revenue scatter plot
        vol_fig = visualizer.create_volatility_vs_revenue_chart(
            price_series=market_data['price_series'],
            total_revenue=fee_data['total_revenue'],
            title="Volatility vs. Revenue"
        )
        
        if len(vol_fig.data) > 0:
            st.plotly_chart(vol_fig, use_container_width=True)
        else:
            st.warning("Not enough data to display volatility analysis. Try a longer simulation.")
        
        # Volatility statistics
        st.subheader("Volatility Statistics")
        
        # Calculate rolling volatility (24h window)
        returns = np.diff(np.log(market_data['price_series']))
        rolling_vol = pd.Series(returns).rolling(window=24).std() * np.sqrt(365 * 24)  # Annualized
        
        # Align the arrays by trimming the fee data to match the rolling volatility
        aligned_fees = fee_data['total_revenue'][-len(rolling_vol):]  # Take the last N elements to match rolling window
        
        # Calculate correlation between volatility and revenue
        valid_idx = ~np.isnan(rolling_vol) & ~np.isnan(aligned_fees)
        if valid_idx.sum() > 1:  # Need at least 2 points to calculate correlation
            corr = np.corrcoef(
                rolling_vol[valid_idx], 
                aligned_fees[valid_idx]
            )[0, 1]
        else:
            corr = np.nan
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Avg. Volatility", f"{rolling_vol[valid_idx].mean() * 100:.1f}%" if valid_idx.any() else "N/A")
        with col2:
            st.metric("📈 Max Volatility", f"{rolling_vol[valid_idx].max() * 100:.1f}%" if valid_idx.any() else "N/A")
        with col3:
            st.metric("🔗 Volatility-Revenue Correlation", f"{corr:.2f}" if not np.isnan(corr) else "N/A")
    
    with tab4:
        # Data export
        st.subheader("Export Simulation Data")
        
        # Create a DataFrame with all simulation data
        data = {
            'timestamp': time_index,
            'price': market_data['price_series'],
            'volume': market_data['volume_series'],
            'liquidation_events': market_data['liquidation_events'],
            'trading_fees': fee_data['trading_fees'],
            'spread_revenue': fee_data['spread_revenue'],
            'total_revenue': fee_data['total_revenue'],
            'cumulative_revenue': fee_data['cumulative_fees'],
            'apr': fee_data['apr_series'] * 100  # Convert to percentage
        }
        
        df = pd.DataFrame(data)
        
        # Display data preview
        st.write("### Data Preview")
        st.dataframe(df.head())
        
        # Download button
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="kavachfi_lp_simulation.csv">💾 Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # Display simulation parameters
        st.write("### Simulation Parameters")
        params = results['simulation_params']
        param_df = pd.DataFrame({
            'Parameter': [
                'Base Daily Volume ($)',
                'Trading Fee (%)',
                'LP Pool Size ($)',
                'Fee Rate',
                'Spread',
                'Insurance Fee Rate',
                'Simulation Duration (days)'
            ],
            'Value': [
                f"${params.get('base_volume', 0):,.0f}",
                f"{params.get('fee_rate', 0) * 100:.2f}%",
                f"${params.get('lp_pool_size', 0):,.0f}",
                f"{params.get('fee_rate', 0):.4f}",
                f"{params.get('spread', 0):.4f}",
                f"{params.get('insurance_fee_rate', 0) * 100:.2f}%",
                f"{params.get('simulation_days', 0)}"
            ]
        })
        st.table(param_df)

# Add footer with app information and disclaimer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>KavachFi LP Fee Simulator | Built with ❤️ for the DeFi community</p>
    <p>Disclaimer: This is a simulation tool. Actual results may vary.</p>
</div>
""", unsafe_allow_html=True)
