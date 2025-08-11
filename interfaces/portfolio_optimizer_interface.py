import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from core.data_fetcher import DataFetcher
from core.robinhood_parser import parse_investment_documents
from utils.equity_vis import render_equity_visualizations
from core.portfolio_optimizer import run_portfolio_optimization

def show_portfolio_optimization_tab():
    """Portfolio optimization tab functionality"""
    st.header("🎯 Portfolio Optimization")
    
    # Check if we have tracker data from the investment tab
    if 'tracker' in st.session_state and hasattr(st.session_state.tracker, 'portfolio') and not st.session_state.tracker.portfolio.empty:
        tracker = st.session_state.tracker
        
        # Show current portfolio summary from uploaded data
        st.success(f"✅ **Portfolio Data Found!** {len(tracker.portfolio)} positions worth ${tracker.portfolio['market_value'].sum():,.2f}")
        
        # Show portfolio preview
        show_portfolio_preview()
        
        # Add optimization information
        st.markdown("""
        <div style="background-color: #f0f8ff; padding: 1.5rem; border-left: 5px solid #15577A; border-radius: 8px; margin-bottom: 2rem;">
            <h4>🎯 About Portfolio Optimization</h4>
            <p style="margin-bottom: 1rem;">
                This feature uses <strong>Modern Portfolio Theory</strong> to optimize your portfolio allocation based on historical data. The optimization process:
            </p>
            <ul style="text-align: left; font-size: 0.95rem; margin-bottom: 1rem;">
                <li>📈 <strong>Fetches historical price data</strong> for all your holdings using Yahoo Finance</li>
                <li>📊 <strong>Calculates risk-return metrics</strong> including expected returns, volatility, and correlations</li>
                <li>🎯 <strong>Optimizes allocations</strong> using various strategies (Maximum Sharpe Ratio, Minimum Volatility, etc.)</li>
                <li>📋 <strong>Provides recommendations</strong> for rebalancing your portfolio</li>
            </ul>
            <p style="color: #666; font-size: 0.9rem; margin-bottom: 0;">
                <strong>Note:</strong> Optimization is based on historical data and assumes past performance patterns will continue. 
                Always consider your risk tolerance and investment goals.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Validate portfolio for optimization
        is_valid, message = validate_portfolio_for_optimization(tracker)
        
        if is_valid:
            st.info(f"🎯 {message}")
            
            # Run optimization interface
            run_optimization_interface(tracker)
        else:
            st.warning(f"⚠️ Portfolio validation failed: {message}")
            show_optimization_requirements()
    
    else:
        # No portfolio data available - show instructions
        st.info("""
        🔄 **Ready to optimize your portfolio?**
        
        Your portfolio data from **Tab 1 (Investment Portfolio)** will automatically be used here for optimization.
        """)
        
        st.warning("""
        ⚠️ **No portfolio data found**
        
        Please go to the **📈 Investment Portfolio** tab first and upload your portfolio data. 
        Once uploaded, return to this tab to run optimization.
        """)
        
        # Show what they need to do
        st.markdown("""
        ### 📋 Steps to get started:
        
        1. **Go to Tab 1**: Click on the "📈 Investment Portfolio" tab
        2. **Upload Your Data**: Upload your CSV/PDF files from your brokerage
        3. **Return Here**: Come back to this tab to see optimization results
        
        Your uploaded data will automatically be available in this tab!
        """)
        
        # Show requirements
        show_optimization_requirements()


def run_optimization_interface(tracker):
    """Optimization interface with integrated charts"""
    try:
        # Extract tickers from portfolio
        tickers = tracker.portfolio['symbol'].unique().tolist()
        valid_tickers = [ticker for ticker in tickers if ticker and str(ticker).upper() not in ['NAN', 'NONE', '']]
        
        if not valid_tickers:
            st.error("No valid tickers found in portfolio")
            return
        
        # Optimization parameters section
        st.subheader("⚙️ Optimization Settings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            optimization_type = st.selectbox(
                "Optimization Method",
                ["max_sharpe", "min_volatility", "max_return"],
                format_func=lambda x: {
                    "max_sharpe": "Maximum Sharpe Ratio",
                    "min_volatility": "Minimum Volatility", 
                    "max_return": "Maximum Return"
                }[x],
                help="Choose the optimization strategy"
            )
        
        with col2:
            data_period = st.selectbox(
                "Historical Data Period",
                ["1y", "2y", "3y", "5y"],
                index=1,  # Default to 2y
                help="Period of historical data to use for optimization"
            )
        
        with col3:
            risk_free_rate = st.number_input(
                "Risk-Free Rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=2.0,
                step=0.1,
                help="Current risk-free rate (e.g., Treasury bill rate)"
            ) / 100
        
        # Optimization comparison option
        st.subheader("🔬 Optimization Comparison")
        col4, col5 = st.columns(2)
        
        with col4:
            compare_methods = st.checkbox(
                "Compare All Methods", 
                value=False,
                help="Run optimization using all three methods and compare results"
            )
        
        with col5:
            if compare_methods:
                st.info("Will run: Max Sharpe, Min Volatility, and Max Return optimizations")
        
        # Advanced options
        with st.expander("🔧 Advanced Options", expanded=False):
            col6, col7 = st.columns(2)
            with col6:
                run_monte_carlo = st.checkbox("Run Monte Carlo Simulation", value=True)
                if run_monte_carlo:
                    num_simulations = st.number_input("Number of Simulations", 1000, 50000, 10000)
                else:
                    num_simulations = 0
                    
                # Asset filtering options
                max_assets = st.number_input("Max Assets in Portfolio", 5, 20, 15, 
                                           help="Maximum number of assets to include in optimization")
            
            with col7:
                generate_frontier = st.checkbox("Generate Efficient Frontier", value=True)
                if generate_frontier:
                    num_frontier_points = st.number_input("Frontier Points", 50, 200, 100)
                else:
                    num_frontier_points = 0
                    
                # Portfolio constraints info (for future implementation)
                st.info("📝 **Note**: Portfolio constraints (Max Single Weight, Min Weight) will be available in a future update. Current optimization uses default bounds from the optimizer.")
        
        # Run optimization button
        if st.button("🚀 Run Portfolio Optimization", type="primary", use_container_width=True):
            with st.spinner("🔄 Running portfolio optimization..."):
                try:
                    # Import here to avoid circular imports
                    from core.portfolio_optimizer import PortfolioOptimizer
                    from utils.portfolio_charts import PortfolioCharts
                    
                    # Initialize optimizer and charts
                    optimizer = PortfolioOptimizer(risk_free_rate=risk_free_rate)
                    charts = PortfolioCharts()
                    
                    # Store in session state for persistence
                    st.session_state.optimizer = optimizer
                    st.session_state.charts = charts
                    
                    # Fetch historical data
                    st.subheader("📊 Fetching Historical Data")
                    price_data = optimizer.fetch_historical_data(valid_tickers, period=data_period)
                    
                    if not price_data.empty:
                        # Generate historical_values format required by risk analysis
                        historical_values = DataFetcher().prepare_historical_values(price_data)
                        st.session_state.tracker.historical_values = historical_values  #  CRITICAL UPDATE
                        st.session_state.tracker = tracker  # Propagate changes


                    if price_data.empty:
                        st.error("❌ Failed to fetch historical data. Please check your internet connection and ticker symbols.")
                        return
                    
                    # Calculate returns
                    returns_data = optimizer.calculate_returns(price_data)
                    
                    if returns_data.empty:
                        st.error("❌ Failed to calculate returns from price data.")
                        return
                    
                    # Display data summary
                    display_data_summary(optimizer, returns_data)
                    
                    # Store results in session state
                    if compare_methods:
                        # Run all three optimization methods
                        all_results = run_multiple_optimizations(optimizer, max_assets)
                        
                        if not all_results:
                            st.error("❌ All optimization methods failed. Please check your data and settings.")
                            return
                            
                        st.session_state.all_optimization_results = all_results
                        
                        # Display comparison results
                        display_optimization_comparison(all_results, charts)
                        
                        # Use the best Sharpe ratio result as the primary optimal portfolio
                        best_sharpe_result = max(all_results.values(), key=lambda x: x['sharpe_ratio'])
                        optimal_portfolio = best_sharpe_result
                        st.session_state.optimization_results = optimal_portfolio
                        
                        st.info(f"🏆 **Best Result**: {best_sharpe_result['method']} (Sharpe: {best_sharpe_result['sharpe_ratio']:.3f})")
                        
                    else:
                        # Run single optimization
                        optimal_portfolio = optimizer.optimize_portfolio(optimization_type, max_assets)
                        
                        if not optimal_portfolio:
                            st.error("❌ Portfolio optimization failed. Please try different settings.")
                            return
                        
                        st.session_state.optimization_results = optimal_portfolio
                        
                        # Display single optimization results
                        display_optimization_results(optimal_portfolio, charts)
                    # Store returns data
                    if compare_methods and 'all_optimization_results' in st.session_state:
                        # Pass all results for comparison mode
                        simulation_df, frontier_df = run_advanced_analysis(
                            optimizer, charts, run_monte_carlo, generate_frontier, 
                            num_simulations, num_frontier_points, optimal_portfolio,
                            all_results=st.session_state.all_optimization_results  # Pass all results
                        )
                    else:
                        # Single optimization mode
                        simulation_df, frontier_df = run_advanced_analysis(
                            optimizer, charts, run_monte_carlo, generate_frontier, 
                            num_simulations, num_frontier_points, optimal_portfolio
                        )
                    if hasattr(st.session_state.tracker, 'historical_values'):
                        show_historical_performance_charts(st.session_state.tracker, charts)
                    
                    # NEW: Show asset performance chart
                    show_asset_performance_chart(st.session_state.tracker, charts)

                    # NEW: Show risk-return scatter
                    show_risk_return_scatter(optimizer, charts)

                    # NEW: Show correlation heatmap
                    show_correlation_heatmap(optimizer, charts)

                    # Compare with current portfolio
                    compare_portfolios(tracker, optimizer, optimal_portfolio, charts)
                    
                    # Display individual asset statistics
                    display_asset_statistics(optimizer)

                    # NEW: Show drawdown analysis
                    show_drawdown_analysis(optimizer, charts)
                    
                    # Export options
                    if compare_methods and 'all_optimization_results' in st.session_state:
                        provide_export_options_comparison(
                            st.session_state.all_optimization_results, data_period, risk_free_rate
                        )
                    else:
                        provide_export_options(optimal_portfolio, data_period, risk_free_rate)
                    
                    st.success("✅ Portfolio optimization completed successfully!")
                    
                except ImportError as e:
                    st.error(f"❌ Missing required modules: {str(e)}")
                    st.info("Please ensure all required files are available in your project structure.")
                except Exception as e:
                    st.error(f"❌ Error during optimization: {str(e)}")
                    with st.expander("🔍 Error Details"):
                        import traceback
                        st.code(traceback.format_exc())
    except Exception as e:
        print(f"Error: {e}")

def display_data_summary(optimizer, returns_data):
    """Display summary of fetched data"""
    st.subheader("📈 Data Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Assets", len(optimizer.tickers))
    with col2:
        st.metric("Data Points", len(returns_data))
    with col3:
        st.metric("Start Date", returns_data.index.min().strftime('%Y-%m-%d'))
    with col4:
        st.metric("End Date", returns_data.index.max().strftime('%Y-%m-%d'))
    
    # Show which tickers were successfully loaded
    st.info(f"📋 **Successfully loaded data for:** {', '.join(optimizer.tickers)}")


def run_multiple_optimizations(optimizer, max_assets, max_single_weight=0.5, min_weight=0.01):
    """Run optimization using all three methods and return results"""
    optimization_methods = {
        "max_sharpe": "Maximum Sharpe Ratio",
        "min_volatility": "Minimum Volatility", 
        "max_return": "Maximum Return"
    }
    
    results = {}
    progress_bar = st.progress(0)
    
    for i, (method_key, method_name) in enumerate(optimization_methods.items()):
        st.write(f"🔄 Running {method_name} optimization...")
        
        try:
            # Only pass parameters that the optimize_portfolio method accepts
            result = optimizer.optimize_portfolio(method_key, max_assets)
            if result:
                results[method_key] = result
                st.success(f"✅ {method_name} completed")
            else:
                st.warning(f"⚠️ {method_name} failed")
        except Exception as e:
            st.error(f"❌ {method_name} error: {str(e)}")
        
        progress_bar.progress((i + 1) / len(optimization_methods))
    
    progress_bar.empty()
    return results


def display_optimization_comparison(all_results, charts):
    """Display comparison of multiple optimization results"""
    if not all_results:
        st.error("No optimization results to compare")
        return
    
    st.subheader("📊 Optimization Method Comparison")
    
    # Create comparison table
    comparison_data = []
    for method_key, result in all_results.items():
        comparison_data.append({
            'Method': result['method'],
            'Expected Return': f"{result['expected_return']:.2%}",
            'Volatility': f"{result['volatility']:.2%}",
            'Sharpe Ratio': f"{result['sharpe_ratio']:.3f}",
            'Top 3 Holdings': get_top_holdings(result['weights'], 3)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Performance metrics comparison chart
    try:
        performance_chart = charts.create_performance_comparison(all_results)
        st.plotly_chart(performance_chart, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating performance comparison chart: {str(e)}")
    
    # Show detailed allocations in tabs
    st.subheader("📈 Detailed Allocations by Method")
    
    method_tabs = st.tabs([result['method'] for result in all_results.values()])
    
    for i, (method_key, result) in enumerate(all_results.items()):
        with method_tabs[i]:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Allocation table
                allocation_df = pd.DataFrame([
                    {'Ticker': ticker, 'Weight': f"{weight:.2%}"}
                    for ticker, weight in sorted(result['weights'].items(), 
                                               key=lambda x: x[1], reverse=True)
                ])
                st.dataframe(allocation_df, use_container_width=True, hide_index=True)
            
            with col2:
                # Pie chart for this method
                try:
                    pie_chart = charts.create_allocation_pie_chart(result['weights'])
                    st.plotly_chart(pie_chart, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating pie chart: {str(e)}")


def get_top_holdings(weights_dict, n=3):
    """Get top n holdings as a formatted string"""
    sorted_weights = sorted(weights_dict.items(), key=lambda x: x[1], reverse=True)
    top_holdings = sorted_weights[:n]
    return ", ".join([f"{ticker} ({weight:.1%})" for ticker, weight in top_holdings])


def display_optimization_results(optimal_portfolio, charts):
    """Display optimization results with charts"""
    st.subheader("⚡ Optimization Results")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Expected Annual Return", 
            f"{optimal_portfolio['expected_return']:.2%}",
            help="Expected annualized return based on historical data"
        )
    with col2:
        st.metric(
            "Annual Volatility", 
            f"{optimal_portfolio['volatility']:.2%}",
            help="Expected annualized volatility (risk)"
        )
    with col3:
        st.metric(
            "Sharpe Ratio", 
            f"{optimal_portfolio['sharpe_ratio']:.3f}",
            help="Risk-adjusted return measure"
        )
    
    # Portfolio allocation
    st.subheader("📊 Optimal Allocation")
    
    # Create allocation dataframe
    allocation_df = pd.DataFrame([
        {
            'Ticker': ticker,
            'Weight': f"{weight:.2%}",
            'Weight_Numeric': weight
        }
        for ticker, weight in optimal_portfolio['weights'].items()
    ]).sort_values('Weight_Numeric', ascending=False)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.dataframe(
            allocation_df[['Ticker', 'Weight']], 
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        # Create and display pie chart
        try:
            pie_chart = charts.create_allocation_pie_chart(optimal_portfolio['weights'])
            st.plotly_chart(pie_chart, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating pie chart: {str(e)}")
    
    # Also show bar chart
    try:
        bar_chart = charts.create_allocation_bar_chart(optimal_portfolio['weights'])
        st.plotly_chart(bar_chart, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating bar chart: {str(e)}")


def run_advanced_analysis(optimizer, charts, run_monte_carlo, generate_frontier, 
                         num_simulations, num_frontier_points, optimal_portfolio, all_results=None):
    """Run Monte Carlo simulation and efficient frontier generation"""
    simulation_df = None
    frontier_df = None
    
    if run_monte_carlo or generate_frontier:
        st.subheader("🔬 Advanced Analysis")
        
        # Run analyses
        if run_monte_carlo:
            with st.spinner("Running Monte Carlo simulation..."):
                try:
                    simulation_df = optimizer.monte_carlo_simulation(num_simulations)
                    if not simulation_df.empty:
                        st.success(f"✅ Generated {len(simulation_df)} random portfolios")
                    else:
                        st.warning("⚠️ Monte Carlo simulation returned empty results")
                except Exception as e:
                    st.error(f"❌ Monte Carlo simulation failed: {str(e)}")
        
        if generate_frontier:
            with st.spinner("Generating efficient frontier..."):
                try:
                    frontier_df = optimizer.generate_efficient_frontier(num_frontier_points)
                    if not frontier_df.empty:
                        st.success(f"✅ Generated efficient frontier with {len(frontier_df)} points")
                    else:
                        st.warning("⚠️ Efficient frontier generation returned empty results")
                except Exception as e:
                    st.error(f"❌ Efficient frontier generation failed: {str(e)}")
        
        # Plot efficient frontier and Monte Carlo results with ALL optimization results
        try:
            # If we have multiple optimization results, show them all on the chart
            if all_results:
                # Create enhanced frontier chart with all optimization methods
                frontier_fig = charts.plot_efficient_frontier_with_multiple_portfolios(
                    frontier_df if frontier_df is not None else pd.DataFrame(),
                    simulation_df,
                    all_results  # Pass all results instead of just one
                )
            else:
                # Single optimization result
                frontier_fig = charts.plot_efficient_frontier(
                    frontier_df if frontier_df is not None else pd.DataFrame(),
                    simulation_df,
                    optimal_portfolio
                )
            st.plotly_chart(frontier_fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error plotting efficient frontier: {str(e)}")
        
        # Show Monte Carlo distribution if available
        if simulation_df is not None and not simulation_df.empty:
            try:
                col1, col2 = st.columns(2)
                with col1:
                    sharpe_dist = charts.create_monte_carlo_distribution(simulation_df, 'Sharpe')
                    st.plotly_chart(sharpe_dist, use_container_width=True)
                with col2:
                    return_dist = charts.create_monte_carlo_distribution(simulation_df, 'Return')
                    st.plotly_chart(return_dist, use_container_width=True)
                    
                # Add comparison stats if we have multiple optimization results
                if all_results:
                    st.subheader("📊 Monte Carlo vs Optimization Methods")
                    create_monte_carlo_comparison(simulation_df, all_results)
                    
            except Exception as e:
                st.error(f"❌ Error creating Monte Carlo distributions: {str(e)}")
    
    return simulation_df, frontier_df


def create_monte_carlo_comparison(simulation_df, all_results):
    """Create comparison between Monte Carlo results and optimization methods"""
    try:
        # Calculate percentiles for Monte Carlo results
        mc_stats = {
            'Return': {
                'Mean': simulation_df['Return'].mean(),
                '25th': simulation_df['Return'].quantile(0.25),
                '75th': simulation_df['Return'].quantile(0.75),
                '90th': simulation_df['Return'].quantile(0.90),
                '95th': simulation_df['Return'].quantile(0.95)
            },
            'Volatility': {
                'Mean': simulation_df['Volatility'].mean(),
                '25th': simulation_df['Volatility'].quantile(0.25),
                '75th': simulation_df['Volatility'].quantile(0.75),
                '90th': simulation_df['Volatility'].quantile(0.90),
                '95th': simulation_df['Volatility'].quantile(0.95)
            },
            'Sharpe': {
                'Mean': simulation_df['Sharpe'].mean(),
                '25th': simulation_df['Sharpe'].quantile(0.25),
                '75th': simulation_df['Sharpe'].quantile(0.75),
                '90th': simulation_df['Sharpe'].quantile(0.90),
                '95th': simulation_df['Sharpe'].quantile(0.95)
            }
        }
        
        # Create comparison table
        comparison_data = []
        
        # Add Monte Carlo percentiles
        for percentile in ['Mean', '75th', '90th', '95th']:
            comparison_data.append({
                'Method': f'Monte Carlo ({percentile})',
                'Return': f"{mc_stats['Return'][percentile]:.2%}",
                'Volatility': f"{mc_stats['Volatility'][percentile]:.2%}",
                'Sharpe': f"{mc_stats['Sharpe'][percentile]:.3f}",
                'Type': 'Random'
            })
        
        # Add optimization results
        for method_key, result in all_results.items():
            comparison_data.append({
                'Method': result['method'],
                'Return': f"{result['expected_return']:.2%}",
                'Volatility': f"{result['volatility']:.2%}",
                'Sharpe': f"{result['sharpe_ratio']:.3f}",
                'Type': 'Optimized'
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Style the dataframe to highlight optimized vs random
        def highlight_type(row):
            if row['Type'] == 'Optimized':
                return ['background-color: #90EE90'] * len(row)  # Light green for optimized
            else:
                return ['background-color: #FFE4B5'] * len(row)  # Light orange for random
        
        # Display the comparison table with simple formatting
        # Separate Monte Carlo and Optimized results for cleaner display
        mc_df = comparison_df[comparison_df['Type'] == 'Random'].drop('Type', axis=1)
        opt_df = comparison_df[comparison_df['Type'] == 'Optimized'].drop('Type', axis=1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎲 Monte Carlo Results (Random Portfolios)**")
            st.dataframe(mc_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**🎯 Optimization Results**")
            st.dataframe(opt_df, use_container_width=True, hide_index=True)
        
        # Show insights
        best_mc_sharpe = mc_stats['Sharpe']['95th']
        best_opt_sharpe = max(all_results.values(), key=lambda x: x['sharpe_ratio'])['sharpe_ratio']
        
        if best_opt_sharpe > best_mc_sharpe:
            improvement = ((best_opt_sharpe - best_mc_sharpe) / best_mc_sharpe) * 100
            st.success(f"🎯 **Optimization Success**: Best optimized Sharpe ({best_opt_sharpe:.3f}) is {improvement:.1f}% better than 95th percentile of random portfolios ({best_mc_sharpe:.3f})")
        else:
            st.info(f"📊 Best optimized Sharpe: {best_opt_sharpe:.3f} vs 95th percentile random: {best_mc_sharpe:.3f}")
            
    except Exception as e:
        st.error(f"Error creating Monte Carlo comparison: {str(e)}")        
        # Style the dataframe to highlight optimized vs random
        def highlight_type(row):
            if row['Type'] == 'Optimized':
                return ['background-color: #90EE90'] * len(row)  # Light green for optimized
            else:
                return ['background-color: #FFE4B5'] * len(row)  # Light orange for random
        
        styled_df = comparison_df.drop('Type', axis=1).style.apply(highlight_type, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Show insights
        best_mc_sharpe = mc_stats['Sharpe']['95th']
        best_opt_sharpe = max(all_results.values(), key=lambda x: x['sharpe_ratio'])['sharpe_ratio']
        
        if best_opt_sharpe > best_mc_sharpe:
            improvement = ((best_opt_sharpe - best_mc_sharpe) / best_mc_sharpe) * 100
            st.success(f"🎯 **Optimization Success**: Best optimized Sharpe ({best_opt_sharpe:.3f}) is {improvement:.1f}% better than 95th percentile of random portfolios ({best_mc_sharpe:.3f})")
        else:
            st.info(f"📊 Best optimized Sharpe: {best_opt_sharpe:.3f} vs 95th percentile random: {best_mc_sharpe:.3f}")
            
    except Exception as e:
        st.error(f"Error creating Monte Carlo comparison: {str(e)}")

def compare_portfolios(tracker, optimizer, optimal_portfolio, charts):
    """Compare current portfolio with optimal portfolio"""
    st.subheader("⚖️ Current vs Optimal Portfolio")
    
    try:
        # Calculate current portfolio weights
        current_weights = {}
        total_value = tracker.portfolio['market_value'].sum()
        
        for _, row in tracker.portfolio.iterrows():
            ticker = row['symbol']
            if ticker in optimizer.tickers:
                current_weights[ticker] = row['market_value'] / total_value
        
        if current_weights:
            # Calculate current portfolio performance
            current_weights_array = np.array([current_weights.get(ticker, 0) for ticker in optimizer.tickers])
            current_performance = optimizer.portfolio_performance(current_weights_array)
            
            # Create comparison table
            comparison_df = pd.DataFrame({
                'Metric': ['Expected Return', 'Volatility', 'Sharpe Ratio'],
                'Current Portfolio': [f"{current_performance[0]:.2%}", 
                                    f"{current_performance[1]:.2%}", 
                                    f"{current_performance[2]:.3f}"],
                'Optimal Portfolio': [f"{optimal_portfolio['expected_return']:.2%}",
                                    f"{optimal_portfolio['volatility']:.2%}",
                                    f"{optimal_portfolio['sharpe_ratio']:.3f}"]
            })
            
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Show improvement metrics
            return_improvement = optimal_portfolio['expected_return'] - current_performance[0]
            risk_change = optimal_portfolio['volatility'] - current_performance[1]
            sharpe_improvement = optimal_portfolio['sharpe_ratio'] - current_performance[2]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Return Improvement", f"{return_improvement:+.2%}")
            with col2:
                st.metric("Risk Change", f"{risk_change:+.2%}", 
                         delta_color="inverse")  # Lower risk is better
            with col3:
                st.metric("Sharpe Improvement", f"{sharpe_improvement:+.3f}")
            
            # Create weights comparison chart
            try:
                weights_comparison_fig = charts.create_weights_comparison(current_weights, optimal_portfolio['weights'])
                st.plotly_chart(weights_comparison_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating weights comparison chart: {str(e)}")
                
        else:
            st.warning("⚠️ Cannot compare portfolios - no matching tickers found between current portfolio and optimization results.")
    
    except Exception as e:
        st.error(f"❌ Error comparing portfolios: {str(e)}")


def display_asset_statistics(optimizer):
    """Display individual asset statistics"""
    st.subheader("📈 Individual Asset Statistics")
    
    try:
        asset_stats = optimizer.get_asset_statistics()
        if not asset_stats.empty:
            # Format the statistics for better display
            display_stats = asset_stats.copy()
            
            # Format percentage columns
            for col in ['Annual_Return', 'Annual_Volatility', 'VaR_5']:
                display_stats[col] = display_stats[col].apply(lambda x: f"{x:.2%}")
            
            # Format ratio columns
            for col in ['Sharpe_Ratio', 'Sortino_Ratio', 'Skewness', 'Kurtosis']:
                display_stats[col] = display_stats[col].apply(lambda x: f"{x:.3f}")
            
            st.dataframe(display_stats, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ No asset statistics available")
    
    except Exception as e:
        st.error(f"❌ Error displaying asset statistics: {str(e)}")


def provide_export_options_comparison(all_results, data_period, risk_free_rate):
    """Provide export options for multiple optimization results"""
    st.subheader("💾 Export Comparison Results")
    
    try:
        # Create comprehensive export with all methods
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        export_content = f"""# Portfolio Optimization Comparison Results
Generated: {timestamp}
Data Period: {data_period}
Risk-Free Rate: {risk_free_rate:.4f}

# Method Comparison Summary
"""
        
        for method_key, result in all_results.items():
            export_content += f"""
## {result['method']}
Expected Return: {result['expected_return']:.4f}
Volatility: {result['volatility']:.4f}
Sharpe Ratio: {result['sharpe_ratio']:.4f}

Weights:
"""
            for ticker, weight in sorted(result['weights'].items(), key=lambda x: x[1], reverse=True):
                export_content += f"{ticker},{weight:.4f},{weight:.2%}\n"
        
        # Create downloadable comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📊 Download All Results",
                data=export_content,
                file_name=f"portfolio_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            # Create CSV with side-by-side comparison
            comparison_csv = create_comparison_csv(all_results)
            st.download_button(
                label="📈 Download CSV Comparison",
                data=comparison_csv,
                file_name=f"portfolio_weights_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Show preview
        with st.expander("👀 Preview Export Data"):
            preview_df = create_comparison_preview(all_results)
            st.dataframe(preview_df, use_container_width=True, hide_index=True)
    
    except Exception as e:
        st.error(f"❌ Error preparing export data: {str(e)}")


def create_comparison_csv(all_results):
    """Create CSV with side-by-side weight comparison"""
    try:
        # Get all unique tickers
        all_tickers = set()
        for result in all_results.values():
            all_tickers.update(result['weights'].keys())
        all_tickers = sorted(list(all_tickers))
        
        # Create comparison dataframe
        comparison_data = {'Ticker': all_tickers}
        
        for method_key, result in all_results.items():
            method_name = result['method'].replace(' ', '_')
            comparison_data[f'{method_name}_Weight'] = [
                result['weights'].get(ticker, 0) for ticker in all_tickers
            ]
            comparison_data[f'{method_name}_Percentage'] = [
                f"{result['weights'].get(ticker, 0):.2%}" for ticker in all_tickers
            ]
        
        df = pd.DataFrame(comparison_data)
        return df.to_csv(index=False)
    
    except Exception as e:
        return f"Error creating comparison CSV: {str(e)}"


def create_comparison_preview(all_results):
    """Create preview dataframe for comparison results"""
    try:
        # Get top 10 tickers by maximum weight across all methods
        ticker_max_weights = {}
        for result in all_results.values():
            for ticker, weight in result['weights'].items():
                ticker_max_weights[ticker] = max(ticker_max_weights.get(ticker, 0), weight)
        
        top_tickers = sorted(ticker_max_weights.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Create preview data
        preview_data = []
        for ticker, _ in top_tickers:
            row = {'Ticker': ticker}
            for method_key, result in all_results.items():
                method_short = method_key.replace('_', ' ').title()
                weight = result['weights'].get(ticker, 0)
                row[method_short] = f"{weight:.2%}" if weight > 0 else "-"
            preview_data.append(row)
        
        return pd.DataFrame(preview_data)
    
    except Exception as e:
        return pd.DataFrame({'Error': [str(e)]})


def provide_export_options(optimal_portfolio, data_period, risk_free_rate):
    """Provide export options for optimization results"""
    st.subheader("💾 Export Results")
    
    try:
        # Create export DataFrame
        export_df = pd.DataFrame([
            {
                'Ticker': ticker, 
                'Optimal_Weight': f"{weight:.4f}",
                'Optimal_Weight_Percentage': f"{weight:.2%}"
            }
            for ticker, weight in optimal_portfolio['weights'].items()
        ]).sort_values('Optimal_Weight', key=pd.to_numeric, ascending=False)
        
        # Add summary information
        summary_info = f"""
# Portfolio Optimization Results
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Method: {optimal_portfolio['method']}
Expected Return: {optimal_portfolio['expected_return']:.4f}
Volatility: {optimal_portfolio['volatility']:.4f}
Sharpe Ratio: {optimal_portfolio['sharpe_ratio']:.4f}
Data Period: {data_period}
Risk-Free Rate: {risk_free_rate:.4f}

# Optimal Weights
"""
        
        export_content = summary_info + export_df.to_csv(index=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📄 Download Detailed Results",
                data=export_content,
                file_name=f"portfolio_optimization_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Simple weights only
            simple_export = export_df[['Ticker', 'Optimal_Weight_Percentage']].to_csv(index=False)
            st.download_button(
                label="📊 Download Weights Only",
                data=simple_export,
                file_name=f"optimal_weights_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Show preview
        with st.expander("👀 Preview Export Data"):
            st.dataframe(export_df, use_container_width=True, hide_index=True)
    
    except Exception as e:
        st.error(f"❌ Error preparing export data: {str(e)}")


def show_optimization_requirements():
    """Display requirements and tips for optimization"""
    with st.expander("📋 Requirements for Portfolio Optimization"):
        st.markdown("""
        ### Requirements:
        
        1. **Valid Portfolio Data**: Upload CSV/PDF files with your current holdings
        2. **Ticker Symbols**: Ensure your portfolio contains recognizable ticker symbols  
        3. **Internet Connection**: Required to fetch historical price data from Yahoo Finance
        4. **Diversified Portfolio**: Works best with 3+ different assets
        5. **Historical Data**: Assets need sufficient trading history (typically 1+ years)
        
        ### Optimization Methods Available:
        
        - **Maximum Sharpe Ratio**: Optimizes risk-adjusted returns (recommended for most investors)
        - **Minimum Volatility**: Minimizes portfolio risk
        - **Maximum Return**: Maximizes expected returns (higher risk)
        """)
    
    with st.expander("💡 Tips for Better Optimization Results"):
        st.markdown("""
        ### Getting the Best Results:
        
        **Data Quality:**
        - Use CSV files when possible (more accurate than PDF parsing)
        - Ensure all your holdings have valid ticker symbols
        - Upload your most recent portfolio statement
        
        **Portfolio Considerations:**
        - Include at least 3-5 different assets for meaningful optimization
        - Mix different asset classes (stocks, ETFs, bonds) for better diversification
        - Consider your investment timeline when choosing optimization period
        
        **Optimization Settings:**
        - **2-3 years** of historical data usually provides good balance
        - **Maximum Sharpe Ratio** is recommended for most investors
        - Adjust risk-free rate to match current treasury yields
        
        **Implementation:**
        - Consider transaction costs when implementing recommendations
        - Rebalance gradually to minimize market impact
        - Review and update optimization quarterly or semi-annually
        """)


def show_optimization_status():
    """Show current optimization status and portfolio info"""
    if 'tracker' in st.session_state and hasattr(st.session_state.tracker, 'portfolio'):
        tracker = st.session_state.tracker
        portfolio = tracker.portfolio
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Positions", 
                value=len(portfolio),
                help="Number of unique positions in portfolio"
            )
        
        with col2:
            total_value = portfolio['market_value'].sum() if 'market_value' in portfolio.columns else 0
            st.metric(
                label="Portfolio Value", 
                value=f"${total_value:,.2f}",
                help="Total market value of all positions"
            )
        
        with col3:
            unique_assets = portfolio['symbol'].nunique() if 'symbol' in portfolio.columns else 0
            st.metric(
                label="Unique Assets", 
                value=unique_assets,
                help="Number of different ticker symbols"
            )
        
        with col4:
            optimization_ready = "✅ Ready" if unique_assets >= 2 else "⚠️ Need 2+ assets"
            st.metric(
                label="Optimization Status", 
                value=optimization_ready,
                help="Portfolio optimization readiness"
            )


def validate_portfolio_for_optimization(tracker):
    """Validate if portfolio is suitable for optimization"""
    if not tracker or tracker.portfolio.empty:
        return False, "No portfolio data available"
    
    portfolio = tracker.portfolio
    unique_symbols = portfolio['symbol'].nunique()
    
    if unique_symbols < 2:
        return False, f"Need at least 2 different assets, found {unique_symbols}"
    
    # Check for valid ticker symbols
    invalid_symbols = []
    for symbol in portfolio['symbol'].unique():
        if not symbol or len(str(symbol).strip()) == 0:
            invalid_symbols.append("Empty symbol")
        elif len(str(symbol)) > 10:  # Most ticker symbols are <= 5 characters
            invalid_symbols.append(str(symbol))
    
    if invalid_symbols:
        return False, f"Invalid ticker symbols found: {', '.join(invalid_symbols[:3])}"
    
    return True, "Portfolio ready for optimization"


def show_portfolio_preview():
    """Show a preview of the current portfolio for optimization"""
    if 'tracker' in st.session_state and hasattr(st.session_state.tracker, 'portfolio'):
        tracker = st.session_state.tracker
        
        # Show key metrics
        show_optimization_status()
        
        # Show portfolio holdings
        if not tracker.portfolio.empty:
            st.subheader("🏠 Current Holdings")
            
            # Select relevant columns for display
            display_columns = ['symbol', 'quantity', 'market_value', 'gain_loss_pct']
            available_columns = [col for col in display_columns if col in tracker.portfolio.columns]
            
            if available_columns:
                preview_df = tracker.portfolio[available_columns].copy()
                
                # Format the dataframe for better display
                if 'market_value' in preview_df.columns:
                    preview_df['market_value'] = preview_df['market_value'].apply(
                        lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A"
                    )
                
                if 'gain_loss_pct' in preview_df.columns:
                    preview_df['gain_loss_pct'] = preview_df['gain_loss_pct'].apply(
                        lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A"
                    )
                
                # Rename columns for better display
                column_names = {
                    'symbol': 'Symbol',
                    'quantity': 'Quantity',
                    'market_value': 'Market Value',
                    'gain_loss_pct': 'Gain/Loss %'
                }
                preview_df = preview_df.rename(columns=column_names)
                
                st.dataframe(
                    preview_df, 
                    use_container_width=True,
                    hide_index=True
                )
            else:
                # Fallback to showing all columns
                st.dataframe(tracker.portfolio.head(10), use_container_width=True)

# Add after show_portfolio_preview() function

def show_historical_performance_charts(tracker, charts):
    """Show historical performance charts"""
    st.subheader("📈 Historical Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            worth_fig = charts.create_portfolio_worth_chart(tracker)
            st.plotly_chart(worth_fig, use_container_width=True, key="portfolio_worth")
        except Exception as e:
            st.error(f"Error creating portfolio worth chart: {str(e)}")
    
    with col2:
        try:
            composition_fig = charts.create_portfolio_composition_over_time(tracker)
            st.plotly_chart(composition_fig, use_container_width=True, key="portfolio_composition")
        except Exception as e:
            st.error(f"Error creating composition chart: {str(e)}")
    
    try:
        metrics_fig = charts.create_portfolio_metrics_dashboard(tracker)
        st.plotly_chart(metrics_fig, use_container_width=True, key="portfolio_metrics")
    except Exception as e:
        st.error(f"Error creating metrics dashboard: {str(e)}")

def show_asset_performance_chart(tracker, charts):
    """Show asset performance chart"""
    st.subheader("📈 Individual Asset Performance")
    try:
        perf_fig = charts.create_asset_performance_chart(tracker)
        st.plotly_chart(perf_fig, use_container_width=True, key="asset_performance")
    except Exception as e:
        st.error(f"Error creating asset performance chart: {str(e)}")

def show_risk_return_scatter(optimizer, charts):
    """Show risk-return scatter plot"""
    st.subheader("📊 Risk-Return Analysis")
    try:
        weights = st.session_state.optimization_results['weights'] if 'optimization_results' in st.session_state else None
        scatter_fig = charts.create_risk_return_scatter(optimizer.returns_data, weights)
        st.plotly_chart(scatter_fig, use_container_width=True, key="risk_return_scatter")
    except Exception as e:
        st.error(f"Error creating risk-return scatter: {str(e)}")

def show_correlation_heatmap(optimizer, charts):
    """Show correlation heatmap"""
    st.subheader("🔗 Asset Correlation Matrix")
    try:
        heatmap_fig = charts.create_correlation_heatmap(optimizer.returns_data)
        st.plotly_chart(heatmap_fig, use_container_width=True, key="correlation_heatmap")
    except Exception as e:
        st.error(f"Error creating correlation heatmap: {str(e)}")

def show_drawdown_analysis(optimizer, charts):
    """Show drawdown analysis"""
    st.subheader("📉 Drawdown Analysis")
    try:
        if 'optimization_results' in st.session_state:
            weights_array = np.array([
                st.session_state.optimization_results['weights'].get(ticker, 0) 
                for ticker in optimizer.tickers
            ])
            drawdown_fig = charts.create_drawdown_chart(optimizer.returns_data, weights_array)
            st.plotly_chart(drawdown_fig, use_container_width=True, key="drawdown_chart")
    except Exception as e:
        st.error(f"Error creating drawdown chart: {str(e)}")


# Example usage
if __name__ == "__main__":
    st.set_page_config(
        page_title="Portfolio Optimization",
        page_icon="🎯",
        layout="wide"
    )
    
    show_portfolio_optimization_tab()