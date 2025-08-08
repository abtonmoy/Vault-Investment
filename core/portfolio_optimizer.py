import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from scipy.optimize import minimize
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional
import warnings
from requests.exceptions import RequestException
from socket import timeout
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.returns_data = None
        self.mean_returns = None
        self.cov_matrix = None
        self.tickers = []
        
    def fetch_historical_data(self, tickers: List[str], period: str = "2y") -> pd.DataFrame:
        """
        Fetch historical price data for given tickers
        
        Args:
            tickers: List of ticker symbols
            period: Time period for historical data (1y, 2y, 5y, max)
            
        Returns:
            DataFrame with adjusted closing prices
        """
        CRYPTO_MAP = {
        'BTC': 'BTC-USD',
        'ETH': 'ETH-USD',
        'DOGE': 'DOGE-USD',
        # Add other crypto mappings as needed
        }
        try:
            # Clean and validate tickers
            clean_tickers = []
            for ticker in tickers:
                # Remove any whitespace and convert to uppercase
                clean_ticker = str(ticker).strip().upper()
                if '.' in clean_ticker:
                    clean_ticker = clean_ticker.replace('.', '-')
                elif clean_ticker in CRYPTO_MAP:
                    clean_ticker = CRYPTO_MAP[clean_ticker]
                if clean_ticker and clean_ticker not in ['NAN', 'NONE', '']:
                    clean_tickers.append(clean_ticker)
            
            if not clean_tickers:
                st.error("No valid tickers provided")
                return pd.DataFrame()
                
            self.tickers = clean_tickers
            st.info(f"Fetching data for {len(clean_tickers)} tickers: {', '.join(clean_tickers)}")
            
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
                st.error("No tickers have sufficient historical data for optimization")
                return pd.DataFrame()
                
            price_data = price_data[valid_tickers]
            self.tickers = valid_tickers
            
            # Forward fill missing values
            price_data = price_data.fillna(method='ffill').dropna()
            
            st.success(f"Successfully fetched data for {len(valid_tickers)} tickers with {len(price_data)} days of data")
            
            return price_data
            
        except (RequestException, timeout) as e:
            st.warning(f"Network error fetching {ticker}: {str(e)}")
            failed_tickers.append(ticker)
        except Exception as e:
            st.warning(f"Failed to fetch data for {ticker}: {str(e)}")
            failed_tickers.append(ticker)
    
    def calculate_returns(self, price_data: pd.DataFrame, return_type: str = "daily") -> pd.DataFrame:
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
            
            self.returns_data = returns
            self.mean_returns = returns.mean()
            self.cov_matrix = returns.cov()
            
            return returns
            
        except Exception as e:
            st.error(f"Error calculating returns: {str(e)}")
            return pd.DataFrame()
    
    def portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio performance metrics
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Tuple of (expected_return, volatility, sharpe_ratio)
        """
        try:
            # Annualize returns (assuming daily returns)
            portfolio_return = np.sum(self.mean_returns * weights) * 252
            portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix * 252, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Calculate Sharpe ratio
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            return portfolio_return, portfolio_volatility, sharpe_ratio
            
        except Exception as e:
            st.error(f"Error calculating portfolio performance: {str(e)}")
            return 0, 0, 0
    
    def optimize_portfolio(self, optimization_type: str = "max_sharpe") -> Dict:
        """
        Optimize portfolio based on selected criteria
        
        Args:
            optimization_type: Type of optimization ('max_sharpe', 'min_volatility', 'max_return')
            
        Returns:
            Dictionary with optimization results
        """
        try:
            if self.returns_data is None or self.returns_data.empty:
                st.error("No returns data available for optimization")
                return {}
            
            num_assets = len(self.tickers)
            
            # Constraints and bounds
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
            bounds = tuple((0.01, 0.4) for _ in range(num_assets))  # Min 1%, max 40% per asset
            
            # Initial guess (equal weights)
            initial_guess = np.array([1.0 / num_assets] * num_assets)
            
            # Objective functions
            def neg_sharpe_ratio(weights):
                return -self.portfolio_performance(weights)[2]
            
            def portfolio_volatility(weights):
                return self.portfolio_performance(weights)[1]
            
            def neg_portfolio_return(weights):
                return -self.portfolio_performance(weights)[0]
            
            # Choose objective function based on optimization type
            if optimization_type == "max_sharpe":
                objective = neg_sharpe_ratio
                method_name = "Maximum Sharpe Ratio"
            elif optimization_type == "min_volatility":
                objective = portfolio_volatility
                method_name = "Minimum Volatility"
            elif optimization_type == "max_return":
                objective = neg_portfolio_return
                method_name = "Maximum Return"
                # For max return, allow higher concentration
                bounds = tuple((0.01, 0.6) for _ in range(num_assets))
            else:
                objective = neg_sharpe_ratio
                method_name = "Maximum Sharpe Ratio"
            
            # Perform optimization
            st.info(f"Optimizing portfolio using {method_name} method...")
            
            result = minimize(
                objective,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'ftol': 1e-9, 'disp': False}
            )
            
            if not result.success:
                st.error(f"Optimization failed: {result.message}")
                return {}
            
            optimal_weights = result.x
            expected_return, volatility, sharpe_ratio = self.portfolio_performance(optimal_weights)
            
            # Create results dictionary
            optimization_results = {
                'method': method_name,
                'weights': dict(zip(self.tickers, optimal_weights)),
                'expected_return': expected_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'success': result.success,
                'raw_weights': optimal_weights
            }
            
            return optimization_results
            
        except Exception as e:
            st.error(f"Error during optimization: {str(e)}")
            return {}
    
    def generate_efficient_frontier(self, num_portfolios: int = 100) -> pd.DataFrame:
        """
        Generate efficient frontier data
        
        Args:
            num_portfolios: Number of portfolios to generate
            
        Returns:
            DataFrame with efficient frontier data
        """
        try:
            if self.returns_data is None or self.returns_data.empty:
                return pd.DataFrame()
            
            num_assets = len(self.tickers)
            results = np.zeros((3, num_portfolios))
            
            # Generate target returns
            min_ret = self.mean_returns.min() * 252
            max_ret = self.mean_returns.max() * 252
            target_returns = np.linspace(min_ret, max_ret, num_portfolios)
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            ]
            bounds = tuple((0.01, 0.5) for _ in range(num_assets))
            
            for i, target in enumerate(target_returns):
                # Add return constraint
                return_constraint = {'type': 'eq', 'fun': lambda x, target=target: 
                                   self.portfolio_performance(x)[0] - target}
                cons = constraints + [return_constraint]
                
                # Minimize volatility for given return
                result = minimize(
                    lambda x: self.portfolio_performance(x)[1],
                    np.array([1.0 / num_assets] * num_assets),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=cons,
                    options={'ftol': 1e-9, 'disp': False}
                )
                
                if result.success:
                    ret, vol, sharpe = self.portfolio_performance(result.x)
                    results[0, i] = ret
                    results[1, i] = vol
                    results[2, i] = sharpe
                else:
                    results[:, i] = np.nan
            
            # Create DataFrame
            frontier_df = pd.DataFrame({
                'Return': results[0],
                'Volatility': results[1],
                'Sharpe': results[2]
            }).dropna()
            
            return frontier_df
            
        except Exception as e:
            st.error(f"Error generating efficient frontier: {str(e)}")
            return pd.DataFrame()
    
    def monte_carlo_simulation(self, num_simulations: int = 10000) -> pd.DataFrame:
        """
        Perform Monte Carlo simulation for random portfolio weights
        
        Args:
            num_simulations: Number of random portfolios to generate
            
        Returns:
            DataFrame with simulation results
        """
        try:
            if self.returns_data is None or self.returns_data.empty:
                return pd.DataFrame()
            
            num_assets = len(self.tickers)
            results = np.zeros((3, num_simulations))
            weights_array = np.zeros((num_simulations, num_assets))
            
            np.random.seed(42)  # For reproducibility
            
            for i in range(num_simulations):
                # Generate random weights
                weights = np.random.random(num_assets)
                weights /= np.sum(weights)  # Normalize to sum to 1
                
                # Store weights
                weights_array[i, :] = weights
                
                # Calculate portfolio metrics
                ret, vol, sharpe = self.portfolio_performance(weights)
                results[0, i] = ret
                results[1, i] = vol
                results[2, i] = sharpe
            
            # Create DataFrame
            simulation_df = pd.DataFrame({
                'Return': results[0],
                'Volatility': results[1],
                'Sharpe': results[2]
            })
            
            # Add weight columns
            for j, ticker in enumerate(self.tickers):
                simulation_df[f'Weight_{ticker}'] = weights_array[:, j]
            
            return simulation_df
            
        except Exception as e:
            st.error(f"Error in Monte Carlo simulation: {str(e)}")
            return pd.DataFrame()
    
    def plot_efficient_frontier(self, frontier_df: pd.DataFrame, 
                              simulation_df: pd.DataFrame = None,
                              optimal_portfolio: Dict = None) -> go.Figure:
        """
        Plot efficient frontier with optimal portfolio
        
        Args:
            frontier_df: Efficient frontier data
            simulation_df: Monte Carlo simulation data
            optimal_portfolio: Optimal portfolio results
            
        Returns:
            Plotly figure
        """
        try:
            fig = go.Figure()
            
            # Plot Monte Carlo simulation if available
            if simulation_df is not None and not simulation_df.empty:
                fig.add_trace(go.Scatter(
                    x=simulation_df['Volatility'],
                    y=simulation_df['Return'],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=simulation_df['Sharpe'],
                        colorscale='Viridis',
                        colorbar=dict(title="Sharpe Ratio"),
                        opacity=0.6
                    ),
                    name='Random Portfolios',
                    hovertemplate="<b>Random Portfolio</b><br>" +
                                "Return: %{y:.2%}<br>" +
                                "Volatility: %{x:.2%}<br>" +
                                "Sharpe: %{marker.color:.3f}<extra></extra>"
                ))
            
            # Plot efficient frontier
            if not frontier_df.empty:
                fig.add_trace(go.Scatter(
                    x=frontier_df['Volatility'],
                    y=frontier_df['Return'],
                    mode='lines',
                    line=dict(color='red', width=3),
                    name='Efficient Frontier',
                    hovertemplate="<b>Efficient Frontier</b><br>" +
                                "Return: %{y:.2%}<br>" +
                                "Volatility: %{x:.2%}<extra></extra>"
                ))
            
            # Plot optimal portfolio
            if optimal_portfolio and 'volatility' in optimal_portfolio:
                fig.add_trace(go.Scatter(
                    x=[optimal_portfolio['volatility']],
                    y=[optimal_portfolio['expected_return']],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='gold',
                        symbol='star',
                        line=dict(color='black', width=2)
                    ),
                    name=f"Optimal Portfolio ({optimal_portfolio.get('method', '')})",
                    hovertemplate="<b>Optimal Portfolio</b><br>" +
                                f"Method: {optimal_portfolio.get('method', 'N/A')}<br>" +
                                "Return: %{y:.2%}<br>" +
                                "Volatility: %{x:.2%}<br>" +
                                f"Sharpe: {optimal_portfolio.get('sharpe_ratio', 0):.3f}<extra></extra>"
                ))
            
            # Update layout
            fig.update_layout(
                title='Efficient Frontier and Portfolio Optimization',
                xaxis_title='Volatility (Risk)',
                yaxis_title='Expected Return',
                hovermode='closest',
                template='plotly_dark',
                width=800,
                height=600
            )
            
            # Format axes as percentages
            fig.update_xaxis(tickformat='.1%')
            fig.update_yaxis(tickformat='.1%')
            
            return fig
            
        except Exception as e:
            st.error(f"Error plotting efficient frontier: {str(e)}")
            return go.Figure()
    
    def create_allocation_chart(self, weights_dict: Dict[str, float]) -> go.Figure:
        """
        Create pie chart for portfolio allocation
        
        Args:
            weights_dict: Dictionary of ticker weights
            
        Returns:
            Plotly figure
        """
        try:
            # Filter out very small weights for cleaner visualization
            filtered_weights = {k: v for k, v in weights_dict.items() if v > 0.005}  # > 0.5%
            other_weight = sum(v for k, v in weights_dict.items() if v <= 0.005)
            
            if other_weight > 0:
                filtered_weights['Others'] = other_weight
            
            fig = go.Figure(data=[go.Pie(
                labels=list(filtered_weights.keys()),
                values=list(filtered_weights.values()),
                hole=0.4,
                textinfo='label+percent',
                textposition='outside',
                hovertemplate="<b>%{label}</b><br>" +
                            "Weight: %{percent}<br>" +
                            "Value: %{value:.3f}<extra></extra>"
            )])
            
            fig.update_layout(
                title='Optimal Portfolio Allocation',
                template='plotly_dark',
                width=600,
                height=600
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating allocation chart: {str(e)}")
            return go.Figure()


def run_portfolio_optimization(tracker, optimization_params: Dict = None):
    """
    Main function to run portfolio optimization from investment tracker data
    
    Args:
        tracker: Investment tracker instance
        optimization_params: Dictionary with optimization parameters
    """
    try:
        if tracker.portfolio.empty:
            st.warning("No portfolio data available for optimization")
            return
        
        # Extract tickers from portfolio
        tickers = tracker.portfolio['symbol'].unique().tolist()
        
        # Remove any invalid tickers
        valid_tickers = [ticker for ticker in tickers if ticker and str(ticker).upper() not in ['NAN', 'NONE', '']]
        
        if not valid_tickers:
            st.error("No valid tickers found in portfolio")
            return
        
        st.header("🎯 Portfolio Optimization")
        
        # Optimization parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            optimization_type = st.selectbox(
                "Optimization Method",
                ["max_sharpe", "min_volatility", "max_return"],
                format_func=lambda x: {
                    "max_sharpe": "Maximum Sharpe Ratio",
                    "min_volatility": "Minimum Volatility", 
                    "max_return": "Maximum Return"
                }[x]
            )
        
        with col2:
            data_period = st.selectbox(
                "Historical Data Period",
                ["1y", "2y", "3y", "5y"],
                index=1  # Default to 2y
            )
        
        with col3:
            risk_free_rate = st.number_input(
                "Risk-Free Rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=2.0,
                step=0.1
            ) / 100
        
        # Advanced options
        with st.expander("Advanced Options"):
            col4, col5 = st.columns(2)
            with col4:
                run_monte_carlo = st.checkbox("Run Monte Carlo Simulation", value=True)
                num_simulations = st.number_input("Number of Simulations", 1000, 50000, 10000)
            
            with col5:
                generate_frontier = st.checkbox("Generate Efficient Frontier", value=True)
                num_frontier_points = st.number_input("Frontier Points", 50, 200, 100)
        
        if st.button("🚀 Run Optimization", type="primary"):
            # Initialize optimizer
            optimizer = PortfolioOptimizer(risk_free_rate=risk_free_rate)
            
            # Fetch historical data
            st.subheader("📊 Fetching Historical Data")
            price_data = optimizer.fetch_historical_data(valid_tickers, period=data_period)
            
            if price_data.empty:
                st.error("Failed to fetch historical data")
                return
            
            # Calculate returns
            returns_data = optimizer.calculate_returns(price_data)
            
            if returns_data.empty:
                st.error("Failed to calculate returns")
                return
            
            # Display data summary
            st.subheader("📈 Data Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Assets", len(optimizer.tickers))
            with col2:
                st.metric("Data Points", len(returns_data))
            with col3:
                st.metric("Date Range", f"{returns_data.index.min().date()} to {returns_data.index.max().date()}")
            
            # Run optimization
            st.subheader("⚡ Optimization Results")
            optimal_portfolio = optimizer.optimize_portfolio(optimization_type)
            
            if not optimal_portfolio:
                st.error("Optimization failed")
                return
            
            # Display results
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
                st.dataframe(allocation_df[['Ticker', 'Weight']], use_container_width=True)
            
            with col2:
                allocation_chart = optimizer.create_allocation_chart(optimal_portfolio['weights'])
                st.plotly_chart(allocation_chart, use_container_width=True)
            
            # Advanced analysis
            if run_monte_carlo or generate_frontier:
                st.subheader("🔬 Advanced Analysis")
                
                simulation_df = None
                frontier_df = None
                
                if run_monte_carlo:
                    with st.spinner("Running Monte Carlo simulation..."):
                        simulation_df = optimizer.monte_carlo_simulation(num_simulations)
                
                if generate_frontier:
                    with st.spinner("Generating efficient frontier..."):
                        frontier_df = optimizer.generate_efficient_frontier(num_frontier_points)
                
                # Plot efficient frontier
                frontier_fig = optimizer.plot_efficient_frontier(
                    frontier_df if frontier_df is not None else pd.DataFrame(),
                    simulation_df,
                    optimal_portfolio
                )
                st.plotly_chart(frontier_fig, use_container_width=True)
            
            # Comparison with current portfolio
            st.subheader("⚖️ Current vs Optimal Portfolio")
            
            # Calculate current portfolio weights
            current_weights = {}
            total_value = tracker.portfolio['market_value'].sum()
            for _, row in tracker.portfolio.iterrows():
                ticker = row['symbol']
                if ticker in optimizer.tickers:
                    current_weights[ticker] = row['market_value'] / total_value
            
            if current_weights:
                current_performance = optimizer.portfolio_performance(
                    np.array([current_weights.get(ticker, 0) for ticker in optimizer.tickers])
                )
                
                comparison_df = pd.DataFrame({
                    'Metric': ['Expected Return', 'Volatility', 'Sharpe Ratio'],
                    'Current Portfolio': [f"{current_performance[0]:.2%}", 
                                        f"{current_performance[1]:.2%}", 
                                        f"{current_performance[2]:.3f}"],
                    'Optimal Portfolio': [f"{optimal_portfolio['expected_return']:.2%}",
                                        f"{optimal_portfolio['volatility']:.2%}",
                                        f"{optimal_portfolio['sharpe_ratio']:.3f}"]
                })
                
                st.dataframe(comparison_df, use_container_width=True)
                
                # Show improvement
                return_improvement = optimal_portfolio['expected_return'] - current_performance[0]
                risk_change = optimal_portfolio['volatility'] - current_performance[1]
                sharpe_improvement = optimal_portfolio['sharpe_ratio'] - current_performance[2]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Return Improvement", f"{return_improvement:+.2%}")
                with col2:
                    st.metric("Risk Change", f"{risk_change:+.2%}")
                with col3:
                    st.metric("Sharpe Improvement", f"{sharpe_improvement:+.3f}")
            
            # Export results
            st.subheader("💾 Export Results")
            
            # Prepare export data
            export_data = {
                'optimization_method': optimal_portfolio['method'],
                'expected_return': optimal_portfolio['expected_return'],
                'volatility': optimal_portfolio['volatility'],
                'sharpe_ratio': optimal_portfolio['sharpe_ratio'],
                'weights': optimal_portfolio['weights'],
                'data_period': data_period,
                'risk_free_rate': risk_free_rate,
                'optimization_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Convert to DataFrame for export
            export_df = pd.DataFrame([
                {'Ticker': ticker, 'Optimal_Weight': weight}
                for ticker, weight in optimal_portfolio['weights'].items()
            ])
            
            st.download_button(
                label="📄 Download Optimal Weights",
                data=export_df.to_csv(index=False),
                file_name=f"optimal_portfolio_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
            
    except Exception as e:
        st.error(f"Error in portfolio optimization: {str(e)}")
        import traceback
        st.error(traceback.format_exc())