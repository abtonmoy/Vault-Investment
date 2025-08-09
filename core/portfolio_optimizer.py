import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from scipy.optimize import minimize
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
from requests.exceptions import RequestException
from socket import timeout
from core.data_fetcher import DataFetcher
from scipy.optimize import differential_evolution
from scipy.stats import mstats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from joblib import Parallel, delayed

# Import the separate charts module
warnings.filterwarnings('ignore')
from utils.portfolio_charts import PortfolioCharts

class PortfolioOptimizer:
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.fetcher = DataFetcher(risk_free_rate)
        self.returns_data = None
        self.mean_returns = None
        self.cov_matrix = None
        self.tickers = []
        self.data_frequency = "daily"  # Track data frequency for annualization
        self.charts = PortfolioCharts()  # Initialize charts handler
        
    def fetch_historical_data(self, tickers: List[str], period: str = "2y") -> pd.DataFrame:
        """
        Fetch historical data using the shared fetcher
        """
        price_data, returns_data, _ = self.fetcher.fetch_historical_data(tickers, period)
        self.returns_data = returns_data
        self.mean_returns = returns_data.mean() if returns_data is not None else None
        self.cov_matrix = returns_data.cov() if returns_data is not None else None
        self.tickers = returns_data.columns.tolist() if returns_data is not None else []
        return price_data
    
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
            self.data_frequency = return_type  # Store data frequency
            
            if return_type == "daily":
                returns = price_data.pct_change().dropna()
            elif return_type == "weekly":
                returns = price_data.resample('W').last().pct_change().dropna()
            elif return_type == "monthly":
                returns = price_data.resample('M').last().pct_change().dropna()
            else:
                returns = price_data.pct_change().dropna()
            
            # Winsorize returns instead of clipping (1% on each tail)
            returns = returns.apply(lambda x: mstats.winsorize(x, limits=[0.01, 0.01]))
            
            self.returns_data = returns
            self.mean_returns = returns.mean()
            self.cov_matrix = returns.cov()
            
            return returns
            
        except Exception as e:
            st.error(f"Error calculating returns: {str(e)}")
            return pd.DataFrame()
    
    def get_annualization_factor(self) -> int:
        """Get annualization factor based on data frequency"""
        if self.data_frequency == "daily":
            return 252
        elif self.data_frequency == "weekly":
            return 52
        elif self.data_frequency == "monthly":
            return 12
        return 252  # Default to daily
    
    def portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio performance metrics
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Tuple of (expected_return, volatility, sharpe_ratio)
        """
        try:
            epsilon = 1e-8  # Small value to avoid division by zero
            annual_factor = self.get_annualization_factor()
            
            # Annualize returns
            portfolio_return = np.sum(self.mean_returns * weights) * annual_factor
            portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix * annual_factor, weights))
            portfolio_volatility = np.sqrt(portfolio_variance) + epsilon
            
            # Calculate Sharpe ratio
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            return portfolio_return, portfolio_volatility, sharpe_ratio
            
        except Exception as e:
            st.error(f"Error calculating portfolio performance: {str(e)}")
            return 0, 0, 0
    
    def optimize_portfolio(self, optimization_type: str = "max_sharpe", max_assets: int = 15) -> Dict:
        """
        Optimize portfolio based on selected criteria
        
        Args:
            optimization_type: Type of optimization ('max_sharpe', 'min_volatility', 'max_return')
            max_assets: Maximum number of assets to include in optimization
            
        Returns:
            Dictionary with optimization results
        """
        try:
            if self.returns_data is None or self.returns_data.empty:
                st.error("No returns data available for optimization")
                return {}
                
            # Pre-filter assets using PCA and clustering
            if len(self.tickers) > max_assets:
                st.info(f"Pre-filtering from {len(self.tickers)} to {max_assets} assets using PCA and clustering")
                
                try:
                    # Transpose returns data so assets are rows and dates are columns
                    asset_returns = self.returns_data.T
                    
                    # Perform PCA to reduce dimensionality
                    pca = PCA(n_components=min(10, len(self.tickers)-1))
                    asset_pca = pca.fit_transform(asset_returns)
                    
                    # Cluster using KMeans
                    kmeans = KMeans(n_clusters=max_assets, random_state=42)
                    clusters = kmeans.fit_predict(asset_pca)
                    
                    # Select representative assets from each cluster
                    selected_tickers = []
                    for cluster_id in range(max_assets):
                        cluster_indices = np.where(clusters == cluster_id)[0]
                        if len(cluster_indices) > 0:
                            # Find asset closest to cluster center
                            distances = np.linalg.norm(
                                asset_pca[cluster_indices] - kmeans.cluster_centers_[cluster_id], 
                                axis=1
                            )
                            closest_idx = cluster_indices[np.argmin(distances)]
                            selected_tickers.append(self.tickers[closest_idx])
                    
                    # Update data for selected tickers only
                    self.tickers = selected_tickers
                    self.returns_data = self.returns_data[selected_tickers]
                    self.mean_returns = self.returns_data.mean()
                    self.cov_matrix = self.returns_data.cov()
                    
                except Exception as e:
                    st.error(f"Error during asset filtering: {str(e)}")
                    st.error("Using all assets for optimization")
                
            num_assets = len(self.tickers)
            
            # Constraints and bounds
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
            
            # Dynamic bounds based on number of assets
            max_weight = min(0.5, 1.5 / num_assets)  # Allow larger weights for smaller portfolios
            min_weight = max(0.005, 0.5 / num_assets)  # Higher minimum for smaller portfolios
            bounds = tuple((min_weight, max_weight) for _ in range(num_assets))
                        
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
            
            # First try SLSQP
            result = minimize(
                objective,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'ftol': 1e-9, 'disp': False}
            )
            
            # If SLSQP fails, try differential evolution
            if not result.success:
                st.warning("SLSQP optimization failed, trying differential evolution...")
                result = differential_evolution(
                    objective,
                    bounds,
                    strategy='best1bin',
                    maxiter=1000,
                    popsize=min(15, 5*num_assets),
                    constraints=constraints,
                    disp=False,
                    tol=1e-7
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
                'raw_weights': optimal_weights,
                'full_tickers': self.tickers.copy()  # Store full set of tickers used
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
            min_ret = self.mean_returns.min() * self.get_annualization_factor()
            max_ret = self.mean_returns.max() * self.get_annualization_factor()
            target_returns = np.linspace(min_ret, max_ret, num_portfolios)
            
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            
            # Dynamic bounds
            max_weight = min(0.5, 1.5 / num_assets)
            min_weight = max(0.005, 0.5 / num_assets)
            bounds = tuple((min_weight, max_weight) for _ in range(num_assets))
            
            # Parallel optimization
            def optimize_target(target):
                # Add return constraint
                return_constraint = {'type': 'eq', 'fun': lambda x: 
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
                    return (ret, vol, sharpe)
                else:
                    return (np.nan, np.nan, np.nan)
            
            # Run optimizations in parallel
            parallel_results = Parallel(n_jobs=-1)(
                delayed(optimize_target)(target) for target in target_returns
            )
            
            # Process results
            for i, res in enumerate(parallel_results):
                results[0, i] = res[0]
                results[1, i] = res[1]
                results[2, i] = res[2]
            
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
            
            # Use Dirichlet distribution for more realistic weights
            alpha = np.ones(num_assets) * 0.5  # Concentration parameter
            
            for i in range(num_simulations):
                # Generate random weights using Dirichlet distribution
                weights = np.random.dirichlet(alpha)
                
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
    
    def calculate_portfolio_metrics(self, weights: np.ndarray) -> Dict:
        """
        Calculate comprehensive portfolio metrics
        
        Args:
            weights: Portfolio weights array
            
        Returns:
            Dictionary with portfolio metrics
        """
        try:
            if self.returns_data is None or self.returns_data.empty:
                return {}
            
            # Get annualization factor
            annual_factor = self.get_annualization_factor()
            epsilon = 1e-8
            
            # Basic performance metrics
            expected_return, volatility, sharpe_ratio = self.portfolio_performance(weights)
            
            # Portfolio returns time series
            portfolio_returns = (self.returns_data * weights).sum(axis=1)
            
            # Additional metrics with consistent annualization
            portfolio_std = portfolio_returns.std() * np.sqrt(annual_factor) + epsilon
            skewness = portfolio_returns.skew()
            kurtosis = portfolio_returns.kurtosis()
            
            # Downside metrics
            downside_returns = portfolio_returns[portfolio_returns < 0]
            if not downside_returns.empty:
                downside_std = downside_returns.std() * np.sqrt(annual_factor) + epsilon
                sortino_ratio = (expected_return - self.risk_free_rate) / downside_std
            else:
                sortino_ratio = np.inf
            
            # Value at Risk (VaR) - 5% confidence level
            var_5 = portfolio_returns.quantile(0.05)
            
            # Conditional Value at Risk (CVaR)
            cvar_5 = portfolio_returns[portfolio_returns <= var_5].mean()
            
            # Maximum drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            peak = cumulative_returns.expanding(min_periods=1).max()
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = drawdown.min()
            
            # Calmar ratio
            calmar_ratio = expected_return / abs(max_drawdown) if max_drawdown < 0 else np.inf
            
            return {
                'expected_return': expected_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'var_5': var_5,
                'cvar_5': cvar_5,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'portfolio_std': portfolio_std
            }
            
        except Exception as e:
            st.error(f"Error calculating portfolio metrics: {str(e)}")
            return {}
    
    def get_asset_statistics(self) -> pd.DataFrame:
        """
        Get individual asset statistics
        
        Returns:
            DataFrame with asset statistics
        """
        try:
            if self.returns_data is None or self.returns_data.empty:
                return pd.DataFrame()
            
            annual_factor = self.get_annualization_factor()
            epsilon = 1e-8
            stats = []
            for ticker in self.tickers:
                returns = self.returns_data[ticker]
                
                # Calculate metrics
                annual_return = returns.mean() * annual_factor
                annual_vol = returns.std() * np.sqrt(annual_factor) + epsilon
                sharpe = (annual_return - self.risk_free_rate) / annual_vol
                
                # Downside metrics
                downside_returns = returns[returns < 0]
                if not downside_returns.empty:
                    downside_vol = downside_returns.std() * np.sqrt(annual_factor) + epsilon
                    sortino = (annual_return - self.risk_free_rate) / downside_vol
                else:
                    sortino = np.inf
                
                stats.append({
                    'Symbol': ticker,
                    'Annual_Return': annual_return,
                    'Annual_Volatility': annual_vol,
                    'Sharpe_Ratio': sharpe,
                    'Sortino_Ratio': sortino,
                    'Skewness': returns.skew(),
                    'Kurtosis': returns.kurtosis(),
                    'VaR_5': returns.quantile(0.05)
                })
            
            return pd.DataFrame(stats)
            
        except Exception as e:
            st.error(f"Error getting asset statistics: {str(e)}")
            return pd.DataFrame()


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
                
            col6, col7 = st.columns(2)
            with col6:
                max_assets = st.number_input("Max Assets in Optimization", 5, 50, 15)
            with col7:
                transaction_cost = st.number_input("Transaction Cost (%)", 0.0, 5.0, 0.1) / 100
        
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
            optimal_portfolio = optimizer.optimize_portfolio(optimization_type, max_assets)
            
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
                # Create allocation chart using the charts module
                allocation_chart = optimizer.charts.create_allocation_pie_chart(optimal_portfolio['weights'])
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
                
                # Plot efficient frontier using the charts module
                frontier_fig = optimizer.charts.plot_efficient_frontier(
                    frontier_df if frontier_df is not None else pd.DataFrame(),
                    simulation_df,
                    optimal_portfolio
                )
                st.plotly_chart(frontier_fig, use_container_width=True)
            
            # Comparison with current portfolio
            st.subheader("⚖️ Current vs Optimal Portfolio")
            
            # Get full list of tickers used in optimization
            full_tickers = optimal_portfolio.get('full_tickers', optimizer.tickers)
            
            # Calculate current portfolio weights
            current_weights = {}
            total_value = tracker.portfolio['market_value'].sum()
            for _, row in tracker.portfolio.iterrows():
                ticker = row['symbol']
                if ticker in full_tickers:
                    current_weights[ticker] = row['market_value'] / total_value
            
            # Create zero weights for assets in optimization but not in current portfolio
            optimal_weights_full = {ticker: 0 for ticker in full_tickers}
            optimal_weights_full.update(optimal_portfolio['weights'])
            
            if current_weights:
                # Create current weights array including all optimization assets
                current_weights_array = np.array([current_weights.get(ticker, 0) for ticker in full_tickers])
                
                current_performance = optimizer.portfolio_performance(current_weights_array)
                
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
                
                # Create weights comparison chart
                weights_comparison_fig = optimizer.charts.create_weights_comparison(
                    current_weights, 
                    optimal_portfolio['weights']
                )
                st.plotly_chart(weights_comparison_fig, use_container_width=True)
            
            # Asset statistics
            st.subheader("📈 Individual Asset Statistics")
            asset_stats = optimizer.get_asset_statistics()
            if not asset_stats.empty:
                # Format percentages
                for col in ['Annual_Return', 'Annual_Volatility', 'VaR_5']:
                    asset_stats[col] = asset_stats[col].apply(lambda x: f"{x:.2%}")
                for col in ['Sharpe_Ratio', 'Sortino_Ratio', 'Skewness', 'Kurtosis']:
                    asset_stats[col] = asset_stats[col].apply(lambda x: f"{x:.3f}")
                
                st.dataframe(asset_stats, use_container_width=True)
            
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