import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional


class PortfolioCharts:
    """Separate class for handling all portfolio visualization charts"""
    
    def __init__(self, template='plotly_dark'):
        self.template = template
    
    def plot_efficient_frontier(self, 
                              frontier_df: pd.DataFrame, 
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
                template=self.template,
                width=800,
                height=600
            )
            
            # Format axes as percentages
            fig.update_xaxes(tickformat='.1%')
            fig.update_yaxes(tickformat='.1%')
            
            return fig
            
        except Exception as e:
            st.error(f"Error plotting efficient frontier: {str(e)}")
            return go.Figure()
    
    def create_allocation_pie_chart(self, weights_dict: Dict[str, float]) -> go.Figure:
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
                template=self.template,
                width=600,
                height=600
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating allocation chart: {str(e)}")
            return go.Figure()
    
    def create_allocation_bar_chart(self, weights_dict: Dict[str, float]) -> go.Figure:
        """
        Create horizontal bar chart for portfolio allocation
        
        Args:
            weights_dict: Dictionary of ticker weights
            
        Returns:
            Plotly figure
        """
        try:
            # Sort by weight descending
            sorted_items = sorted(weights_dict.items(), key=lambda x: x[1], reverse=True)
            symbols = [item[0] for item in sorted_items]
            weights = [item[1] for item in sorted_items]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=weights,
                    y=symbols,
                    orientation='h',
                    marker_color='skyblue',
                    hovertemplate="<b>%{y}</b><br>" +
                                "Weight: %{x:.2%}<extra></extra>"
                )
            ])
            
            fig.update_layout(
                title='Portfolio Allocation by Weight',
                xaxis_title='Weight (%)',
                yaxis_title='Assets',
                template=self.template,
                height=max(400, len(symbols) * 30)
            )
            
            fig.update_xaxes(tickformat='.1%')
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating bar chart: {str(e)}")
            return go.Figure()
    
    def create_risk_return_scatter(self, returns_data: pd.DataFrame, 
                                 weights_dict: Dict[str, float] = None) -> go.Figure:
        """
        Create scatter plot of individual asset risk vs return
        
        Args:
            returns_data: DataFrame with asset returns
            weights_dict: Optional portfolio weights to highlight
            
        Returns:
            Plotly figure
        """
        try:
            # Calculate annual metrics for each asset
            annual_returns = returns_data.mean() * 252
            annual_volatility = returns_data.std() * np.sqrt(252)
            
            # Create scatter plot
            fig = go.Figure()
            
            for asset in returns_data.columns:
                # Determine marker size based on portfolio weight if provided
                marker_size = 10
                if weights_dict and asset in weights_dict:
                    marker_size = max(10, weights_dict[asset] * 200)  # Scale weight to marker size
                
                fig.add_trace(go.Scatter(
                    x=[annual_volatility[asset]],
                    y=[annual_returns[asset]],
                    mode='markers+text',
                    marker=dict(size=marker_size, opacity=0.7),
                    text=[asset],
                    textposition="top center",
                    name=asset,
                    hovertemplate="<b>%{text}</b><br>" +
                                f"Return: {annual_returns[asset]:.2%}<br>" +
                                f"Volatility: {annual_volatility[asset]:.2%}<br>" +
                                f"Weight: {weights_dict.get(asset, 0):.2%}<extra></extra>" if weights_dict else
                                "<b>%{text}</b><br>" +
                                f"Return: {annual_returns[asset]:.2%}<br>" +
                                f"Volatility: {annual_volatility[asset]:.2%}<extra></extra>"
                ))
            
            fig.update_layout(
                title='Individual Asset Risk vs Return',
                xaxis_title='Volatility (Risk)',
                yaxis_title='Expected Return',
                template=self.template,
                showlegend=False,
                width=800,
                height=600
            )
            
            fig.update_xaxes(tickformat='.1%')
            fig.update_yaxes(tickformat='.1%')
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating risk-return scatter: {str(e)}")
            return go.Figure()
    
    def create_correlation_heatmap(self, returns_data: pd.DataFrame) -> go.Figure:
        """
        Create correlation heatmap of asset returns
        
        Args:
            returns_data: DataFrame with asset returns
            
        Returns:
            Plotly figure
        """
        try:
            correlation_matrix = returns_data.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title="Correlation"),
                hovertemplate="<b>%{y} vs %{x}</b><br>" +
                            "Correlation: %{z:.3f}<extra></extra>"
            ))
            
            fig.update_layout(
                title='Asset Correlation Matrix',
                template=self.template,
                width=600,
                height=600
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating correlation heatmap: {str(e)}")
            return go.Figure()
    
    def create_performance_comparison(self, portfolios: Dict[str, Dict]) -> go.Figure:
        """
        Create bar chart comparing different portfolio performances
        
        Args:
            portfolios: Dictionary of portfolio name to performance metrics
            
        Returns:
            Plotly figure
        """
        try:
            portfolio_names = list(portfolios.keys())
            returns = [portfolios[name]['expected_return'] for name in portfolio_names]
            volatilities = [portfolios[name]['volatility'] for name in portfolio_names]
            sharpe_ratios = [portfolios[name]['sharpe_ratio'] for name in portfolio_names]
            
            fig = go.Figure()
            
            # Add return bars
            fig.add_trace(go.Bar(
                x=portfolio_names,
                y=returns,
                name='Expected Return',
                yaxis='y',
                hovertemplate="<b>%{x}</b><br>Return: %{y:.2%}<extra></extra>"
            ))
            
            # Add Sharpe ratio line
            fig.add_trace(go.Scatter(
                x=portfolio_names,
                y=sharpe_ratios,
                mode='lines+markers',
                name='Sharpe Ratio',
                yaxis='y2',
                line=dict(color='red', width=3),
                hovertemplate="<b>%{x}</b><br>Sharpe: %{y:.3f}<extra></extra>"
            ))
            
            # Update layout with secondary y-axis
            fig.update_layout(
                title='Portfolio Performance Comparison',
                xaxis_title='Portfolio',
                yaxis=dict(title='Expected Return', tickformat='.1%'),
                yaxis2=dict(
                    title='Sharpe Ratio',
                    overlaying='y',
                    side='right'
                ),
                template=self.template,
                width=800,
                height=500
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating performance comparison: {str(e)}")
            return go.Figure()
    
    def create_monte_carlo_distribution(self, simulation_df: pd.DataFrame, 
                                      metric: str = 'Sharpe') -> go.Figure:
        """
        Create histogram of Monte Carlo simulation results
        
        Args:
            simulation_df: DataFrame with simulation results
            metric: Metric to plot distribution for ('Return', 'Volatility', 'Sharpe')
            
        Returns:
            Plotly figure
        """
        try:
            if metric not in simulation_df.columns:
                st.error(f"Metric '{metric}' not found in simulation data")
                return go.Figure()
            
            fig = go.Figure(data=[
                go.Histogram(
                    x=simulation_df[metric],
                    nbinsx=50,
                    marker_color='lightblue',
                    opacity=0.7,
                    hovertemplate=f"<b>{metric}</b><br>" +
                                f"Range: %{{x}}<br>" +
                                "Count: %{y}<extra></extra>"
                )
            ])
            
            # Add mean line
            mean_value = simulation_df[metric].mean()
            fig.add_vline(
                x=mean_value,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_value:.3f}"
            )
            
            fig.update_layout(
                title=f'Monte Carlo Simulation - {metric} Distribution',
                xaxis_title=metric,
                yaxis_title='Frequency',
                template=self.template,
                width=700,
                height=500
            )
            
            if metric in ['Return']:
                fig.update_xaxes(tickformat='.1%')
            elif metric in ['Volatility']:
                fig.update_xaxes(tickformat='.1%')
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating Monte Carlo distribution: {str(e)}")
            return go.Figure()
    
    def create_drawdown_chart(self, returns_data: pd.DataFrame, 
                            weights: np.ndarray) -> go.Figure:
        """
        Create drawdown chart for portfolio
        
        Args:
            returns_data: DataFrame with asset returns
            weights: Portfolio weights array
            
        Returns:
            Plotly figure
        """
        try:
            # Calculate portfolio returns
            portfolio_returns = (returns_data * weights).sum(axis=1)
            
            # Calculate cumulative returns and drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            peak = cumulative_returns.expanding(min_periods=1).max()
            drawdown = (cumulative_returns - peak) / peak
            
            fig = go.Figure()
            
            # Add cumulative returns
            fig.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns,
                mode='lines',
                name='Cumulative Returns',
                line=dict(color='blue'),
                hovertemplate="Date: %{x}<br>Value: %{y:.2f}<extra></extra>"
            ))
            
            # Add peak line
            fig.add_trace(go.Scatter(
                x=peak.index,
                y=peak,
                mode='lines',
                name='Peak',
                line=dict(color='green', dash='dash'),
                hovertemplate="Date: %{x}<br>Peak: %{y:.2f}<extra></extra>"
            ))
            
            # Add drawdown on secondary axis
            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown,
                mode='lines',
                name='Drawdown',
                yaxis='y2',
                line=dict(color='red'),
                fill='tonexty',
                hovertemplate="Date: %{x}<br>Drawdown: %{y:.2%}<extra></extra>"
            ))
            
            fig.update_layout(
                title='Portfolio Performance and Drawdown',
                xaxis_title='Date',
                yaxis=dict(title='Cumulative Returns'),
                yaxis2=dict(
                    title='Drawdown',
                    overlaying='y',
                    side='right',
                    tickformat='.1%'
                ),
                template=self.template,
                width=900,
                height=600
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating drawdown chart: {str(e)}")
            return go.Figure()
    
    def create_weights_comparison(self, current_weights: Dict[str, float], 
                                optimal_weights: Dict[str, float]) -> go.Figure:
        """
        Create side-by-side comparison of current vs optimal weights
        
        Args:
            current_weights: Current portfolio weights
            optimal_weights: Optimal portfolio weights
            
        Returns:
            Plotly figure
        """
        try:
            # Get all unique symbols
            all_symbols = list(set(list(current_weights.keys()) + list(optimal_weights.keys())))
            
            current_vals = [current_weights.get(symbol, 0) for symbol in all_symbols]
            optimal_vals = [optimal_weights.get(symbol, 0) for symbol in all_symbols]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Current Portfolio',
                x=all_symbols,
                y=current_vals,
                marker_color='lightblue',
                hovertemplate="<b>%{x}</b><br>Current: %{y:.2%}<extra></extra>"
            ))
            
            fig.add_trace(go.Bar(
                name='Optimal Portfolio',
                x=all_symbols,
                y=optimal_vals,
                marker_color='lightcoral',
                hovertemplate="<b>%{x}</b><br>Optimal: %{y:.2%}<extra></extra>"
            ))
            
            fig.update_layout(
                title='Current vs Optimal Portfolio Weights',
                xaxis_title='Assets',
                yaxis_title='Weight (%)',
                barmode='group',
                template=self.template,
                width=800,
                height=500
            )
            
            fig.update_yaxes(tickformat='.1%')
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating weights comparison: {str(e)}")
            return go.Figure()
        
    def plot_efficient_frontier_with_multiple_portfolios(self, 
                                                   frontier_df: pd.DataFrame, 
                                                   simulation_df: pd.DataFrame = None,
                                                   all_portfolios: Dict = None) -> go.Figure:
        """
        Plot efficient frontier with multiple optimization results
        
        Args:
            frontier_df: Efficient frontier data
            simulation_df: Monte Carlo simulation data
            all_portfolios: Dictionary of all optimization results
            
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
            
            # Plot all optimization results if available
            if all_portfolios:
                colors = ['gold', 'cyan', 'magenta', 'lime', 'orange']  # Different colors for each method
                symbols = ['star', 'diamond', 'square', 'triangle-up', 'circle']
                
                for i, (method_key, portfolio) in enumerate(all_portfolios.items()):
                    if 'volatility' in portfolio and 'expected_return' in portfolio:
                        fig.add_trace(go.Scatter(
                            x=[portfolio['volatility']],
                            y=[portfolio['expected_return']],
                            mode='markers',
                            marker=dict(
                                size=15,
                                color=colors[i % len(colors)],
                                symbol=symbols[i % len(symbols)],
                                line=dict(color='black', width=2)
                            ),
                            name=f"{portfolio.get('method', method_key)}",
                            hovertemplate=f"<b>{portfolio.get('method', method_key)}</b><br>" +
                                        "Return: %{y:.2%}<br>" +
                                        "Volatility: %{x:.2%}<br>" +
                                        f"Sharpe: {portfolio.get('sharpe_ratio', 0):.3f}<extra></extra>"
                        ))
            
            # Update layout
            fig.update_layout(
                title='Efficient Frontier with Multiple Optimization Methods',
                xaxis_title='Volatility (Risk)',
                yaxis_title='Expected Return',
                hovermode='closest',
                template=self.template,
                width=800,
                height=600,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255,255,255,0.8)"
                )
            )
            
            # Format axes as percentages
            fig.update_xaxes(tickformat='.1%')
            fig.update_yaxes(tickformat='.1%')
            
            return fig
            
        except Exception as e:
            st.error(f"Error plotting enhanced efficient frontier: {str(e)}")
            return go.Figure()