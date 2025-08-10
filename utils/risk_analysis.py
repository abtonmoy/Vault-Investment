import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import helpers
from utils.theme import THEME_COLORS, THEME_GRADIENTS

# Define consistent color palettes from theme
CHART_COLORS = [
    THEME_COLORS["lapis_lazuli"],
    THEME_COLORS["midnight_green"], 
    THEME_COLORS["pakistan_green"],
    THEME_COLORS["dark_green"],
    "#2E8B8B",  # Complementary teal
    "#4A90A4",  # Complementary blue-gray
    "#5B9279",  # Complementary green-gray
    "#6B7A8F"   # Complementary gray-blue
]

# High contrast colors for scatter plots against dark backgrounds
SCATTER_COLORS = [
    "#FF6B6B",  # Bright coral red
    "#4ECDC4",  # Bright teal
    "#45B7D1",  # Bright blue
    "#96CEB4",  # Mint green
    "#FFEAA7",  # Light yellow
    "#DDA0DD",  # Plum
    "#98D8C8",  # Aquamarine
    "#F7DC6F",  # Light gold
    "#BB8FCE",  # Light purple
    "#85C1E9",  # Light blue
    "#F8C471",  # Light orange
    "#82E0AA"   # Light green
]

def risk_return_scatter(portfolio_df):
    """Visualize risk-return relationship through volatility scatter plot"""
    if portfolio_df.empty or 'gain_loss_pct' not in portfolio_df.columns:
        return None

    # Ensure relevant columns are numeric
    portfolio_df['gain_loss_pct'] = pd.to_numeric(portfolio_df['gain_loss_pct'], errors='coerce')
    portfolio_df['market_value'] = pd.to_numeric(portfolio_df['market_value'], errors='coerce')
    portfolio_df['quantity'] = pd.to_numeric(portfolio_df['quantity'], errors='coerce')

    # Drop rows with missing or invalid data
    portfolio_df = portfolio_df.dropna(subset=['gain_loss_pct', 'market_value', 'quantity'])

    # Filter out rows with non-positive quantity or market value
    portfolio_df = portfolio_df[(portfolio_df['quantity'] > 0) & (portfolio_df['market_value'] > 0)]

    if portfolio_df.empty:
        st.warning("Insufficient data for risk analysis after cleaning.")
        return None

    fig = px.scatter(
        portfolio_df,
        x='market_value',
        y='gain_loss_pct',
        size='quantity',
        color='symbol',
        hover_name='symbol',
        title='Risk-Return Analysis: Position Size vs. Performance',
        labels={
            'market_value': 'Current Value ($)',
            'gain_loss_pct': 'Gain/Loss (%)'
        },
        color_discrete_sequence=SCATTER_COLORS
    )

    # Update marker styling
    fig.update_traces(
        marker=dict(
            line=dict(width=2, color='white'),
            opacity=0.8,
            sizemin=8
        )
    )

    # Add quadrant guide lines
    median_value = portfolio_df['market_value'].median()

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="rgba(255,255,255,0.8)",
        line_width=2,
        annotation_text="Break-even line",
        annotation_position="bottom right"
    )
    fig.add_vline(
        x=median_value,
        line_dash="dash",
        line_color="rgba(255,255,255,0.8)",
        line_width=2,
        annotation_text="Median position size",
        annotation_position="top left"
    )

    fig.update_layout(
        title=dict(
            font=dict(size=18, color='white'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text='Current Value ($)', font=dict(color='white')),
            tickfont=dict(color='white'),
            gridcolor="rgba(255,255,255,0.1)",
            type="log"  # Log scale for better visualization of different position sizes
        ),
        yaxis=dict(
            title=dict(text='Gain/Loss (%)', font=dict(color='white')),
            tickfont=dict(color='white'),
            gridcolor="rgba(255,255,255,0.1)",
            tickformat=".1%"
        ),
        legend=dict(
            font=dict(color='white'),
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        paper_bgcolor=THEME_COLORS["dark_green"],
        plot_bgcolor=THEME_COLORS["dark_green"],
        font=dict(color='white')
    )

    return fig

def portfolio_volatility_chart(historical_values):
    """Create rolling volatility chart to show risk over time"""
    if historical_values.empty or 'date' not in historical_values.columns:
        return None

    # Focus on total portfolio value
    portfolio_history = historical_values[
        historical_values['symbol'] == 'PORTFOLIO_TOTAL'
    ].copy()
    
    if portfolio_history.empty:
        return None

    portfolio_history['date'] = pd.to_datetime(portfolio_history['date'])
    portfolio_history = portfolio_history.sort_values('date')
    
    # Calculate returns
    portfolio_history['returns'] = portfolio_history['market_value'].pct_change()
    
    # Calculate rolling volatility (30-day)
    portfolio_history['rolling_volatility'] = portfolio_history['returns'].rolling(
        window=30, min_periods=10
    ).std() * np.sqrt(252)  # Annualized
    
    # Remove NaN values
    portfolio_history = portfolio_history.dropna()
    
    if portfolio_history.empty:
        return None

    fig = go.Figure()
    
    # Add volatility line
    fig.add_trace(go.Scatter(
        x=portfolio_history['date'],
        y=portfolio_history['rolling_volatility'],
        mode='lines',
        name='30-Day Rolling Volatility',
        line=dict(color=THEME_COLORS["lapis_lazuli"], width=3),
        fill='tonexty',
        fillcolor='rgba(52, 152, 219, 0.2)',
        hovertemplate="<b>Portfolio Volatility</b><br>Date: %{x}<br>Volatility: %{y:.2%}<extra></extra>"
    ))
    
    # Add average volatility line
    avg_volatility = portfolio_history['rolling_volatility'].mean()
    fig.add_hline(
        y=avg_volatility,
        line_dash="dash",
        line_color=THEME_COLORS["midnight_green"],
        line_width=2,
        annotation_text=f"Average: {avg_volatility:.2%}",
        annotation_position="top left"
    )

    fig.update_layout(
        title=dict(
            text='Portfolio Risk Analysis: Rolling 30-Day Volatility',
            font=dict(size=18, color='white'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text='Date', font=dict(color='white')),
            tickfont=dict(color='white'),
            gridcolor="rgba(255,255,255,0.1)"
        ),
        yaxis=dict(
            title=dict(text='Annualized Volatility', font=dict(color='white')),
            tickfont=dict(color='white'),
            gridcolor="rgba(255,255,255,0.1)",
            tickformat=".1%"
        ),
        paper_bgcolor=THEME_COLORS["dark_green"],
        plot_bgcolor=THEME_COLORS["dark_green"],
        font=dict(color='white'),
        showlegend=False
    )

    return fig

def drawdown_chart(historical_values):
    """Create drawdown chart to visualize portfolio declines"""
    if historical_values.empty or 'date' not in historical_values.columns:
        return None

    # Focus on total portfolio value
    portfolio_history = historical_values[
        historical_values['symbol'] == 'PORTFOLIO_TOTAL'
    ].copy()
    
    if portfolio_history.empty:
        return None

    portfolio_history['date'] = pd.to_datetime(portfolio_history['date'])
    portfolio_history = portfolio_history.sort_values('date')
    
    # Calculate running maximum and drawdown
    portfolio_history['running_max'] = portfolio_history['market_value'].expanding().max()
    portfolio_history['drawdown'] = (
        portfolio_history['market_value'] / portfolio_history['running_max'] - 1
    ) * 100
    
    fig = go.Figure()
    
    # Add drawdown area chart
    fig.add_trace(go.Scatter(
        x=portfolio_history['date'],
        y=portfolio_history['drawdown'],
        mode='lines',
        name='Drawdown',
        line=dict(color=THEME_COLORS["rojo"], width=2),
        fill='tonexty',
        fillcolor='rgba(231, 76, 60, 0.3)',
        hovertemplate="<b>Portfolio Drawdown</b><br>Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>"
    ))
    
    # Add zero line
    fig.add_hline(
        y=0,
        line_color="rgba(255,255,255,0.8)",
        line_width=1
    )
    
    # Highlight maximum drawdown
    max_dd_idx = portfolio_history['drawdown'].idxmin()
    max_drawdown = portfolio_history.loc[max_dd_idx]
    
    fig.add_trace(go.Scatter(
        x=[max_drawdown['date']],
        y=[max_drawdown['drawdown']],
        mode='markers',
        name='Maximum Drawdown',
        marker=dict(color='white', size=12, symbol='circle'),
        hovertemplate=f"<b>Max Drawdown</b><br>Date: {max_drawdown['date']}<br>Drawdown: {max_drawdown['drawdown']:.2f}%<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text='Portfolio Drawdown Analysis',
            font=dict(size=18, color='white'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text='Date', font=dict(color='white')),
            tickfont=dict(color='white'),
            gridcolor="rgba(255,255,255,0.1)"
        ),
        yaxis=dict(
            title=dict(text='Drawdown (%)', font=dict(color='white')),
            tickfont=dict(color='white'),
            gridcolor="rgba(255,255,255,0.1)"
        ),
        legend=dict(
            font=dict(color='white'),
            yanchor="bottom",
            y=0.02,
            xanchor="right",
            x=0.98
        ),
        paper_bgcolor=THEME_COLORS["dark_green"],
        plot_bgcolor=THEME_COLORS["dark_green"],
        font=dict(color='white')
    )

    return fig

def value_at_risk_chart(historical_values, confidence_levels=[0.95, 0.99]):
    """Calculate and visualize Value at Risk (VaR) at different confidence levels"""
    if historical_values.empty or 'date' not in historical_values.columns:
        return None

    # Focus on total portfolio value
    portfolio_history = historical_values[
        historical_values['symbol'] == 'PORTFOLIO_TOTAL'
    ].copy()
    
    if portfolio_history.empty:
        return None

    portfolio_history['date'] = pd.to_datetime(portfolio_history['date'])
    portfolio_history = portfolio_history.sort_values('date')
    
    # Calculate returns
    portfolio_history['returns'] = portfolio_history['market_value'].pct_change()
    portfolio_history = portfolio_history.dropna()
    
    if len(portfolio_history) < 30:  # Need sufficient data
        return None

    # Calculate VaR for different confidence levels
    var_data = []
    for confidence in confidence_levels:
        var_percentile = (1 - confidence) * 100
        var_value = np.percentile(portfolio_history['returns'], var_percentile) * 100
        var_data.append({
            'confidence': f"{confidence*100:.0f}%",
            'var': var_value,
            'var_abs': var_value  # For absolute positioning
        })
    
    var_df = pd.DataFrame(var_data)
    
    # Create distribution histogram
    fig = go.Figure()
    
    # Add histogram of returns
    fig.add_trace(go.Histogram(
        x=portfolio_history['returns'] * 100,
        nbinsx=50,
        name='Return Distribution',
        marker_color=THEME_COLORS["lapis_lazuli"],
        opacity=0.7,
        yaxis='y',
        hovertemplate="<b>Returns</b><br>Range: %{x}%<br>Frequency: %{y}<extra></extra>"
    ))
    
    # Add VaR lines
    colors = [THEME_COLORS["rojo"], "#FF4444"]
    for i, row in var_df.iterrows():
        fig.add_vline(
            x=row['var'],
            line_dash="dash",
            line_color=colors[i % len(colors)],
            line_width=3,
            annotation_text=f"VaR {row['confidence']}: {row['var']:.2f}%",
            annotation_position="top"
        )

    fig.update_layout(
        title=dict(
            text='Value at Risk (VaR) Analysis',
            font=dict(size=18, color='white'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text='Daily Returns (%)', font=dict(color='white')),
            tickfont=dict(color='white'),
            gridcolor="rgba(255,255,255,0.1)"
        ),
        yaxis=dict(
            title=dict(text='Frequency', font=dict(color='white')),
            tickfont=dict(color='white'),
            gridcolor="rgba(255,255,255,0.1)"
        ),
        paper_bgcolor=THEME_COLORS["dark_green"],
        plot_bgcolor=THEME_COLORS["dark_green"],
        font=dict(color='white'),
        showlegend=False
    )

    return fig

def correlation_heatmap(portfolio_df, historical_values):
    """Create correlation matrix heatmap for portfolio assets"""
    if historical_values.empty or portfolio_df.empty:
        return None

    # Get individual asset returns
    asset_returns = []
    symbols = portfolio_df['symbol'].unique()
    
    for symbol in symbols:
        symbol_data = historical_values[
            historical_values['symbol'] == symbol
        ].copy()
        
        if len(symbol_data) > 10:  # Need sufficient data
            symbol_data['date'] = pd.to_datetime(symbol_data['date'])
            symbol_data = symbol_data.sort_values('date')
            symbol_data[f'{symbol}_returns'] = symbol_data['market_value'].pct_change()
            
            asset_returns.append(
                symbol_data[['date', f'{symbol}_returns']].dropna()
            )
    
    if len(asset_returns) < 2:  # Need at least 2 assets
        return None

    # Merge all returns on date
    returns_df = asset_returns[0]
    for df in asset_returns[1:]:
        returns_df = pd.merge(returns_df, df, on='date', how='outer')
    
    # Calculate correlation matrix
    return_cols = [col for col in returns_df.columns if col.endswith('_returns')]
    if len(return_cols) < 2:
        return None
        
    corr_matrix = returns_df[return_cols].corr()
    
    # Clean column names for display
    corr_matrix.columns = [col.replace('_returns', '') for col in corr_matrix.columns]
    corr_matrix.index = [idx.replace('_returns', '') for idx in corr_matrix.index]

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,  # Center colorscale at 0
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10, "color": "white"},
        hoverongaps=False,
        hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>",
        colorbar=dict(
            title="Correlation",
            title_side="right",
            tickfont=dict(color='white'),
            title_font=dict(color='white')
        )
    ))

    fig.update_layout(
        title=dict(
            text='Asset Correlation Matrix',
            font=dict(size=18, color='white'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text='Assets', font=dict(color='white')),
            tickfont=dict(color='white'),
            tickangle=45
        ),
        yaxis=dict(
            title=dict(text='Assets', font=dict(color='white')),
            tickfont=dict(color='white')
        ),
        paper_bgcolor=THEME_COLORS["dark_green"],
        plot_bgcolor=THEME_COLORS["dark_green"],
        font=dict(color='white'),
        width=600,
        height=600
    )

    return fig

def beta_analysis_chart(portfolio_df, historical_values, benchmark_symbol='SPY'):
    """Analyze portfolio beta relative to a benchmark"""
    if historical_values.empty or portfolio_df.empty:
        return None

    # Get portfolio returns
    portfolio_data = historical_values[
        historical_values['symbol'] == 'PORTFOLIO_TOTAL'
    ].copy()
    
    # In a real implementation, you'd fetch benchmark data
    # For now, we'll create a synthetic benchmark or use existing data
    # This is a placeholder - in practice, you'd fetch actual market data
    
    if portfolio_data.empty or len(portfolio_data) < 30:
        return None

    portfolio_data['date'] = pd.to_datetime(portfolio_data['date'])
    portfolio_data = portfolio_data.sort_values('date')
    portfolio_data['portfolio_returns'] = portfolio_data['market_value'].pct_change()
    
    # Create synthetic benchmark returns for demonstration
    np.random.seed(42)  # For reproducibility
    benchmark_returns = np.random.normal(0.0003, 0.01, len(portfolio_data))
    portfolio_data['benchmark_returns'] = benchmark_returns
    
    # Remove NaN values
    clean_data = portfolio_data.dropna()
    
    if len(clean_data) < 10:
        return None

    # Calculate beta
    covariance = np.cov(clean_data['portfolio_returns'], clean_data['benchmark_returns'])[0][1]
    benchmark_variance = np.var(clean_data['benchmark_returns'])
    beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
    
    # Create scatter plot
    fig = px.scatter(
        clean_data,
        x='benchmark_returns',
        y='portfolio_returns',
        title=f'Portfolio Beta Analysis (β = {beta:.2f})',
        labels={
            'benchmark_returns': f'{benchmark_symbol} Returns',
            'portfolio_returns': 'Portfolio Returns'
        },
        color_discrete_sequence=[THEME_COLORS["lapis_lazuli"]]
    )
    
    # Add regression line
    z = np.polyfit(clean_data['benchmark_returns'], clean_data['portfolio_returns'], 1)
    p = np.poly1d(z)
    
    x_line = np.linspace(clean_data['benchmark_returns'].min(), 
                        clean_data['benchmark_returns'].max(), 100)
    y_line = p(x_line)
    
    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        name=f'Beta Line (β = {beta:.2f})',
        line=dict(color=THEME_COLORS["rojo"], width=3)
    ))

    fig.update_layout(
        title=dict(
            font=dict(size=18, color='white'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text=f'{benchmark_symbol} Returns', font=dict(color='white')),
            tickfont=dict(color='white'),
            gridcolor="rgba(255,255,255,0.1)",
            tickformat=".1%"
        ),
        yaxis=dict(
            title=dict(text='Portfolio Returns', font=dict(color='white')),
            tickfont=dict(color='white'),
            gridcolor="rgba(255,255,255,0.1)",
            tickformat=".1%"
        ),
        legend=dict(
            font=dict(color='white'),
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        paper_bgcolor=THEME_COLORS["dark_green"],
        plot_bgcolor=THEME_COLORS["dark_green"],
        font=dict(color='white')
    )

    return fig

def risk_metrics_summary(portfolio_df, historical_values, risk_free_rate=0.02):
    """Calculate and display comprehensive risk metrics"""
    if historical_values.empty:
        return None

    # Get portfolio history
    portfolio_history = historical_values[
        historical_values['symbol'] == 'PORTFOLIO_TOTAL'
    ].copy()
    
    if portfolio_history.empty:
        return None

    portfolio_history['date'] = pd.to_datetime(portfolio_history['date'])
    portfolio_history = portfolio_history.sort_values('date')
    portfolio_history['returns'] = portfolio_history['market_value'].pct_change()
    
    clean_returns = portfolio_history['returns'].dropna()
    
    if len(clean_returns) < 30:
        return None

    # Calculate risk metrics
    metrics = {}
    
    # Basic statistics
    metrics['Volatility (Annualized)'] = clean_returns.std() * np.sqrt(252)
    metrics['Skewness'] = clean_returns.skew()
    metrics['Kurtosis'] = clean_returns.kurtosis()
    
    # VaR and CVaR
    metrics['VaR (95%)'] = np.percentile(clean_returns, 5)
    metrics['VaR (99%)'] = np.percentile(clean_returns, 1)
    metrics['CVaR (95%)'] = clean_returns[clean_returns <= metrics['VaR (95%)']].mean()
    
    # Drawdown metrics
    portfolio_history['running_max'] = portfolio_history['market_value'].expanding().max()
    portfolio_history['drawdown'] = (
        portfolio_history['market_value'] / portfolio_history['running_max'] - 1
    )
    metrics['Maximum Drawdown'] = portfolio_history['drawdown'].min()
    
    # Ratios
    excess_returns = clean_returns.mean() - risk_free_rate / 252
    metrics['Sharpe Ratio'] = excess_returns / clean_returns.std() * np.sqrt(252)
    
    # Downside deviation for Sortino ratio
    downside_returns = clean_returns[clean_returns < 0]
    if len(downside_returns) > 0:
        downside_std = downside_returns.std()
        metrics['Sortino Ratio'] = excess_returns / downside_std * np.sqrt(252)
    else:
        metrics['Sortino Ratio'] = np.inf

    return metrics

def render_risk_analysis(tracker):
    """Render all risk analysis visualizations"""
    if tracker.portfolio.empty:
        st.warning("No portfolio data available for risk analysis.")
        return
    
    st.header("⚠️ Risk Analysis Dashboard")
    
    # Validate data
    clean_portfolio = helpers.validate_portfolio_data(tracker.portfolio.copy())
    
    # Risk metrics summary
    risk_metrics = risk_metrics_summary(
        clean_portfolio, 
        tracker.historical_values, 
        getattr(tracker, 'risk_free_rate', 0.02)
    )
    
    if risk_metrics:
        st.subheader("📈 Risk Metrics Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Volatility", f"{risk_metrics['Volatility (Annualized)']:.2%}")
            st.metric("Skewness", f"{risk_metrics['Skewness']:.3f}")
        
        with col2:
            st.metric("VaR (95%)", f"{risk_metrics['VaR (95%)']:.2%}")
            st.metric("CVaR (95%)", f"{risk_metrics['CVaR (95%)']:.2%}")
        
        with col3:
            st.metric("Max Drawdown", f"{risk_metrics['Maximum Drawdown']:.2%}")
            st.metric("Sharpe Ratio", f"{risk_metrics['Sharpe Ratio']:.3f}")
        
        with col4:
            st.metric("Sortino Ratio", f"{risk_metrics['Sortino Ratio']:.3f}")
            st.metric("Kurtosis", f"{risk_metrics['Kurtosis']:.3f}")
    
    # Risk-Return Scatter
    risk_return_fig = risk_return_scatter(clean_portfolio)
    if risk_return_fig:
        st.plotly_chart(risk_return_fig, use_container_width=True)
    
    # Volatility and Drawdown Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        volatility_fig = portfolio_volatility_chart(tracker.historical_values)
        if volatility_fig:
            st.plotly_chart(volatility_fig, use_container_width=True)
    
    with col2:
        drawdown_fig = drawdown_chart(tracker.historical_values)
        if drawdown_fig:
            st.plotly_chart(drawdown_fig, use_container_width=True)
    
    # Value at Risk Analysis
    var_fig = value_at_risk_chart(tracker.historical_values)
    if var_fig:
        st.plotly_chart(var_fig, use_container_width=True)
    
    # Correlation Analysis
    col3, col4 = st.columns(2)
    
    with col3:
        correlation_fig = correlation_heatmap(clean_portfolio, tracker.historical_values)
        if correlation_fig:
            st.plotly_chart(correlation_fig, use_container_width=True)
    
    with col4:
        beta_fig = beta_analysis_chart(clean_portfolio, tracker.historical_values)
        if beta_fig:
            st.plotly_chart(beta_fig, use_container_width=True)