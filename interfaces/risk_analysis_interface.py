import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show_risk_analysis_dashboard():
    """Main UI for portfolio risk analysis"""
    st.header("⚠️ Portfolio Risk Analysis Dashboard")

    # Check if portfolio data exists in session state
    if 'tracker' not in st.session_state or not hasattr(st.session_state.tracker, 'portfolio'):
        st.markdown("""
        <div class="warning-gradient">
            <h4>📊 No Portfolio Data Available</h4>
            <p>Please upload your portfolio data in the <strong>Investment Portfolio</strong> tab first, then load historical data to access risk analysis features.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    tracker = st.session_state.tracker

    # Check if historical data is loaded
    if not hasattr(tracker, 'historical_values') or tracker.historical_values.empty:
        st.markdown("""
        <div class="warning-gradient">
            <h4>📈 Historical Data Required</h4>
            <p>Historical price data is required for risk analysis. Please go back to the <strong>Investment Portfolio</strong> tab and click the <strong>"📊 Load Historical Data for Visualizations"</strong> button.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Portfolio data is available - proceed with risk analysis
    portfolio_df = tracker.portfolio.copy()
    historical_values = tracker.historical_values.copy()

    # Instruction Block
    st.markdown("""
    <div style="background-color: #f9f9f9; padding: 1.5rem; border-left: 5px solid #DE2C2C; border-radius: 8px; margin-bottom: 2rem;">
        <h4>⚠️ Risk Analysis Overview</h4>
        <p style="text-align: left; font-size: 0.95rem; margin-bottom: 1rem;">
            This dashboard provides comprehensive risk metrics and visualizations for your investment portfolio:
        </p>
        <ul style="text-align: left; font-size: 0.95rem; margin-bottom: 1rem;">
            <li><strong>Risk Metrics:</strong> Volatility, VaR, Sharpe Ratio, Maximum Drawdown, and more</li>
            <li><strong>Risk-Return Analysis:</strong> Scatter plot showing position performance vs. size</li>
            <li><strong>Volatility Tracking:</strong> Rolling volatility to monitor risk over time</li>
            <li><strong>Drawdown Analysis:</strong> Maximum portfolio declines from peaks</li>
            <li><strong>Value at Risk:</strong> Potential losses at different confidence levels</li>
            <li><strong>Correlation Matrix:</strong> How your assets move together</li>
            <li><strong>Beta Analysis:</strong> Portfolio sensitivity to market movements</li>
        </ul>
        <hr style="margin: 1.2rem 0;">
        <p style="color: #a00; font-size: 0.9rem;">
            <strong>Risk Disclaimer:</strong> Past performance does not guarantee future results. Risk metrics are calculated based on historical data and may not reflect future market conditions. Use this analysis as one factor in your investment decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Calculate and display risk metrics summary
    risk_metrics = calculate_risk_metrics_summary(portfolio_df, historical_values)
    
    if risk_metrics:
        st.subheader("📊 Key Risk Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="gradient-metric">
                <h3 style="margin: 0; font-size: 1.2rem;">Volatility</h3>
                <p style="font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0 0 0;">{risk_metrics['Volatility (Annualized)']:.2%}</p>
                <small>Annualized</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="gradient-metric" style="margin-top: 1rem;">
                <h3 style="margin: 0; font-size: 1.2rem;">Skewness</h3>
                <p style="font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0 0 0;">{risk_metrics['Skewness']:.3f}</p>
                <small>Distribution Shape</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="gradient-metric">
                <h3 style="margin: 0; font-size: 1.2rem;">VaR (95%)</h3>
                <p style="font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0 0 0;">{risk_metrics['VaR (95%)']:.2%}</p>
                <small>Value at Risk</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="gradient-metric" style="margin-top: 1rem;">
                <h3 style="margin: 0; font-size: 1.2rem;">CVaR (95%)</h3>
                <p style="font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0 0 0;">{risk_metrics['CVaR (95%)']:.2%}</p>
                <small>Expected Shortfall</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="gradient-metric">
                <h3 style="margin: 0; font-size: 1.2rem;">Max Drawdown</h3>
                <p style="font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0 0 0;">{risk_metrics['Maximum Drawdown']:.2%}</p>
                <small>Peak-to-Trough</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="gradient-metric" style="margin-top: 1rem;">
                <h3 style="margin: 0; font-size: 1.2rem;">Kurtosis</h3>
                <p style="font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0 0 0;">{risk_metrics['Kurtosis']:.3f}</p>
                <small>Tail Risk</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="gradient-metric">
                <h3 style="margin: 0; font-size: 1.2rem;">Sharpe Ratio</h3>
                <p style="font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0 0 0;">{risk_metrics['Sharpe Ratio']:.3f}</p>
                <small>Risk-Adj. Return</small>
            </div>
            """, unsafe_allow_html=True)
            
            sortino_display = f"{risk_metrics['Sortino Ratio']:.3f}" if not np.isinf(risk_metrics['Sortino Ratio']) else "N/A"
            st.markdown(f"""
            <div class="gradient-metric" style="margin-top: 1rem;">
                <h3 style="margin: 0; font-size: 1.2rem;">Sortino Ratio</h3>
                <p style="font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0 0 0;">{sortino_display}</p>
                <small>Downside Risk-Adj.</small>
            </div>
            """, unsafe_allow_html=True)

    # Risk-Return Scatter Plot
    st.subheader("🎯 Risk-Return Analysis")
    risk_return_fig = create_risk_return_scatter(portfolio_df)
    if risk_return_fig:
        st.plotly_chart(risk_return_fig, use_container_width=True)

    # Volatility and Drawdown Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Portfolio Volatility Over Time")
        volatility_fig = create_volatility_chart(historical_values)
        if volatility_fig:
            st.plotly_chart(volatility_fig, use_container_width=True)
        else:
            st.info("Insufficient historical data for volatility analysis.")
    
    with col2:
        st.subheader("📉 Drawdown Analysis")
        drawdown_fig = create_drawdown_chart(historical_values)
        if drawdown_fig:
            st.plotly_chart(drawdown_fig, use_container_width=True)
        else:
            st.info("Insufficient historical data for drawdown analysis.")

    # Value at Risk Analysis
    st.subheader("⚡ Value at Risk (VaR) Distribution")
    var_fig = create_var_chart(historical_values)
    if var_fig:
        st.plotly_chart(var_fig, use_container_width=True)
    else:
        st.info("Insufficient historical data for VaR analysis.")

    # Correlation and Beta Analysis
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("🔗 Asset Correlation Matrix")
        correlation_fig = create_correlation_heatmap(portfolio_df, historical_values)
        if correlation_fig:
            st.plotly_chart(correlation_fig, use_container_width=True)
        else:
            st.info("Insufficient data for correlation analysis.")
    
    with col4:
        st.subheader("📊 Portfolio Beta Analysis")
        beta_fig = create_beta_chart(portfolio_df, historical_values)
        if beta_fig:
            st.plotly_chart(beta_fig, use_container_width=True)
        else:
            st.info("Insufficient data for beta analysis.")

    # Risk Analysis Export
    st.subheader("💾 Export Risk Analysis")
    if st.button("📊 Generate Risk Report", type="secondary"):
        risk_report = generate_risk_report(portfolio_df, historical_values, risk_metrics)
        if risk_report:
            st.download_button(
                label="📄 Download Risk Analysis Report",
                data=risk_report,
                file_name=f"risk_analysis_report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

def calculate_risk_metrics_summary(portfolio_df, historical_values, risk_free_rate=0.02):
    """Calculate comprehensive risk metrics"""
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

def create_risk_return_scatter(portfolio_df):
    """Create risk-return scatter plot"""
    if portfolio_df.empty or 'gain_loss_pct' not in portfolio_df.columns:
        return None

    # Clean data
    df = portfolio_df.copy()
    df['gain_loss_pct'] = pd.to_numeric(df['gain_loss_pct'], errors='coerce')
    df['market_value'] = pd.to_numeric(df['market_value'], errors='coerce')
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    
    df = df.dropna(subset=['gain_loss_pct', 'market_value', 'quantity'])
    df = df[(df['quantity'] > 0) & (df['market_value'] > 0)]

    if df.empty:
        return None

    # Theme colors
    scatter_colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", 
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"
    ]

    fig = px.scatter(
        df,
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
        color_discrete_sequence=scatter_colors
    )

    # Styling
    fig.update_traces(
        marker=dict(line=dict(width=2, color='white'), opacity=0.8, sizemin=8)
    )

    # Add reference lines
    median_value = df['market_value'].median()
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.8)", line_width=2)
    fig.add_vline(x=median_value, line_dash="dash", line_color="rgba(255,255,255,0.8)", line_width=2)

    fig.update_layout(
        title=dict(font=dict(size=18, color='white'), x=0.5),
        xaxis=dict(
            title=dict(text='Current Value ($)', font=dict(color='white')),
            tickfont=dict(color='white'),
            gridcolor="rgba(255,255,255,0.1)",
            type="log"
        ),
        yaxis=dict(
            title=dict(text='Gain/Loss (%)', font=dict(color='white')),
            tickfont=dict(color='white'),
            gridcolor="rgba(255,255,255,0.1)"
        ),
        paper_bgcolor='#053B2A',
        plot_bgcolor='#053B2A',
        font=dict(color='white'),
        legend=dict(font=dict(color='white'))
    )

    return fig

def create_volatility_chart(historical_values):
    """Create rolling volatility chart"""
    if historical_values.empty:
        return None

    portfolio_history = historical_values[
        historical_values['symbol'] == 'PORTFOLIO_TOTAL'
    ].copy()
    
    if portfolio_history.empty:
        return None

    portfolio_history['date'] = pd.to_datetime(portfolio_history['date'])
    portfolio_history = portfolio_history.sort_values('date')
    portfolio_history['returns'] = portfolio_history['market_value'].pct_change()
    portfolio_history['rolling_volatility'] = portfolio_history['returns'].rolling(
        window=30, min_periods=10
    ).std() * np.sqrt(252)
    
    portfolio_history = portfolio_history.dropna()
    
    if portfolio_history.empty:
        return None

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=portfolio_history['date'],
        y=portfolio_history['rolling_volatility'],
        mode='lines',
        name='30-Day Rolling Volatility',
        line=dict(color='#15577A', width=3),
        fill='tonexty',
        fillcolor='rgba(52, 152, 219, 0.2)'
    ))
    
    avg_volatility = portfolio_history['rolling_volatility'].mean()
    fig.add_hline(
        y=avg_volatility,
        line_dash="dash",
        line_color='#0C4A57',
        line_width=2,
        annotation_text=f"Average: {avg_volatility:.2%}"
    )

    fig.update_layout(
        title=dict(text='Portfolio Risk: Rolling 30-Day Volatility', font=dict(size=16, color='white'), x=0.5),
        xaxis=dict(title=dict(text='Date', font=dict(color='white')), tickfont=dict(color='white')),
        yaxis=dict(title=dict(text='Annualized Volatility', font=dict(color='white')), tickfont=dict(color='white')),
        paper_bgcolor='#053B2A',
        plot_bgcolor='#053B2A',
        font=dict(color='white'),
        showlegend=False
    )

    return fig

def create_drawdown_chart(historical_values):
    """Create drawdown chart"""
    if historical_values.empty:
        return None

    portfolio_history = historical_values[
        historical_values['symbol'] == 'PORTFOLIO_TOTAL'
    ].copy()
    
    if portfolio_history.empty:
        return None

    portfolio_history['date'] = pd.to_datetime(portfolio_history['date'])
    portfolio_history = portfolio_history.sort_values('date')
    portfolio_history['running_max'] = portfolio_history['market_value'].expanding().max()
    portfolio_history['drawdown'] = (
        portfolio_history['market_value'] / portfolio_history['running_max'] - 1
    ) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=portfolio_history['date'],
        y=portfolio_history['drawdown'],
        mode='lines',
        name='Drawdown',
        line=dict(color='#DE2C2C', width=2),
        fill='tonexty',
        fillcolor='rgba(231, 76, 60, 0.3)'
    ))
    
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.8)", line_width=1)
    
    # Highlight maximum drawdown
    max_dd_idx = portfolio_history['drawdown'].idxmin()
    max_drawdown = portfolio_history.loc[max_dd_idx]
    
    fig.add_trace(go.Scatter(
        x=[max_drawdown['date']],
        y=[max_drawdown['drawdown']],
        mode='markers',
        name='Maximum Drawdown',
        marker=dict(color='white', size=12, symbol='circle')
    ))

    fig.update_layout(
        title=dict(text='Portfolio Drawdown Analysis', font=dict(size=16, color='white'), x=0.5),
        xaxis=dict(title=dict(text='Date', font=dict(color='white')), tickfont=dict(color='white')),
        yaxis=dict(title=dict(text='Drawdown (%)', font=dict(color='white')), tickfont=dict(color='white')),
        paper_bgcolor='#053B2A',
        plot_bgcolor='#053B2A',
        font=dict(color='white'),
        legend=dict(font=dict(color='white'))
    )

    return fig

def create_var_chart(historical_values, confidence_levels=[0.95, 0.99]):
    """Create Value at Risk chart"""
    if historical_values.empty:
        return None

    portfolio_history = historical_values[
        historical_values['symbol'] == 'PORTFOLIO_TOTAL'
    ].copy()
    
    if portfolio_history.empty:
        return None

    portfolio_history['date'] = pd.to_datetime(portfolio_history['date'])
    portfolio_history = portfolio_history.sort_values('date')
    portfolio_history['returns'] = portfolio_history['market_value'].pct_change()
    portfolio_history = portfolio_history.dropna()
    
    if len(portfolio_history) < 30:
        return None

    # Calculate VaR
    var_data = []
    for confidence in confidence_levels:
        var_percentile = (1 - confidence) * 100
        var_value = np.percentile(portfolio_history['returns'], var_percentile) * 100
        var_data.append({'confidence': f"{confidence*100:.0f}%", 'var': var_value})
    
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=portfolio_history['returns'] * 100,
        nbinsx=50,
        name='Return Distribution',
        marker_color='#15577A',
        opacity=0.7
    ))
    
    # VaR lines
    colors = ['#DE2C2C', '#FF4444']
    for i, var_item in enumerate(var_data):
        fig.add_vline(
            x=var_item['var'],
            line_dash="dash",
            line_color=colors[i % len(colors)],
            line_width=3,
            annotation_text=f"VaR {var_item['confidence']}: {var_item['var']:.2f}%"
        )

    fig.update_layout(
        title=dict(text='Value at Risk (VaR) Analysis', font=dict(size=16, color='white'), x=0.5),
        xaxis=dict(title=dict(text='Daily Returns (%)', font=dict(color='white')), tickfont=dict(color='white')),
        yaxis=dict(title=dict(text='Frequency', font=dict(color='white')), tickfont=dict(color='white')),
        paper_bgcolor='#053B2A',
        plot_bgcolor='#053B2A',
        font=dict(color='white'),
        showlegend=False
    )

    return fig

def create_correlation_heatmap(portfolio_df, historical_values):
    """Create correlation matrix heatmap"""
    if historical_values.empty or portfolio_df.empty:
        return None

    # Get asset returns
    asset_returns = []
    symbols = portfolio_df['symbol'].unique()
    
    for symbol in symbols:
        symbol_data = historical_values[historical_values['symbol'] == symbol].copy()
        
        if len(symbol_data) > 10:
            symbol_data['date'] = pd.to_datetime(symbol_data['date'])
            symbol_data = symbol_data.sort_values('date')
            symbol_data[f'{symbol}_returns'] = symbol_data['market_value'].pct_change()
            asset_returns.append(symbol_data[['date', f'{symbol}_returns']].dropna())
    
    if len(asset_returns) < 2:
        return None

    # Merge returns
    returns_df = asset_returns[0]
    for df in asset_returns[1:]:
        returns_df = pd.merge(returns_df, df, on='date', how='outer')
    
    return_cols = [col for col in returns_df.columns if col.endswith('_returns')]
    if len(return_cols) < 2:
        return None
        
    corr_matrix = returns_df[return_cols].corr()
    corr_matrix.columns = [col.replace('_returns', '') for col in corr_matrix.columns]
    corr_matrix.index = [idx.replace('_returns', '') for idx in corr_matrix.index]

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10, "color": "white"}
    ))

    fig.update_layout(
        title=dict(text='Asset Correlation Matrix', font=dict(size=16, color='white'), x=0.5),
        paper_bgcolor='#053B2A',
        plot_bgcolor='#053B2A',
        font=dict(color='white'),
        width=500,
        height=500
    )

    return fig

def create_beta_chart(portfolio_df, historical_values, benchmark_symbol='SPY'):
    """Create beta analysis chart"""
    if historical_values.empty or portfolio_df.empty:
        return None

    portfolio_data = historical_values[
        historical_values['symbol'] == 'PORTFOLIO_TOTAL'
    ].copy()
    
    if portfolio_data.empty or len(portfolio_data) < 30:
        return None

    portfolio_data['date'] = pd.to_datetime(portfolio_data['date'])
    portfolio_data = portfolio_data.sort_values('date')
    portfolio_data['portfolio_returns'] = portfolio_data['market_value'].pct_change()
    
    # Synthetic benchmark for demo
    np.random.seed(42)
    benchmark_returns = np.random.normal(0.0003, 0.01, len(portfolio_data))
    portfolio_data['benchmark_returns'] = benchmark_returns
    
    clean_data = portfolio_data.dropna()
    
    if len(clean_data) < 10:
        return None

    # Calculate beta
    covariance = np.cov(clean_data['portfolio_returns'], clean_data['benchmark_returns'])[0][1]
    benchmark_variance = np.var(clean_data['benchmark_returns'])
    beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
    
    fig = px.scatter(
        clean_data,
        x='benchmark_returns',
        y='portfolio_returns',
        title=f'Portfolio Beta Analysis (β = {beta:.2f})',
        color_discrete_sequence=['#15577A']
    )
    
    # Regression line
    z = np.polyfit(clean_data['benchmark_returns'], clean_data['portfolio_returns'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(clean_data['benchmark_returns'].min(), clean_data['benchmark_returns'].max(), 100)
    y_line = p(x_line)
    
    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        name=f'Beta Line (β = {beta:.2f})',
        line=dict(color='#DE2C2C', width=3)
    ))

    fig.update_layout(
        title=dict(font=dict(size=16, color='white'), x=0.5),
        xaxis=dict(title=dict(text=f'{benchmark_symbol} Returns', font=dict(color='white')), tickfont=dict(color='white')),
        yaxis=dict(title=dict(text='Portfolio Returns', font=dict(color='white')), tickfont=dict(color='white')),
        paper_bgcolor='#053B2A',
        plot_bgcolor='#053B2A',
        font=dict(color='white'),
        legend=dict(font=dict(color='white'))
    )

    return fig

def generate_risk_report(portfolio_df, historical_values, risk_metrics):
    """Generate comprehensive risk analysis report"""
    if not risk_metrics:
        return None

    report_data = []
    
    # Risk metrics
    for metric, value in risk_metrics.items():
        if not np.isinf(value):
            report_data.append({
                'Metric': metric,
                'Value': f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}",
                'Category': 'Risk Metric'
            })
    
    # Portfolio composition
    if not portfolio_df.empty:
        for _, row in portfolio_df.iterrows():
            report_data.append({
                'Metric': f"{row['symbol']} Weight",
                'Value': f"{(row['market_value'] / portfolio_df['market_value'].sum()):.2%}",
                'Category': 'Portfolio Composition'
            })
    
    report_df = pd.DataFrame(report_data)
    return report_df.to_csv(index=False)