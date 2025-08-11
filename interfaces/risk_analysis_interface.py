import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from core.data_fetcher import DataFetcher
from utils.risk_analysis import (
    risk_return_scatter,
    portfolio_volatility_chart,
    drawdown_chart,
    value_at_risk_chart,
    correlation_heatmap,
    beta_analysis_chart,
    risk_metrics_summary,
    individual_asset_risk_return_scatter,
    portfolio_return_vs_drawdown_chart,
    rolling_sharpe_ratio_chart
)
from utils.risk_beta_analysis import nonlinear_beta_analysis_chart

def show_risk_analysis_dashboard():
    """Main UI for portfolio risk analysis"""
    st.header("⚠️ Portfolio Risk Analysis Dashboard")

    # Check if portfolio data exists in session state
    if 'tracker' not in st.session_state or not hasattr(st.session_state.tracker, 'portfolio'):
        st.markdown("""
        <div class="warning-gradient">
            <h4>📊 No Portfolio Data Available</h4>
            <p>Please upload your portfolio data in the <strong>Investment Portfolio</strong> tab first, then run the optimization to load historical data to access risk analysis features.</p>
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
        <h4> Risk Analysis Overview</h4>
        <p style="text-align: left; font-size: 0.95rem; margin-bottom: 1rem;">
                (Make sure to run the optimization in the Portfolio Optimization tab) <br>
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

    # Calculate and display risk metrics summary using utils function
    risk_metrics = risk_metrics_summary(portfolio_df, historical_values)
    
    if risk_metrics:
        st.subheader(" Key Risk Metrics")
        
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

    # Risk-Return Scatter Plot using utils function
    st.subheader(" Risk-Return Analysis")
    risk_return_fig = risk_return_scatter(portfolio_df)
    if risk_return_fig:
        st.plotly_chart(risk_return_fig, use_container_width=True)

    # Individual Asset Analysis and Return vs Drawdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Individual Asset Risk vs Return")
        individual_risk_fig = individual_asset_risk_return_scatter(portfolio_df, historical_values)
        if individual_risk_fig:
            st.plotly_chart(individual_risk_fig, use_container_width=True)
        else:
            st.info("Insufficient data for individual asset risk-return analysis.")
    
    with col2:
        st.subheader(" Portfolio Return vs Drawdown")
        return_drawdown_fig = portfolio_return_vs_drawdown_chart(historical_values)
        if return_drawdown_fig:
            st.plotly_chart(return_drawdown_fig, use_container_width=True)
        else:
            st.info("Insufficient data for return vs drawdown analysis.")

    # Rolling Sharpe Ratio using utils function
    st.subheader(" Rolling Risk-Adjusted Performance")
    rolling_sharpe_fig = rolling_sharpe_ratio_chart(historical_values)
    if rolling_sharpe_fig:
        st.plotly_chart(rolling_sharpe_fig, use_container_width=True)
    else:
        st.info("Insufficient data for rolling Sharpe ratio analysis.")

    # Volatility and Drawdown Analysis using utils functions
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader(" Portfolio Volatility Over Time")
        volatility_fig = portfolio_volatility_chart(historical_values)
        if volatility_fig:
            st.plotly_chart(volatility_fig, use_container_width=True)
        else:
            st.info("Insufficient historical data for volatility analysis.")
    
    with col4:
        st.subheader(" Drawdown Analysis")
        drawdown_fig = drawdown_chart(historical_values)
        if drawdown_fig:
            st.plotly_chart(drawdown_fig, use_container_width=True)
        else:
            st.info("Insufficient historical data for drawdown analysis.")

    # Value at Risk Analysis using utils function
    st.subheader("⚡ Value at Risk (VaR) Distribution")
    var_fig = value_at_risk_chart(historical_values)
    if var_fig:
        st.plotly_chart(var_fig, use_container_width=True)
    else:
        st.info("Insufficient historical data for VaR analysis.")

    # Correlation and Beta Analysis using utils functions
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader("🔗 Asset Correlation Matrix")
        correlation_fig = correlation_heatmap(portfolio_df, historical_values)
        if correlation_fig:
            st.plotly_chart(correlation_fig, use_container_width=True)
        else:
            st.info("Insufficient data for correlation analysis.")
    
    with col6:
        st.subheader("📊 Portfolio Beta Analysis")
        beta_fig = beta_analysis_chart(portfolio_df, historical_values)
        if beta_fig:
            st.plotly_chart(beta_fig, use_container_width=True)
        else:
            st.info("Insufficient data for beta analysis.")
    st.subheader(" Nonlinear Beta Analysis")
    nonlinear_beta_fig = nonlinear_beta_analysis_chart(portfolio_df, historical_values)
    if nonlinear_beta_fig:
        st.plotly_chart(nonlinear_beta_fig, use_container_width=True)
    else:
        st.info("Insufficient data for nonlinear beta analysis.")

    # Risk Analysis Export
    st.subheader(" Export Risk Analysis")
    if st.button(" Generate Risk Report", type="secondary"):
        risk_report = generate_risk_report(portfolio_df, historical_values, risk_metrics)
        if risk_report:
            st.download_button(
                label=" Download Risk Analysis Report",
                data=risk_report,
                file_name=f"risk_analysis_report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.error("Could not generate report. Check if you have valid portfolio data and risk metrics.")

def generate_risk_report(portfolio_df, historical_values, risk_metrics):
    """Generate comprehensive risk analysis report"""
    report_data = []
    
    # Risk metrics
    if risk_metrics:
        for metric, value in risk_metrics.items():
            if not np.isinf(value):
                report_data.append({
                    'Metric': metric,
                    'Value': f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}",
                    'Category': 'Risk Metric'
                })
    
    # Portfolio composition - FILTER OUT PORTFOLIO_TOTAL
    valid_assets = portfolio_df[portfolio_df['symbol'] != 'PORTFOLIO_TOTAL']
    total_value = valid_assets['market_value'].sum()
    
    if not valid_assets.empty and total_value > 0:
        for _, row in valid_assets.iterrows():
            report_data.append({
                'Metric': f"{row['symbol']} Weight",
                'Value': f"{(row['market_value'] / total_value):.2%}",
                'Category': 'Portfolio Composition'
            })
    
    if not report_data:
        return None

    report_df = pd.DataFrame(report_data)
    return report_df.to_csv(index=False)