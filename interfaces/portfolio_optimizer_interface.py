import streamlit as st
import pandas as pd
from datetime import datetime
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
            
            # Run portfolio optimization
            try:
                with st.spinner("🔄 Running portfolio optimization..."):
                    run_portfolio_optimization(tracker)
            except Exception as e:
                st.error(f"❌ Error running portfolio optimization: {str(e)}")
                
                # Show detailed error in expander
                with st.expander("🔍 Error Details"):
                    import traceback
                    st.code(traceback.format_exc())
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


# Additional utility functions
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
        if not symbol or len(symbol.strip()) == 0:
            invalid_symbols.append("Empty symbol")
        elif len(symbol) > 10:  # Most ticker symbols are <= 5 characters
            invalid_symbols.append(symbol)
    
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
                    preview_df['market_value'] = preview_df['market_value'].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A")
                
                if 'gain_loss_pct' in preview_df.columns:
                    preview_df['gain_loss_pct'] = preview_df['gain_loss_pct'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
                
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


# Example usage
if __name__ == "__main__":
    st.set_page_config(
        page_title="Portfolio Optimization",
        page_icon="🎯",
        layout="wide"
    )
    
    show_portfolio_optimization_tab()