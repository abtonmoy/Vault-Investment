import streamlit as st
import pandas as pd
from datetime import datetime
from core.robinhood_parser import parse_investment_documents
from utils.equity_vis import render_equity_visualizations
from core.portfolio_optimizer import run_portfolio_optimization  # Import our new optimizer

def show_investment_dashboard():
    """Main UI for investment tracking and analysis with portfolio optimization"""
    st.header("📈 Investment Portfolio Analysis & Optimization")

    # Instruction + Disclaimer Block
    st.markdown("""
    <div style="background-color: #f9f9f9; padding: 1.5rem; border-left: 5px solid #1f77b4; border-radius: 8px; margin-bottom: 2rem;">
        <h4>📌 How to Use This Tool</h4>
        <ul style="text-align: left; font-size: 0.95rem;">
            <li><strong>Get Your Reports:</strong> Download your <strong>transaction history</strong> as CSV or PDF from your brokerage (e.g., Robinhood → Account (Person icon) → Menu → Reports and statement → Reports → Generate new report → Download).</li>
            <li><strong>CSV files are strongly recommended</strong> for best parsing accuracy and detail. PDF support is available but may be limited by formatting inconsistencies.</li>
            <li><strong>Portfolio Optimization:</strong> After uploading your portfolio data, use the new optimization feature to find optimal asset allocations based on historical performance and modern portfolio theory.</li>
            <li><strong>Uploading multiple files?</strong>
                <ul>
                    <li>You <strong>can upload overlapping date ranges</strong> (e.g., Jan–June + Jan–July), but make sure the <strong>latest report is uploaded last</strong>.</li>
                    <li>This ensures the most recent data correctly overwrites any older overlapping entries.</li>
                    <li>We <strong>DO NOT</strong> store any data. After each session, all uploaded files and portfolio data are <strong>cleared automatically</strong>. You will need to re-upload them next time.</li>
                    <li>Recommended to use in <strong>Light Mode</strong> for best viewing experience.</li>
                </ul>
            </li>
        </ul>
        <hr style="margin: 1.2rem 0;">
        <p style="color: #a00; font-size: 0.9rem;">
            <strong>Disclaimer:</strong> This tool is for informational purposes only. While care has been taken in its design, it may occasionally produce incorrect results depending on data format. Portfolio optimization is based on historical data and may not predict future performance. The developer is <strong>not responsible</strong> for any financial decisions or actions taken based on this tool.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["📊 Portfolio Analysis", "🎯 Portfolio Optimization"])
    
    with tab1:
        # Original portfolio analysis functionality
        st.subheader("Upload Portfolio Data")
        
        # Document type selection
        doc_type = st.selectbox("Select Document Type", ["CSV", "PDF"])
        
        # Brokerage selection
        brokerage = st.selectbox(
            "Select Brokerage",
            ["Robinhood", "Fidelity", "Charles Schwab", "Vanguard", "Generic"]
        )
        
        # File uploader
        uploaded_files = st.file_uploader(
            f"Upload {brokerage} {doc_type} Reports",
            type=[doc_type.lower()],
            accept_multiple_files=True,
            key="investment_files",
            help="Upload CSV or PDF files from your brokerage",
            label_visibility="visible" 
        )
        
        if uploaded_files:
            # Parse investment documents
            tracker = parse_investment_documents(uploaded_files, doc_type, brokerage)
            
            if tracker and not tracker.portfolio.empty:
                # Store tracker in session state for optimization tab
                st.session_state.tracker = tracker
                
                # Portfolio summary metrics
                st.subheader("📊 Portfolio Summary")
                summary = tracker.calculate_portfolio_summary()
                
                if summary:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Value", f"${summary['total_value']:,.2f}")
                    col2.metric("Total Gain/Loss", 
                               f"${summary['total_gain_loss']:,.2f}", 
                               f"{summary['gain_loss_pct']:.2f}%")
                    col3.metric("Positions", summary['num_positions'])
                    col4.metric("Brokerages", summary['num_brokerages'])
                
                # Asset class breakdown
                st.subheader("📦 Asset Class Breakdown")
                asset_class_df = tracker.get_asset_class_breakdown()
                if asset_class_df is not None:
                    st.dataframe(asset_class_df, use_container_width=True)
                
                # Top and bottom performers
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🚀 Top Performers")
                    top_performers = tracker.get_top_performers()
                    if not top_performers.empty:
                        st.dataframe(top_performers[['symbol', 'market_value', 'gain_loss_pct']], use_container_width=True)
                
                with col2:
                    st.subheader("📉 Bottom Performers")
                    bottom_performers = tracker.get_bottom_performers()
                    if not bottom_performers.empty:
                        st.dataframe(bottom_performers[['symbol', 'market_value', 'gain_loss_pct']], use_container_width=True)
                
                # Portfolio details
                st.subheader("🔍 Full Portfolio Details")
                st.dataframe(tracker.portfolio, use_container_width=True)
                
                # Export button
                st.download_button(
                    label="💾 Export Portfolio Data",
                    data=tracker.portfolio.to_csv(index=False),
                    file_name=f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                # Show visualizations
                render_equity_visualizations(tracker)
            else:
                st.warning("No valid investment data found in the uploaded files.")
    
    with tab2:
        # Portfolio optimization functionality
        st.subheader("Portfolio Optimization")
        
        # Check if we have tracker data
        if 'tracker' in st.session_state and not st.session_state.tracker.portfolio.empty:
            tracker = st.session_state.tracker
            
            # Show current portfolio summary
            st.info(f"📊 Current Portfolio: {len(tracker.portfolio)} positions worth ${tracker.portfolio['market_value'].sum():,.2f}")
            
            # Add optimization information
            st.markdown("""
            ### About Portfolio Optimization
            
            This feature uses **Modern Portfolio Theory** to optimize your portfolio allocation based on historical data. The optimization process:
            
            - 📈 **Fetches historical price data** for all your holdings using Yahoo Finance
            - 📊 **Calculates risk-return metrics** including expected returns, volatility, and correlations
            - 🎯 **Optimizes allocations** using various strategies (Maximum Sharpe Ratio, Minimum Volatility, etc.)
            - 📋 **Provides recommendations** for rebalancing your portfolio
            
            **Note:** Optimization is based on historical data and assumes past performance patterns will continue. Always consider your risk tolerance and investment goals.
            """)
            
            # Run portfolio optimization
            try:
                run_portfolio_optimization(tracker)
            except Exception as e:
                st.error(f"Error running portfolio optimization: {str(e)}")
                
                # Show detailed error in expander
                with st.expander("Error Details"):
                    import traceback
                    st.code(traceback.format_exc())
        
        else:
            st.warning("""
            ⚠️ **No portfolio data available for optimization**
            
            Please upload your portfolio data in the **Portfolio Analysis** tab first, then return here to run optimization.
            
            ### Requirements for Optimization:
            - ✅ Valid portfolio data with ticker symbols
            - ✅ Internet connection (to fetch historical data)
            - ✅ At least 2-3 different assets in your portfolio
            """)
            
            # Option to upload files directly in this tab
            st.subheader("Quick Upload for Optimization")
            
            col1, col2 = st.columns(2)
            with col1:
                quick_doc_type = st.selectbox("Document Type", ["CSV", "PDF"], key="quick_doc_type")
            with col2:
                quick_brokerage = st.selectbox(
                    "Brokerage", 
                    ["Robinhood", "Fidelity", "Charles Schwab", "Vanguard", "Generic"],
                    key="quick_brokerage"
                )
            
            quick_files = st.file_uploader(
                f"Upload {quick_brokerage} {quick_doc_type} Reports",
                type=[quick_doc_type.lower()],
                accept_multiple_files=True,
                key="quick_investment_files",
                help="Upload files directly for optimization"
            )
            
            if quick_files:
                # Parse investment documents
                tracker = parse_investment_documents(quick_files, quick_doc_type, quick_brokerage)
                
                if tracker and not tracker.portfolio.empty:
                    # Store tracker in session state
                    st.session_state.tracker = tracker
                    st.success("✅ Portfolio data loaded successfully! You can now run optimization below.")
                    st.rerun()  # Refresh to show optimization interface


# Additional utility functions for optimization integration

def get_optimization_requirements():
    """Return requirements text for optimization"""
    return """
    ### Requirements for Portfolio Optimization:
    
    1. **Valid Portfolio Data**: Upload CSV/PDF files with your current holdings
    2. **Ticker Symbols**: Ensure your portfolio contains recognizable ticker symbols
    3. **Internet Connection**: Required to fetch historical price data from Yahoo Finance
    4. **Diversified Portfolio**: Works best with 3+ different assets
    5. **Historical Data**: Assets need sufficient trading history (typically 1+ years)
    
    ### Optimization Methods Available:
    
    - **Maximum Sharpe Ratio**: Optimizes risk-adjusted returns
    - **Minimum Volatility**: Minimizes portfolio risk
    - **Maximum Return**: Maximizes expected returns (higher risk)
    """

def show_optimization_tips():
    """Display tips for better optimization results"""
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


# Example usage and integration notes
if __name__ == "__main__":
    st.set_page_config(
        page_title="Investment Portfolio Analyzer & Optimizer",
        page_icon="📈",
        layout="wide"
    )
    
    show_investment_dashboard()