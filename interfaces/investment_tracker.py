import streamlit as st
import pandas as pd
from datetime import datetime
from core.robinhood_parser import parse_investment_documents
from utils.equity_vis import render_equity_visualizations
from core.data_fetcher import DataFetcher


def show_investment_dashboard():
    """Main UI for investment tracking and analysis"""
    st.header("📈 Investment Portfolio Analysis")

    # Instruction + Disclaimer Block
    st.markdown("""
    <div style="background-color: #f9f9f9; padding: 1.5rem; border-left: 5px solid #1f77b4; border-radius: 8px; margin-bottom: 2rem;">
        <h4>📌 How to Use This Tool</h4>
        <ul style="text-align: left; font-size: 0.95rem;">
            <li><strong>Get Your Reports:</strong> Download your <strong>transaction history</strong> as CSV or PDF from your brokerage (e.g., Robinhood → Account (Person icon) → Menu → Reports and statement → Reports → Generate new report → Download).</li>
            <li><strong>CSV files are strongly recommended</strong> for best parsing accuracy and detail. PDF support is available but may be limited by formatting inconsistencies.</li>
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
            <strong>Disclaimer:</strong> This tool is for informational purposes only. While care has been taken in its design, it may occasionally produce incorrect results depending on data format. The developer is <strong>not responsible</strong> for any financial decisions or actions taken based on this tool.
        </p>
    </div>
    """, unsafe_allow_html=True)

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

        if hasattr(tracker, "portfolio") and not tracker.portfolio.empty:
            # Add button to fetch historical data
            if st.button("📊 Load Historical Data for Visualizations", type="secondary"):
                with st.spinner("Fetching historical data. This may take a few minutes..."):
                    # Get symbols from portfolio
                    symbols = tracker.portfolio['symbol'].unique().tolist()
                    
                    # Fetch historical data
                    fetcher = DataFetcher()
                    _, _, historical_values = fetcher.fetch_historical_data(symbols, "1y")
                    
                    # Store in tracker
                    tracker.historical_values = historical_values
                    st.session_state.tracker = tracker
                    st.success("Historical data loaded!")
            
        if tracker and not tracker.portfolio.empty:
            st.session_state.tracker = tracker
            # Portfolio summary metrics
            st.subheader("📊 Portfolio Summary")
            summary = tracker.calculate_portfolio_summary()
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
                st.dataframe(tracker.get_top_performers(), use_container_width=True)
            
            with col2:
                st.subheader("📉 Bottom Performers")
                st.dataframe(tracker.get_bottom_performers(), use_container_width=True)
            
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
