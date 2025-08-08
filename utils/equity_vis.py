import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import helpers
from utils.theme import THEME_COLORS, THEME_GRADIENTS

# Define a consistent color palette from theme
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

# Comprehensive crypto symbol list for better recognition
CRYPTO_SYMBOLS = {
    'BTC', 'ETH', 'DOGE', 'ADA', 'DOT', 'LINK', 'LTC', 'BCH', 'XLM', 'ETC',
    'UNI', 'AAVE', 'SUSHI', 'COMP', 'MKR', 'YFI', 'SNX', 'CRV', 'BAL', 'REN',
    'ZRX', 'KNC', 'LRC', 'OMG', 'STORJ', 'DNT', 'BAT', 'ZEC', 'XRP', 'TRX',
    'VET', 'THETA', 'FIL', 'EOS', 'ATOM', 'ALGO', 'XTZ', 'DASH', 'NEO', 'QTUM',
    'ICX', 'ZIL', 'ONT', 'IOST', 'KAVA', 'BAND', 'CRO', 'HOT', 'ENJ', 'MANA',
    'SAND', 'AXS', 'SHIB', 'MATIC', 'SOL', 'AVAX', 'LUNA', 'NEAR', 'FTT', 'CRV'
}

# ETF symbols for better classification
ETF_SYMBOLS = {
    'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'AGG', 'BND', 'LQD',
    'HYG', 'JNK', 'TLT', 'IEF', 'SHY', 'TIP', 'VTEB', 'MUB', 'PFF', 'SCHD',
    'VIG', 'VYM', 'HDV', 'NOBL', 'DGRO', 'VUG', 'VTV', 'VXUS', 'IEFA', 'IEMG',
    'GLD', 'SLV', 'IAU', 'PDBC', 'DBA', 'USO', 'UNG', 'GDXJ', 'GDX', 'SIL',
    'XLE', 'XLF', 'XLK', 'XLV', 'XLI', 'XLP', 'XLU', 'XLB', 'XLY', 'XLRE',
    'VGT', 'VHT', 'VFH', 'VIS', 'VCR', 'VDC', 'VPU', 'VAW', 'VNQ', 'RWR',
    'IYR', 'SCHB', 'SCHA', 'SCHF', 'SCHE', 'SCHM', 'SCHO', 'SCHP', 'SCHQ',
    'SPDW', 'SPEM', 'SPTM', 'SPSM', 'SPMD', 'SPMB', 'SPAB', 'SPBO', 'SPTS'
}

def enhanced_asset_classification(row):
    """Enhanced asset classification with better crypto and ETF recognition"""
    symbol = str(row['symbol']).upper().strip()
    
    # Check for asset_class column first
    if 'asset_class' in row and pd.notna(row['asset_class']):
        return row['asset_class']
    elif 'asset_type' in row and pd.notna(row['asset_type']):
        return row['asset_type']
    
    # Enhanced classification logic
    if symbol in CRYPTO_SYMBOLS:
        return 'Crypto'
    elif symbol in ETF_SYMBOLS:
        return 'ETF'
    elif any(keyword in symbol for keyword in ['BTC', 'ETH', 'DOGE', 'CRYPTO', 'COIN']):
        return 'Crypto'
    elif any(keyword in symbol for keyword in ['BOND', 'TREASURY', 'GOVT', 'TLT', 'AGG', 'BND']):
        return 'Bonds'
    elif symbol in ['CASH', 'USD', 'MONEY'] or 'CASH' in symbol:
        return 'Cash'
    elif any(keyword in symbol for keyword in ['REIT', 'VNQ', 'RWR', 'IYR']):
        return 'Real Estate'
    elif any(keyword in symbol for keyword in ['GLD', 'SLV', 'IAU', 'GOLD', 'SILVER']):
        return 'Commodities'
    elif len(symbol) <= 5 and symbol.isalpha():  # Most stocks are 1-5 letter symbols
        return 'Stock'
    else:
        return 'Other'

def portfolio_allocation_chart(portfolio_df):
    """Create a modern pie chart of portfolio allocation"""
    if portfolio_df.empty or 'market_value' not in portfolio_df.columns:
        return None
    
    # Group by symbol for allocation
    allocation = portfolio_df.groupby('symbol')['market_value'].sum().reset_index()
    
    # Calculate percentages to determine which labels to show
    allocation['percentage'] = allocation['market_value'] / allocation['market_value'].sum() * 100
    
    fig = px.pie(
        allocation,
        names='symbol',
        values='market_value',
        title='Portfolio Allocation by Holding',
        hole=0.5,
        color_discrete_sequence=CHART_COLORS
    )
    
    # Create custom text - only show labels for holdings > 3%
    text_info = []
    for i, row in allocation.iterrows():
        if row['percentage'] > 3:
            text_info.append(f"{row['symbol']}<br>{row['percentage']:.1f}%")
        else:
            text_info.append("")  # Empty string for small slices
    
    fig.update_traces(
        textposition='outside',
        text=text_info,
        textinfo='text',
        textfont=dict(size=11, color='white'),
        hovertemplate="<b>%{label}</b><br>Value: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>",
        marker=dict(line=dict(color='white', width=2))
    )
    
    fig.update_layout(
        paper_bgcolor=THEME_COLORS["dark_green"],
        plot_bgcolor=THEME_COLORS["dark_green"],
        font=dict(color='white', size=12),
        title=dict(
            font=dict(size=18, color='white'),
            x=0.5
        ),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            font=dict(color='white')
        )
    )
    
    return fig

def asset_flow_sankey(portfolio_df):
    """Create a Sankey diagram showing asset flow with enhanced classification"""
    if portfolio_df.empty or 'market_value' not in portfolio_df.columns:
        return None
    
    # Create a copy to avoid modifying original dataframe
    portfolio_summary = portfolio_df.copy()
    
    # Group by symbol first to get totals
    portfolio_summary = portfolio_summary.groupby('symbol').agg({
        'market_value': 'sum',
        **({col: 'first' for col in portfolio_df.columns if col not in ['symbol', 'market_value']})
    }).reset_index()
    
    # Apply enhanced asset classification
    portfolio_summary['asset_type'] = portfolio_summary.apply(enhanced_asset_classification, axis=1)
    
    # Calculate total portfolio value
    total_value = portfolio_summary['market_value'].sum()
    
    # Get unique asset types and symbols
    asset_types = sorted(portfolio_summary['asset_type'].unique())
    symbols = sorted(portfolio_summary['symbol'].unique())
    
    # Create node labels: Total Assets → Asset Types → Individual Symbols
    all_nodes = ['Total Portfolio'] + asset_types + symbols
    
    # Create source and target indices, and values
    sources = []
    targets = []
    values = []
    
    # First level: Total Portfolio → Asset Types
    asset_type_totals = portfolio_summary.groupby('asset_type')['market_value'].sum()
    for asset_type, total_val in asset_type_totals.items():
        total_assets_idx = all_nodes.index('Total Portfolio')
        asset_type_idx = all_nodes.index(asset_type)
        
        sources.append(total_assets_idx)
        targets.append(asset_type_idx)
        values.append(total_val)
    
    # Second level: Asset Types → Individual Holdings
    for _, row in portfolio_summary.iterrows():
        asset_type_idx = all_nodes.index(row['asset_type'])
        symbol_idx = all_nodes.index(row['symbol'])
        
        sources.append(asset_type_idx)
        targets.append(symbol_idx)
        values.append(row['market_value'])
    
    # Enhanced color mapping with distinct colors for each asset class
    asset_class_colors = {
        'Stock': "#3498DB",      # Bright blue
        'ETF': "#E74C3C",        # Bright red
        'Mutual Fund': "#2ECC71", # Bright green  
        'Crypto': "#F39C12",     # Bright orange
        'Bonds': "#9B59B6",      # Purple
        'Cash': "#1ABC9C",       # Teal
        'Real Estate': "#E67E22", # Dark orange
        'Commodities': "#34495E", # Dark gray
        'Options': "#FF6B9D",    # Pink
        'Forex': "#4ECDC4",      # Aqua
        'Other': "#95A5A6"       # Light gray
    }
    
    node_colors = []
    for node in all_nodes:
        if node == 'Total Portfolio':
            node_colors.append('#FFFFFF')  # White for total portfolio
        elif node in asset_types:
            node_colors.append(asset_class_colors.get(node, '#95A5A6'))
        else:
            # Symbol colors - use scatter colors for individual holdings
            symbol_idx = len([n for n in all_nodes[:all_nodes.index(node)] if n not in asset_types and n != 'Total Portfolio'])
            node_colors.append(SCATTER_COLORS[symbol_idx % len(SCATTER_COLORS)])
    
    # Create link colors (match source nodes but with transparency)
    link_colors = []
    for source_idx in sources:
        base_color = node_colors[source_idx]
        # Convert hex to rgba with transparency
        if base_color.startswith('#'):
            hex_color = base_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            link_colors.append(f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.6)")
        else:
            link_colors.append(base_color)
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="rgba(255,255,255,0.8)", width=2),
            label=all_nodes,
            color=node_colors,
            hovertemplate="<b>%{label}</b><br>Total Value: $%{value:,.2f}<extra></extra>"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            hovertemplate="<b>%{source.label}</b> → <b>%{target.label}</b><br>Value: $%{value:,.2f}<extra></extra>"
        )
    )])
    
    fig.update_layout(
        title=dict(
            text="Asset Flow: Total Portfolio → Asset Classes → Holdings",
            font=dict(size=18, color='white'),
            x=0.5
        ),
        font=dict(size=12, color='white'),
        paper_bgcolor=THEME_COLORS["dark_green"],
        plot_bgcolor=THEME_COLORS["dark_green"],
        height=600
    )
    
    return fig

def asset_class_breakdown_chart(portfolio_df):
    """Create a detailed breakdown chart by asset class"""
    if portfolio_df.empty or 'market_value' not in portfolio_df.columns:
        return None
    
    # Create a copy and apply enhanced classification
    portfolio_copy = portfolio_df.copy()
    portfolio_copy['asset_type'] = portfolio_copy.apply(enhanced_asset_classification, axis=1)
    
    # Group by asset type
    asset_breakdown = portfolio_copy.groupby('asset_type')['market_value'].sum().reset_index()
    asset_breakdown['percentage'] = asset_breakdown['market_value'] / asset_breakdown['market_value'].sum() * 100
    
    # Sort by value
    asset_breakdown = asset_breakdown.sort_values('market_value', ascending=True)
    
    fig = px.bar(
        asset_breakdown,
        x='market_value',
        y='asset_type',
        orientation='h',
        title='Portfolio Breakdown by Asset Class',
        color='asset_type',
        color_discrete_map={
            'Stock': "#3498DB",
            'ETF': "#E74C3C", 
            'Crypto': "#F39C12",
            'Bonds': "#9B59B6",
            'Cash': "#1ABC9C",
            'Real Estate': "#E67E22",
            'Commodities': "#34495E",
            'Other': "#95A5A6"
        }
    )
    
    # Add percentage labels
    for i, row in asset_breakdown.iterrows():
        fig.add_annotation(
            x=row['market_value'],
            y=row['asset_type'],
            text=f"${row['market_value']:,.0f} ({row['percentage']:.1f}%)",
            showarrow=False,
            font=dict(color='white', size=11),
            xshift=10
        )
    
    fig.update_layout(
        title=dict(
            font=dict(size=18, color='white'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text='Market Value ($)', font=dict(color='white')),
            tickfont=dict(color='white'),
            gridcolor="rgba(255,255,255,0.1)"
        ),
        yaxis=dict(
            title=dict(text='Asset Class', font=dict(color='white')),
            tickfont=dict(color='white')
        ),
        paper_bgcolor=THEME_COLORS["dark_green"],
        plot_bgcolor=THEME_COLORS["dark_green"],
        font=dict(color='white'),
        showlegend=False
    )
    
    return fig

def crypto_holdings_chart(portfolio_df):
    """Create a specific chart for crypto holdings if any exist"""
    if portfolio_df.empty or 'market_value' not in portfolio_df.columns:
        return None
    
    # Create a copy and apply enhanced classification  
    portfolio_copy = portfolio_df.copy()
    portfolio_copy['asset_type'] = portfolio_copy.apply(enhanced_asset_classification, axis=1)
    
    # Filter for crypto holdings
    crypto_holdings = portfolio_copy[portfolio_copy['asset_type'] == 'Crypto'].copy()
    
    if crypto_holdings.empty:
        return None
    
    # Sort by market value
    crypto_holdings = crypto_holdings.sort_values('market_value', ascending=True)
    
    fig = px.bar(
        crypto_holdings,
        x='market_value',
        y='symbol',
        orientation='h',
        title='Cryptocurrency Holdings',
        color='symbol',
        color_discrete_sequence=SCATTER_COLORS
    )
    
    # Add value labels
    for i, row in crypto_holdings.iterrows():
        fig.add_annotation(
            x=row['market_value'],
            y=row['symbol'],
            text=f"${row['market_value']:,.2f}",
            showarrow=False,
            font=dict(color='white', size=11),
            xshift=10
        )
    
    fig.update_layout(
        title=dict(
            font=dict(size=18, color='white'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text='Market Value ($)', font=dict(color='white')),
            tickfont=dict(color='white'),
            gridcolor="rgba(255,255,255,0.1)"
        ),
        yaxis=dict(
            title=dict(text='Cryptocurrency', font=dict(color='white')),
            tickfont=dict(color='white')
        ),
        paper_bgcolor=THEME_COLORS["dark_green"],
        plot_bgcolor=THEME_COLORS["dark_green"],
        font=dict(color='white'),
        showlegend=False
    )
    
    return fig

def gain_loss_waterfall(portfolio_df):
    """Create modern waterfall chart of gains and losses"""
    if portfolio_df.empty or 'gain_loss' not in portfolio_df.columns:
        return None
    
    # Sort by gain/loss
    sorted_df = portfolio_df.sort_values('gain_loss', ascending=False)
    
    fig = go.Figure(go.Waterfall(
        name="Performance",
        orientation="v",
        measure=["relative"] * len(sorted_df),
        x=sorted_df['symbol'],
        textposition="outside",
        text=[f"${x:,.0f}" for x in sorted_df['gain_loss']],
        textfont=dict(color='white', size=11),
        y=sorted_df['gain_loss'],
        connector=dict(line=dict(color="rgba(255,255,255,0.3)", width=1)),
        increasing=dict(marker=dict(color=THEME_COLORS["lapis_lazuli"])),
        decreasing=dict(marker=dict(color=THEME_COLORS["rojo"])),
        hovertemplate="<b>%{x}</b><br>Amount: $%{y:,.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(
            text="Gain/Loss by Holding",
            font=dict(size=18, color='white'),
            x=0.5
        ),
        showlegend=False,
        yaxis=dict(
            title=dict(text="Amount ($)", font=dict(color='white')),
            tickfont=dict(color='white'),
            gridcolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.3)"
        ),
        xaxis=dict(
            title=dict(text="Holding", font=dict(color='white')),
            tickfont=dict(color='white'),
            tickangle=45
        ),
        paper_bgcolor=THEME_COLORS["dark_green"],
        plot_bgcolor=THEME_COLORS["dark_green"],
        font=dict(color='white')
    )
    
    # Add overall performance line
    total_gain = portfolio_df['gain_loss'].sum()
    fig.add_hline(
        y=total_gain,
        line_dash="dash",
        line_color=THEME_COLORS["lapis_lazuli"] if total_gain >= 0 else THEME_COLORS["rojo"],
        line_width=2,
        annotation_text=f"Total: ${total_gain:,.2f}",
        annotation_position="top right",
        annotation_font=dict(color='white', size=12)
    )
    
    return fig

def historical_performance_chart(historical_values):
    """Chart historical portfolio value and individual holdings over time with enhanced visualization"""
    # --- Check for required data structure ---
    required_cols = {'date', 'symbol', 'market_value'}
    if not required_cols.issubset(historical_values.columns):
        st.warning("Insufficient data for historical performance chart. Need 'date', 'symbol', and 'market_value'.")
        return None
    # Ensure 'date' is datetime
    historical_values['date'] = pd.to_datetime(historical_values['date'])
    # Separate Total Portfolio Value
    total_hist = historical_values[historical_values['symbol'] == 'PORTFOLIO_TOTAL'].sort_values('date')
    # Get individual holdings history (exclude the total marker)
    asset_hist = historical_values[historical_values['symbol'] != 'PORTFOLIO_TOTAL'].sort_values(['symbol', 'date'])
    if total_hist.empty and asset_hist.empty:
        st.warning("No historical data available to plot.")
        return None
    fig = go.Figure()
    # --- Plot Total Portfolio Value ---
    if not total_hist.empty:
        fig.add_trace(go.Scatter(
            x=total_hist['date'],
            y=total_hist['market_value'],
            mode='lines+markers',
            name='Total Portfolio Value',
            line=dict(color=THEME_COLORS.get("lapis_lazuli", "#3498db"), width=4), # Thicker line for total
            marker=dict(size=8, color=THEME_COLORS.get("lapis_lazuli", "#3498db")),
            hovertemplate="<b>Total Portfolio</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>",
            visible=True  # Always visible
        ))
    # --- Plot Individual Holdings ---
    if not asset_hist.empty:
        symbols = asset_hist['symbol'].unique()
        # Use SCATTER_COLORS or CHART_COLORS for distinct asset lines
        num_colors = len(SCATTER_COLORS)
        for i, symbol in enumerate(symbols):
            symbol_data = asset_hist[asset_hist['symbol'] == symbol].sort_values('date')
            if not symbol_data.empty:
                color = SCATTER_COLORS[i % num_colors] # Cycle through colors
                # --- Line Chart for Asset Value Over Time ---
                fig.add_trace(go.Scatter(
                    x=symbol_data['date'],
                    y=symbol_data['market_value'],
                    mode='lines+markers',
                    name=f'{symbol} Value',
                    line=dict(color=color, width=2),
                    marker=dict(size=6, color=color),
                    opacity=0.8, # Slightly transparent
                    legendgroup=symbol, # Group traces for toggling
                    hovertemplate=f"<b>{symbol} Value</b><br>Date: %{{x}}<br>Value: $%{{y:,.2f}}<extra></extra>",
                    visible='legendonly'  # Hidden by default
                ))
    # --- Layout ---
    fig.update_layout(
        title=dict(
            text='Historical Portfolio & Asset Values',
            font=dict(size=18, color='white'),
            x=0.5
        ),
        hovermode='x unified', # Unified hover across x-axis for easier comparison
        xaxis=dict(
            title=dict(text='Date', font=dict(color='white')),
            tickfont=dict(color='white'),
            gridcolor="rgba(255,255,255,0.1)",
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        ),
        yaxis=dict(
            title=dict(text='Value ($)', font=dict(color='white')), # Updated title
            tickfont=dict(color='white'),
            gridcolor="rgba(255,255,255,0.1)",
            tickformat="$,.2f" # Format y-axis ticks as currency
        ),
        legend=dict(
            font=dict(color='white'),
            orientation="v",  # Vertical legend
            yanchor="top",
            y=1.0,  # Top of the chart
            xanchor="right",
            x=1.0,  # Right side of the chart
            bgcolor="rgba(0,0,0,0)" # Transparent legend background
        ),
        paper_bgcolor=THEME_COLORS["dark_green"],
        plot_bgcolor=THEME_COLORS["dark_green"],
        font=dict(color='black'),
        margin=dict(l=50, r=50, t=80, b=80),
        hoverlabel=dict(font_size=12, font_family="Arial", font_color="black"),
        xaxis_rangeselector=dict(
            font=dict(color='black'),
            activecolor="#00666f",
            borderwidth=1,
            bordercolor="rgba(255,255,255,0.2)"
        )
    )
    # Add range slider for better navigation
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    return fig

def sector_allocation_chart(portfolio_df):
    """Create modern sunburst chart of sector allocation"""
    if portfolio_df.empty or 'sector' not in portfolio_df.columns:
        return None
    
    # Group by sector and holding
    sector_data = portfolio_df.groupby(['sector', 'symbol'])['market_value'].sum().reset_index()
    
    fig = px.sunburst(
        sector_data,
        path=['sector', 'symbol'],
        values='market_value',
        title='Sector Allocation',
        color_discrete_sequence=CHART_COLORS
    )
    
    fig.update_traces(
        textinfo='label+percent parent',
        textfont=dict(color='white', size=11),
        hovertemplate="<b>%{label}</b><br>Value: $%{value:,.2f}<br>Percentage: %{percentParent}<extra></extra>",
        marker=dict(line=dict(color='white', width=1))
    )
    
    fig.update_layout(
        title=dict(
            font=dict(size=18, color='white'),
            x=0.5
        ),
        paper_bgcolor=THEME_COLORS["dark_green"],
        plot_bgcolor=THEME_COLORS["dark_green"],
        font=dict(color='white')
    )
    
    return fig

def risk_analysis_chart(portfolio_df):
    """Visualize risk analysis through volatility with modern styling"""
    if portfolio_df.empty or 'gain_loss_pct' not in portfolio_df.columns:
        return None

    # Ensure relevant columns are numeric
    portfolio_df['gain_loss_pct'] = pd.to_numeric(portfolio_df['gain_loss_pct'], errors='coerce')
    portfolio_df['market_value'] = pd.to_numeric(portfolio_df['market_value'], errors='coerce')
    portfolio_df['quantity'] = pd.to_numeric(portfolio_df['quantity'], errors='coerce')

    # Drop rows with missing or invalid data
    portfolio_df = portfolio_df.dropna(subset=['gain_loss_pct', 'market_value', 'quantity'])

    # Filter out rows with non-positive quantity or market value (optional)
    portfolio_df = portfolio_df[(portfolio_df['quantity'] > 0) & (portfolio_df['market_value'] > 0)]

    fig = px.scatter(
        portfolio_df,
        x='market_value',
        y='gain_loss_pct',
        size='quantity',
        color='symbol',
        hover_name='symbol',
        title='Risk Analysis: Value vs. Performance',
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
        line_width=2
    )
    fig.add_vline(
        x=median_value,
        line_dash="dash",
        line_color="rgba(255,255,255,0.8)",
        line_width=2
    )

    fig.update_layout(
        title=dict(
            font=dict(size=18, color='white'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text='Current Value ($)', font=dict(color='white')),
            tickfont=dict(color='white'),
            gridcolor="rgba(255,255,255,0.1)"
        ),
        yaxis=dict(
            title=dict(text='Gain/Loss (%)', font=dict(color='white')),
            tickfont=dict(color='white'),
            gridcolor="rgba(255,255,255,0.1)"
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


def capital_allocation_line(portfolio_df, risk_free_rate):
    """Create modern Capital Allocation Line visualization"""
    if portfolio_df.empty or 'annualized_return' not in portfolio_df.columns or 'annualized_volatility' not in portfolio_df.columns:
        return None
    
    # Create efficient frontier (simplified)
    portfolios = []
    for i in range(100):
        w = np.random.random(len(portfolio_df))
        w /= np.sum(w)
        
        port_return = np.sum(w * portfolio_df['annualized_return'])
        port_volatility = np.sqrt(np.sum(w**2 * portfolio_df['annualized_volatility']**2))
        
        portfolios.append({
            'return': port_return,
            'volatility': port_volatility,
            'sharpe': (port_return - risk_free_rate) / port_volatility if port_volatility > 0 else 0
        })
    
    portfolios_df = pd.DataFrame(portfolios)
    
    # Create figure
    fig = go.Figure()
    
    # Scatter plot of portfolios
    fig.add_trace(go.Scatter(
        x=portfolios_df['volatility'],
        y=portfolios_df['return'],
        mode='markers',
        marker=dict(
            color=portfolios_df['sharpe'],
            colorscale=[[0, THEME_COLORS["rojo"]], [0.5, THEME_COLORS["midnight_green"]], [1, THEME_COLORS["lapis_lazuli"]]],
            size=8,
            opacity=0.7,
            colorbar=dict(title="Sharpe Ratio", titlefont=dict(color='white'), tickfont=dict(color='white'))
        ),
        name='Efficient Frontier',
        hovertemplate="<b>Portfolio</b><br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: %{marker.color:.2f}<extra></extra>"
    ))
    
    # Add CAL line
    if not portfolios_df.empty:
        max_sharpe_idx = portfolios_df['sharpe'].idxmax()
        max_sharpe = portfolios_df.loc[max_sharpe_idx]
        cal_slope = (max_sharpe['return'] - risk_free_rate) / max_sharpe['volatility']
        
        cal_x = np.linspace(0, portfolios_df['volatility'].max() * 1.2, 50)
        cal_y = risk_free_rate + cal_slope * cal_x
        
        fig.add_trace(go.Scatter(
            x=cal_x,
            y=cal_y,
            mode='lines',
            name='Capital Allocation Line',
            line=dict(color=THEME_COLORS["rojo"], width=3, dash='dash'),
            hovertemplate="<b>CAL</b><br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>"
        ))
        
        # Add risk-free rate point
        fig.add_trace(go.Scatter(
            x=[0],
            y=[risk_free_rate],
            mode='markers',
            name='Risk-Free Rate',
            marker=dict(color='white', size=12, symbol='star'),
            hovertemplate="<b>Risk-Free Rate</b><br>Return: %{y:.2%}<extra></extra>"
        ))
        
        # Add optimal portfolio
        fig.add_trace(go.Scatter(
            x=[max_sharpe['volatility']],
            y=[max_sharpe['return']],
            mode='markers',
            name='Optimal Portfolio',
            marker=dict(color=THEME_COLORS["lapis_lazuli"], size=12, symbol='diamond'),
            hovertemplate="<b>Optimal Portfolio</b><br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>"
        ))
    
    fig.update_layout(
        title=dict(
            text='Capital Allocation Line (CAL)',
            font=dict(size=18, color='white'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text='Volatility (Standard Deviation)', font=dict(color='white')),
            tickfont=dict(color='white'),
            gridcolor="rgba(255,255,255,0.1)",
            tickformat='.1%'
        ),
        yaxis=dict(
            title=dict(text='Expected Return', font=dict(color='white')),
            tickfont=dict(color='white'),
            gridcolor="rgba(255,255,255,0.1)",
            tickformat='.1%'
        ),
        legend=dict(
            font=dict(color='white'),
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        paper_bgcolor=THEME_COLORS["dark_green"],
        plot_bgcolor=THEME_COLORS["dark_green"],
        font=dict(color='white')
    )
    
    return fig



def historical_portfolio_heatmap(historical_values):
    """
    Create a heatmap showing asset values over time.
    X-axis: Date
    Y-axis: Asset Symbol
    Cell Color/Intensity: Market Value (or Percentage Contribution)
    """
    # --- Check for required data structure ---
    required_cols = {'date', 'symbol', 'market_value'}
    if not required_cols.issubset(historical_values.columns):
        st.warning("Insufficient data for historical heatmap. Need 'date', 'symbol', and 'market_value'.")
        return None

    # Ensure 'date' is datetime
    historical_values['date'] = pd.to_datetime(historical_values['date'])

    # --- Prepare Data ---
    # Pivot the data to have dates as columns and symbols as rows for market_value
    # Use 'first' or 'mean' in case there are duplicates for a symbol on a specific date (unlikely but safe)
    pivot_df = historical_values.pivot_table(
        index='symbol',
        columns='date',
        values='market_value',
        aggfunc='first' # or 'mean' or 'sum' if multiple entries per symbol/date
    ).fillna(0) # Fill NaNs with 0 for assets not present on a date

    # Optional: Remove the 'PORTFOLIO_TOTAL' row if it's included, as it's not an individual asset
    if 'PORTFOLIO_TOTAL' in pivot_df.index:
        pivot_df = pivot_df.drop('PORTFOLIO_TOTAL')

    # Sort symbols for consistent y-axis order (e.g., alphabetically)
    pivot_df = pivot_df.sort_index()

    # Sort dates for consistent x-axis order
    pivot_df = pivot_df.sort_index(axis=1)

    # --- Handle Case Where There's No Individual Asset Data ---
    if pivot_df.empty:
        st.info("No individual asset historical data available for heatmap.")
        return None

    # --- Create Heatmap ---
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns, # Dates
        y=pivot_df.index,   # Symbols
        colorscale='mint', # You can experiment with other scales like 'Viridis', 'Cividis', 'Plasma'
        hoverongaps=False,
        hovertemplate=
            "<b>%{y}</b><br>" + # Symbol
            "Date: %{x}<br>" + # Date
            "Value: $%{z:,.2f}<extra></extra>", # Market Value
        # Make colorbar labels white for dark theme
        colorbar=dict(
            title="Market Value ($)",
            title_side="right", # Corrected property name
            tickfont=dict(color='white'),
            title_font=dict(color='white') # Corrected property name
        )
    ))

    # --- Layout ---
    fig.update_layout(
        title=dict(
            text='Historical Portfolio Value Heatmap',
            font=dict(size=18, color='white'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text='Date', font=dict(color='white')),
            tickfont=dict(color='white'),
            # type='date', # Plotly usually infers this correctly
            tickangle=45, # Angle dates for better readability
            tickformat="%Y-%m-%d" # Format date ticks if needed
        ),
        yaxis=dict(
            title=dict(text='Asset Symbol', font=dict(color='white')),
            tickfont=dict(color='white'),
            automargin=True # Adjust margin to fit long symbol names
        ),
        paper_bgcolor=THEME_COLORS["dark_green"],
        plot_bgcolor=THEME_COLORS["dark_green"],
        font=dict(color='white'),
        # Adjust margins if needed
        margin=dict(l=100, r=50, t=80, b=100) # Increase left/bottom margins for labels
    )

    return fig

# --- Alternative: Percentage Contribution Heatmap ---
def historical_portfolio_percentage_heatmap(historical_values):
    """
    Create a heatmap showing the percentage contribution of each asset to the total portfolio on each date.
    X-axis: Date
    Y-axis: Asset Symbol
    Cell Color/Intensity: Percentage of Total Portfolio
    """
    # --- Check for required data structure ---
    required_cols = {'date', 'symbol', 'market_value'}
    if not required_cols.issubset(historical_values.columns):
        st.warning("Insufficient data for historical percentage heatmap. Need 'date', 'symbol', and 'market_value'.")
        return None

    # Ensure 'date' is datetime
    historical_values['date'] = pd.to_datetime(historical_values['date'])

    # --- Prepare Data ---
    # Pivot the data
    pivot_df = historical_values.pivot_table(
        index='symbol',
        columns='date',
        values='market_value',
        aggfunc='first'
    ).fillna(0)

    # Sort symbols and dates
    pivot_df = pivot_df.sort_index().sort_index(axis=1)

    # --- Calculate Percentage Contribution ---
    # Get the total portfolio value for each date
    total_by_date = historical_values[historical_values['symbol'] == 'PORTFOLIO_TOTAL'].set_index('date')['market_value']
    
    # Reindex total_by_date to match the columns of pivot_df (dates) to handle missing dates gracefully
    total_by_date = total_by_date.reindex(pivot_df.columns, fill_value=0)

    # Avoid division by zero: if total is 0 on a date, percentage is 0
    # Use np.where to handle division safely
    percentage_df = pivot_df.divide(total_by_date, axis=1) * 100
    percentage_df = percentage_df.fillna(0) # In case of any remaining NaNs

    # --- Handle Case Where There's No Individual Asset Data ---
    if percentage_df.empty:
        st.info("No individual asset historical data available for percentage heatmap.")
        return None

    # --- Create Heatmap ---
    fig = go.Figure(data=go.Heatmap(
        z=percentage_df.values,
        x=percentage_df.columns,
        y=percentage_df.index,
        # Use a diverging or sequential colorscale that works well for percentages
        # 'Blues' or 'Purples' are good for sequential, 'RdBu' or 'RdYlGn' for diverging
        colorscale='mint', 
        # zmin=0, # Set min/max for consistent color scaling if desired
        # zmax=100,
        hoverongaps=False,
        hovertemplate=
            "<b>%{y}</b><br>" +
            "Date: %{x}<br>" +
            "Contribution: %{z:.2f}%<extra></extra>", # Percentage
        colorbar=dict(
            title="Contribution (%)",
            title_side="right", # Corrected property name
            tickfont=dict(color='white'),
            title_font=dict(color='white') # Corrected property name
            # tickformat=".1f" # Format colorbar ticks
        )
    ))

    # --- Layout ---
    fig.update_layout(
        title=dict(
            text='Historical Portfolio Composition Heatmap',
            font=dict(size=18, color='white'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text='Date', font=dict(color='white')),
            tickfont=dict(color='white'),
            tickangle=45,
            tickformat="%Y-%m-%d"
        ),
        yaxis=dict(
            title=dict(text='Asset Symbol', font=dict(color='white')),
            tickfont=dict(color='white'),
            automargin=True
        ),
        paper_bgcolor=THEME_COLORS["dark_green"],
        plot_bgcolor=THEME_COLORS["dark_green"],
        font=dict(color='white'),
        margin=dict(l=100, r=50, t=80, b=100)
    )

    return fig

def render_equity_visualizations(tracker):
    """Render all equity visualizations with fixed metrics"""
    if tracker.portfolio.empty:
        return
    
    st.header("📊 Portfolio Visualizations")
    
    # Validate data first
    clean_portfolio = helpers.validate_portfolio_data(tracker.portfolio.copy())
    
    # Debug data quality
    with st.expander("🔧 Data Quality Debug"):
        helpers.debug_portfolio_metrics(clean_portfolio, tracker.historical_values)
        
        # Show corrected metrics
        metrics = helpers.calculate_portfolio_metrics(
            clean_portfolio, 
            tracker.historical_values, 
            getattr(tracker, 'risk_free_rate', 0.02)
        )
        
        # st.subheader("Corrected Metrics:")
        # col1, col2, col3, col4 = st.columns(4)
        
        # with col1:
        #     st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")
        #     st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.3f}")
        
        # with col2:
        #     st.metric("Annualized Return", f"{metrics['annualized_return']:.2f}%")
        #     st.metric("Calmar Ratio", f"{metrics['calmar_ratio']:.3f}")
        
        # with col3:
        #     st.metric("Annualized Volatility", f"{metrics['annualized_volatility']:.2f}%")
        
        # with col4:
        #     st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
    
    # Asset Flow Sankey Diagram - prominently displayed
    sankey_fig = asset_flow_sankey(tracker.portfolio)
    if sankey_fig:
        st.plotly_chart(sankey_fig, use_container_width=True)
    
    # Asset class breakdown and crypto-specific charts
    col1, col2 = st.columns(2)
    with col1:
        asset_breakdown_fig = asset_class_breakdown_chart(tracker.portfolio) 
        if asset_breakdown_fig:
            st.plotly_chart(asset_breakdown_fig, use_container_width=True)
    
    with col2:
        crypto_fig = crypto_holdings_chart(tracker.portfolio)
        if crypto_fig:
            st.plotly_chart(crypto_fig, use_container_width=True)
        else:
            st.info("No cryptocurrency holdings detected")
    
    # Traditional charts
    col3, col4 = st.columns(2)
    with col3:
        allocation_fig = portfolio_allocation_chart(tracker.portfolio)
        if allocation_fig:
            st.plotly_chart(allocation_fig, use_container_width=True)
        
    with col4:
        waterfall_fig = gain_loss_waterfall(tracker.portfolio)
        if waterfall_fig:
            st.plotly_chart(waterfall_fig, use_container_width=True)
    
    # Historical performance
    hist_fig = historical_performance_chart(tracker.historical_values)
    if hist_fig:
        st.plotly_chart(hist_fig, use_container_width=True)
    
    # Capital Allocation Line
    cal_fig = capital_allocation_line(tracker.portfolio, tracker.risk_free_rate)
    if cal_fig:
        st.plotly_chart(cal_fig, use_container_width=True)
    
    # Sector allocation
    sector_fig = sector_allocation_chart(tracker.portfolio)
    if sector_fig:
        st.plotly_chart(sector_fig, use_container_width=True)

    historic_heatmap_fig = historical_portfolio_heatmap(tracker.historical_values)
    if historic_heatmap_fig:
        st.plotly_chart(historic_heatmap_fig, use_container_width=True)

    historic_per_heatmap_fig =historical_portfolio_percentage_heatmap(tracker.historical_values)
    if historic_per_heatmap_fig:
        st.plotly_chart(historic_per_heatmap_fig, use_container_width=True)

    # Risk analysis
    risk_fig = risk_analysis_chart(tracker.portfolio)
    if risk_fig:
        st.plotly_chart(risk_fig, use_container_width=True)