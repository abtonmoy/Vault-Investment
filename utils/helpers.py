import streamlit as st
import re
import pandas as pd
import numpy as np


def debug_print(title, content):
    """Helper for debugging output"""
    if st.checkbox(f"Show {title}", key=f"debug_{title.replace(' ', '_').lower()}"):
        st.markdown(f"### 🔍 {title}")
        if isinstance(content, list):
            for i, item in enumerate(content[:20]):  # Limit to first 20 items
                st.text(f"{i+1}: {item}")
        elif isinstance(content, dict):
            for key, value in list(content.items())[:20]:
                st.text(f"{key}: {value}")
        else:
            st.text(str(content)[:2000])  # Limit text length


def calculate_financial_metrics(df):
    """Centralized function for consistent financial calculations"""
    
    # Income: Only transactions categorized as 'Income' with positive amounts
    income_mask = (df['category'] == 'Income') & (df['amount'] > 0)
    total_income = df[income_mask]['amount'].sum()
    
    # Expenses: Negative amounts excluding transfers and income categories
    expense_mask = (df['amount'] < 0) & (df['category'] != 'Transfer') & (df['category'] != 'Income')
    total_expenses = abs(df[expense_mask]['amount'].sum())
    
    # Net income
    net_income = total_income - total_expenses
    
    # Additional metrics
    total_transactions = len(df)
    categories = df['category'].nunique()
    
    return {
        'total_income': total_income,
        'total_expenses': total_expenses,
        'net_income': net_income,
        'total_transactions': total_transactions,
        'categories': categories
    }


def safe_divide(numerator, denominator, default=0):
    """Safely divide two numbers, handling division by zero"""
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return default
    return numerator / denominator

def calculate_portfolio_metrics(portfolio_df, historical_values, risk_free_rate=0.02):
    """Calculate portfolio metrics with proper error handling"""
    
    if portfolio_df.empty or historical_values.empty:
        return {
            'sharpe_ratio': 0,
            'annualized_return': 0,
            'annualized_volatility': 0,
            'max_drawdown': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0
        }
    
    try:
        # Ensure historical_values is sorted by date
        historical_values = historical_values.sort_values('date')
        
        # Calculate daily returns from historical values
        historical_values['daily_return'] = historical_values['total_value'].pct_change()
        
        # Remove NaN values
        daily_returns = historical_values['daily_return'].dropna()
        
        if len(daily_returns) == 0:
            return {
                'sharpe_ratio': 0,
                'annualized_return': 0,
                'annualized_volatility': 0,
                'max_drawdown': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0
            }
        
        # Calculate annualized return (assuming daily data)
        trading_days = 252
        mean_daily_return = daily_returns.mean()
        annualized_return = (1 + mean_daily_return) ** trading_days - 1
        
        # Calculate annualized volatility
        daily_volatility = daily_returns.std()
        annualized_volatility = daily_volatility * np.sqrt(trading_days)
        
        # Calculate Sharpe ratio
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = safe_divide(excess_return, annualized_volatility)
        
        # Calculate max drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0
        
        # Calculate downside deviation for Sortino ratio
        negative_returns = daily_returns[daily_returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(trading_days) if len(negative_returns) > 0 else 0
        sortino_ratio = safe_divide(excess_return, downside_deviation)
        
        # Calculate Calmar ratio
        calmar_ratio = safe_divide(annualized_return, max_drawdown)
        
        return {
            'sharpe_ratio': round(sharpe_ratio, 3),
            'annualized_return': round(annualized_return * 100, 2),  # Convert to percentage
            'annualized_volatility': round(annualized_volatility * 100, 2),  # Convert to percentage
            'max_drawdown': round(max_drawdown * 100, 2),  # Convert to percentage
            'sortino_ratio': round(sortino_ratio, 3),
            'calmar_ratio': round(calmar_ratio, 3)
        }
        
    except Exception as e:
        print(f"Error in calculate_portfolio_metrics: {e}")
        return {
            'sharpe_ratio': 0,
            'annualized_return': 0,
            'annualized_volatility': 0,
            'max_drawdown': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0
        }

def validate_portfolio_data(portfolio_df):
    """Validate and clean portfolio data"""
    
    # Check for required columns
    required_columns = ['symbol', 'market_value']
    missing_columns = [col for col in required_columns if col not in portfolio_df.columns]
    
    if missing_columns:
        print(f"Warning: Missing required columns: {missing_columns}")
        return portfolio_df
    
    # Clean numeric columns
    numeric_columns = ['market_value', 'gain_loss', 'gain_loss_pct', 'quantity']
    for col in numeric_columns:
        if col in portfolio_df.columns:
            # Convert to numeric, replacing errors with 0
            portfolio_df[col] = pd.to_numeric(portfolio_df[col], errors='coerce').fillna(0)
    
    # Remove rows with zero or negative market values
    portfolio_df = portfolio_df[portfolio_df['market_value'] > 0]
    
    return portfolio_df

def debug_portfolio_metrics(portfolio_df, historical_values):
    """Debug function to check data quality"""
    
    print("=== Portfolio Data Debug ===")
    print(f"Portfolio rows: {len(portfolio_df)}")
    print(f"Historical data rows: {len(historical_values)}")
    
    if not portfolio_df.empty:
        print(f"Portfolio columns: {list(portfolio_df.columns)}")
        print(f"Total market value: ${portfolio_df['market_value'].sum():,.2f}")
        
        # Check for NaN values
        nan_columns = portfolio_df.columns[portfolio_df.isna().any()].tolist()
        if nan_columns:
            print(f"Columns with NaN values: {nan_columns}")
    
    if not historical_values.empty:
        print(f"Historical data columns: {list(historical_values.columns)}")
        print(f"Date range: {historical_values['date'].min()} to {historical_values['date'].max()}")
        
        if 'total_value' in historical_values.columns:
            print(f"Value range: ${historical_values['total_value'].min():,.2f} to ${historical_values['total_value'].max():,.2f}")
