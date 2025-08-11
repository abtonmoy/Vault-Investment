import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict
from datetime import datetime


class PortfolioCharts:
    """Handles all portfolio visualization charts, including historical performance,
    composition over time, asset performance, metrics dashboards, optimization visuals, etc."""

    def __init__(self, template='plotly_dark'):
        self.template = template

    # -------------------------
    # Utility: Robust Date Parsing
    # -------------------------
    @staticmethod
    def _parse_dates_safe(dates):
        """
        Safely parse a list/Series of dates with flexible format detection.
        Forces UTC to avoid timezone-related plotting issues.
        """
        parsed = pd.to_datetime(dates, errors='coerce', infer_datetime_format=True, utc=True)
        if parsed.isna().any():
            bad_vals = [str(d) for d, p in zip(dates, parsed) if pd.isna(p)]
            if bad_vals:
                print(f"[WARN] Dropped unparseable dates: {bad_vals[:5]}{'...' if len(bad_vals) > 5 else ''}")
        return parsed

    # -------------------------
    # Portfolio Worth Chart
    # -------------------------
    def create_portfolio_worth_chart(self, tracker, historical_data: pd.DataFrame = None) -> go.Figure:
        """
        Create portfolio worth visualization tracking ACTUAL value over time.
        This shows how your current holdings would have performed historically.
        """
        try:
            # Method 1: Use transaction data to build true portfolio timeline
            if hasattr(tracker, 'transactions') and tracker.transactions is not None and not tracker.transactions.empty:
                portfolio_timeline = self._build_portfolio_timeline_from_transactions(tracker.transactions)
                if portfolio_timeline:
                    return self._create_timeline_chart(portfolio_timeline)
            
            # Method 2: Use historical data to show how current holdings performed over time
            if hasattr(tracker, 'historical_values') and tracker.historical_values is not None:
                portfolio_timeline = self._calculate_historical_portfolio_value(tracker)
                if portfolio_timeline:
                    return self._create_timeline_chart(portfolio_timeline)
            
            # Method 3: Fallback to current holdings visualization
            if hasattr(tracker, 'portfolio') and not tracker.portfolio.empty:
                return self._create_current_holdings_chart(tracker.portfolio)
            
            return self._create_empty_chart("No portfolio data available for tracking")

        except Exception as e:
            st.error(f"Error creating portfolio worth chart: {str(e)}")
            print(f"[ERROR] Portfolio worth chart failed: {e}")
            return self._create_empty_chart(f"Error: {str(e)}")
    def _calculate_historical_portfolio_value(self, tracker):
        """
        Calculate how current portfolio holdings would have performed historically.
        This gives proper scale by using actual portfolio quantities with historical prices.
        """
        try:
            portfolio = tracker.portfolio
            historical_values = tracker.historical_values
            
            if isinstance(historical_values, dict):
                # Check if it's date-based dictionary
                dates = [k for k in historical_values.keys() if isinstance(k, str) and len(k) >= 8]
                if not dates:
                    return []
                
                portfolio_timeline = []
                
                for date_str in sorted(dates):
                    try:
                        parsed_date = pd.to_datetime(date_str, errors='coerce')
                        if pd.isnull(parsed_date):
                            continue
                            
                        date_data = historical_values[date_str]
                        total_value = 0.0
                        valid_assets = 0
                        
                        # Calculate portfolio value using CURRENT holdings with HISTORICAL prices
                        for _, holding in portfolio.iterrows():
                            symbol = holding['symbol']
                            current_quantity = holding.get('quantity', 0)
                            
                            if symbol and pd.notnull(current_quantity) and current_quantity > 0:
                                if symbol in date_data:
                                    try:
                                        historical_price = float(date_data[symbol])
                                        asset_value = current_quantity * historical_price
                                        total_value += asset_value
                                        valid_assets += 1
                                    except (ValueError, TypeError):
                                        continue
                        
                        # Only add if we have data for at least some assets
                        if valid_assets > 0:
                            portfolio_timeline.append({
                                'Date': parsed_date,
                                'Portfolio Value': total_value,
                                'Assets Tracked': valid_assets
                            })
                    
                    except Exception as e:
                        print(f"[WARN] Skipped date {date_str}: {e}")
                        continue
                
                return portfolio_timeline
                
            elif isinstance(historical_values, pd.DataFrame):
                return self._calculate_from_dataframe(tracker, historical_values)
            
            return []
            
        except Exception as e:
            print(f"[ERROR] Failed to calculate historical portfolio value: {e}")
            return []

    def _calculate_from_dataframe(self, tracker, df):
        """Calculate portfolio value from DataFrame format historical data."""
        try:
            if 'date' not in df.columns or 'symbol' not in df.columns:
                return []
                
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            
            portfolio = tracker.portfolio
            portfolio_timeline = []
            
            # Get unique dates
            for date in sorted(df['date'].unique()):
                date_data = df[df['date'] == date]
                total_value = 0.0
                valid_assets = 0
                
                for _, holding in portfolio.iterrows():
                    symbol = holding['symbol']
                    current_quantity = holding.get('quantity', 0)
                    
                    if symbol and pd.notnull(current_quantity) and current_quantity > 0:
                        symbol_data = date_data[date_data['symbol'] == symbol]
                        if not symbol_data.empty:
                            # Use price if available, otherwise try market_value per share
                            if 'price' in symbol_data.columns:
                                price = symbol_data['price'].iloc[0]
                            elif 'market_value' in symbol_data.columns:
                                # Assume market_value is total value, need to get per-share
                                symbol_qty = symbol_data.get('quantity', current_quantity).iloc[0]
                                if symbol_qty > 0:
                                    price = symbol_data['market_value'].iloc[0] / symbol_qty
                                else:
                                    continue
                            else:
                                continue
                                
                            try:
                                asset_value = current_quantity * float(price)
                                total_value += asset_value
                                valid_assets += 1
                            except (ValueError, TypeError):
                                continue
                
                if valid_assets > 0:
                    portfolio_timeline.append({
                        'Date': date,
                        'Portfolio Value': total_value,
                        'Assets Tracked': valid_assets
                    })
            
            return portfolio_timeline
            
        except Exception as e:
            print(f"[ERROR] Failed to calculate from DataFrame: {e}")
            return []

        
    def _build_portfolio_timeline_from_transactions(self, transactions_df):
        """
        Build true portfolio value timeline from actual transactions.
        This tracks when you actually bought/sold assets and shows real performance.
        """
        try:
            df = transactions_df.copy()
            df = df.dropna(how='all').reset_index(drop=True)
            
            # Clean and parse dates
            df['Activity Date'] = pd.to_datetime(df['Activity Date'], format='%m/%d/%Y', errors='coerce')
            df = df.dropna(subset=['Activity Date'])
            
            # Clean numeric columns
            df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)
            df['Price'] = pd.to_numeric(df['Price'].astype(str).str.replace('$', '', regex=False), errors='coerce').fillna(0)
            df['Amount'] = pd.to_numeric(df['Amount'].astype(str).str.replace('$', '', regex=False), errors='coerce').fillna(0)
            
            # Sort by date
            df = df.sort_values('Activity Date')
            
            # Track portfolio state over time
            cash = 0.0
            holdings = {}  # {asset: shares}
            current_prices = {}  # Latest known price for each asset
            portfolio_timeline = []
            
            for _, transaction in df.iterrows():
                date = transaction['Activity Date']
                trans_code = transaction['Trans Code']
                instrument = transaction['Instrument']
                quantity = transaction['Quantity']
                price = transaction['Price']
                amount = transaction['Amount']
                
                # Update holdings and cash based on transaction type
                if trans_code == 'Buy':
                    holdings[instrument] = holdings.get(instrument, 0) + quantity
                    current_prices[instrument] = price
                    cash += amount  # Amount is negative for purchases
                    
                elif trans_code == 'Sell':
                    holdings[instrument] = holdings.get(instrument, 0) - quantity
                    if holdings[instrument] <= 0:
                        holdings.pop(instrument, None)
                    current_prices[instrument] = price
                    cash += amount  # Amount is positive for sales
                    
                elif trans_code in ['CDIV', 'RTP', 'ACH']:  # Dividends, transfers
                    cash += amount
                    
                elif trans_code in ['AFEE', 'DTAX']:  # Fees, taxes
                    cash += amount
                
                # Update current price for the instrument
                if instrument and price > 0:
                    current_prices[instrument] = price
                
                # Calculate total portfolio value at this point in time
                portfolio_value = cash
                for asset, shares in holdings.items():
                    if asset in current_prices:
                        portfolio_value += shares * current_prices[asset]
                
                # Record this portfolio snapshot
                portfolio_timeline.append({
                    'Date': date,
                    'Portfolio Value': portfolio_value,
                    'Cash': cash,
                    'Holdings Value': portfolio_value - cash,
                    'Transaction': f"{trans_code}: {instrument}" if instrument else trans_code
                })
            
            return portfolio_timeline
            
        except Exception as e:
            print(f"[ERROR] Failed to build timeline from transactions: {e}")
            return []
        
    def _extract_actual_portfolio_snapshots(self, historical_values):
        """
        Extract actual portfolio snapshots from historical data.
        This assumes historical_values contains real portfolio snapshots, not reconstructed data.
        """
        try:
            portfolio_timeline = []
            
            if isinstance(historical_values, dict):
                # Check if this looks like date-keyed snapshots
                date_keys = [k for k in historical_values.keys() if isinstance(k, str) and len(k) >= 8]
                if date_keys:
                    for date_str in sorted(date_keys):
                        try:
                            date_data = historical_values[date_str]
                            parsed_date = pd.to_datetime(date_str, errors='coerce')
                            
                            if pd.isna(parsed_date):
                                continue
                            
                            # Look for total portfolio value
                            if isinstance(date_data, dict) and 'total_value' in date_data:
                                total_value = float(date_data['total_value'])
                            elif isinstance(date_data, dict) and 'portfolio_value' in date_data:
                                total_value = float(date_data['portfolio_value'])
                            elif isinstance(date_data, (int, float)):
                                total_value = float(date_data)
                            else:
                                # Skip this entry if we can't determine portfolio value
                                continue
                            
                            portfolio_timeline.append({
                                'Date': parsed_date,
                                'Portfolio Value': total_value
                            })
                            
                        except (ValueError, TypeError) as e:
                            print(f"[WARN] Skipped snapshot for {date_str}: {e}")
                            continue
                            
            elif isinstance(historical_values, pd.DataFrame):
                # DataFrame with portfolio snapshots
                df = historical_values.copy()
                
                # Find date and value columns
                date_col = None
                value_col = None
                
                for col in df.columns:
                    if 'date' in col.lower():
                        date_col = col
                    elif any(term in col.lower() for term in ['total', 'portfolio', 'value', 'worth']):
                        value_col = col
                
                if date_col and value_col:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    df = df.dropna(subset=[date_col])
                    
                    for _, row in df.iterrows():
                        portfolio_timeline.append({
                            'Date': row[date_col],
                            'Portfolio Value': float(row[value_col])
                        })
            
            return portfolio_timeline if portfolio_timeline else []
            
        except Exception as e:
            print(f"[ERROR] Failed to extract portfolio snapshots: {e}")
            return []


    def _calculate_portfolio_value_from_transactions(self, transactions_df):
        """
        Calculate portfolio value over time from transaction data.
        Fixed version that properly handles all transaction types.
        """
        try:
            # Make a copy to avoid modifying the original
            df = transactions_df.copy()
            
            # Clean up the data
            df = df.dropna(how='all').reset_index(drop=True)
            
            # Convert date columns
            df['Activity Date'] = pd.to_datetime(df['Activity Date'], format='%m/%d/%Y', errors='coerce')
            df = df.dropna(subset=['Activity Date'])
            
            # Convert numeric columns
            df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)
            df['Price'] = pd.to_numeric(df['Price'].astype(str).str.replace('$', '', regex=False), errors='coerce').fillna(0)
            df['Amount'] = pd.to_numeric(df['Amount'].astype(str).str.replace('$', '', regex=False), errors='coerce').fillna(0)
            
            # Sort by date
            df = df.sort_values('Activity Date')
            
            # Initialize portfolio tracking
            cash = 0.0
            holdings = {}
            portfolio_values = []
            
            # Get all unique dates
            min_date = df['Activity Date'].min()
            max_date = df['Activity Date'].max()
            all_dates = pd.date_range(start=min_date, end=max_date)
            
            # Create a price history for each asset
            price_history = {}
            for asset in df['Instrument'].unique():
                if asset and asset != '':
                    asset_data = df[df['Instrument'] == asset].copy()
                    asset_data = asset_data.sort_values('Activity Date')
                    price_history[asset] = asset_data.set_index('Activity Date')['Price']
            
            # Process each date
            for date in all_dates:
                # Get transactions for this date
                day_transactions = df[df['Activity Date'] == date]
                
                # Process each transaction
                for _, transaction in day_transactions.iterrows():
                    trans_code = transaction['Trans Code']
                    instrument = transaction['Instrument']
                    quantity = transaction['Quantity']
                    price = transaction['Price']
                    amount = transaction['Amount']
                    
                    # Handle different transaction types
                    if trans_code == 'Buy':
                        # Add to holdings
                        if instrument in holdings:
                            holdings[instrument] += quantity
                        else:
                            holdings[instrument] = quantity
                        # Update cash (negative amount for purchase)
                        cash += amount
                        
                    elif trans_code == 'Sell':
                        # Remove from holdings
                        if instrument in holdings:
                            holdings[instrument] -= quantity
                            if holdings[instrument] <= 0:
                                del holdings[instrument]
                        # Update cash (positive amount from sale)
                        cash += amount
                        
                    elif trans_code in ['CDIV', 'RTP', 'ACH']:
                        # Cash dividend or transfer
                        cash += amount
                        
                    elif trans_code in ['AFEE', 'DTAX']:
                        # Fee or tax
                        cash += amount
                
                # Calculate portfolio value for this date
                portfolio_value = cash
                
                # Add value of all holdings using the latest known price
                for asset, shares in holdings.items():
                    if asset in price_history:
                        # Get the latest price on or before this date
                        asset_prices = price_history[asset]
                        if date in asset_prices.index:
                            current_price = asset_prices.loc[date]
                        else:
                            # Find the last price before this date
                            earlier_prices = asset_prices[asset_prices.index <= date]
                            if not earlier_prices.empty:
                                current_price = earlier_prices.iloc[-1]
                            else:
                                continue
                        
                        portfolio_value += shares * current_price
                
                # Record the portfolio value for this date
                portfolio_values.append({
                    'Date': date,
                    'Portfolio Value': portfolio_value
                })
            
            # Convert to DataFrame
            portfolio_df = pd.DataFrame(portfolio_values)
            
            # Calculate percentage change from initial value
            if not portfolio_df.empty:
                initial_value = portfolio_df['Portfolio Value'].iloc[0]
                portfolio_df['Pct Change'] = (portfolio_df['Portfolio Value'] - initial_value) / initial_value * 100
            
            return portfolio_df
            
        except Exception as e:
            print(f"[ERROR] Failed to calculate portfolio value: {e}")
            return pd.DataFrame()



    def _create_historical_worth_chart(self, tracker, historical_data: pd.DataFrame = None) -> go.Figure:
        """
        Chooses the correct method to process historical data depending on its structure:
        - Date-keyed dictionary
        - Structured dict with columns
        - DataFrame
        """
        try:
            hv = tracker.historical_values
            print(f"[INFO] Historical data type: {type(hv).__name__}")

            if isinstance(hv, dict):
                date_keys = [k for k in hv.keys() if isinstance(k, str) and len(k) >= 6]
                if date_keys:
                    return self._create_chart_from_date_dict(tracker, hv)
                elif all(isinstance(v, list) for v in hv.values()):
                    return self._create_chart_from_structured_data(tracker, hv)
            elif isinstance(hv, pd.DataFrame):
                return self._create_chart_from_dataframe(tracker, hv)

            st.warning("Historical data structure not recognized, showing current holdings")
            return self._create_current_holdings_chart(tracker.portfolio)

        except Exception as e:
            st.error(f"Error creating historical worth chart: {str(e)}")
            print(f"[ERROR] Historical worth chart failed: {e}")
            return self._create_current_holdings_chart(tracker.portfolio)

    def _create_chart_from_date_dict(self, tracker, historical_values):
        """
        Handles case where historical_values is a dict keyed by date strings.
        Each value is a dict of {symbol: price} or contains a 'total_value' key.
        """
        try:
            portfolio_timeline = []
            date_keys = sorted([k for k in historical_values.keys() if isinstance(k, str) and len(k) >= 6])

            for date_str in date_keys:
                try:
                    date_data = historical_values[date_str]
                    total_value = 0.0

                    # Prefer pre-calculated portfolio total
                    if isinstance(date_data, dict) and 'total_value' in date_data:
                        total_value = float(date_data['total_value'])
                    elif isinstance(date_data, dict) and all(isinstance(v, (int, float)) for v in date_data.values()):
                        # If values are already market values per asset
                        total_value = sum(date_data.values())
                    else:
                        # Fallback: current qty * historical price
                        for _, holding in tracker.portfolio.iterrows():
                            symbol = holding['symbol']
                            qty = holding.get('quantity', 0)
                            if symbol and symbol in date_data:
                                try:
                                    price = float(date_data[symbol])
                                    total_value += qty * price
                                except (ValueError, TypeError):
                                    continue


                    parsed_date = self._parse_dates_safe([date_str])[0]
                    if not pd.isna(parsed_date):
                        portfolio_timeline.append({'date': parsed_date, 'total_value': total_value})

                except Exception as e:
                    print(f"[WARN] Skipped date {date_str}: {e}")

            if not portfolio_timeline:
                return self._create_current_holdings_chart(tracker.portfolio)

            return self._create_timeline_chart(portfolio_timeline)

        except Exception as e:
            print(f"[ERROR] _create_chart_from_date_dict failed: {e}")
            return self._create_current_holdings_chart(tracker.portfolio)

    def _create_chart_from_structured_data(self, tracker, historical_values):
        """
        Handles case where historical_values is a dict with structured lists for each column.
        """
        try:
            df = pd.DataFrame(historical_values)
            return self._create_chart_from_dataframe(tracker, df)
        except Exception as e:
            print(f"[ERROR] Structured data conversion failed: {e}")
            return self._create_current_holdings_chart(tracker.portfolio)

    def _create_chart_from_dataframe(self, tracker, df):
        """
        Handles case where historical_values is already a DataFrame.
        Groups by date, sums values, and plots timeline.
        """
        try:
            if not {'date', 'market_value'}.issubset(df.columns):
                st.warning("DataFrame missing required columns")
                return self._create_current_holdings_chart(tracker.portfolio)

            df['date'] = self._parse_dates_safe(df['date'])
            df = df.dropna(subset=['date'])
            df = df.groupby('date', as_index=False)['market_value'].sum()
            df = df.rename(columns={'market_value': 'total_value'})
            return self._create_timeline_chart(df.to_dict('records'))


        except Exception as e:
            print(f"[ERROR] DataFrame chart creation failed: {e}")
            return self._create_current_holdings_chart(tracker.portfolio)

    def _create_timeline_chart(self, portfolio_data):
        """
        Create timeline chart from portfolio data with proper scaling.
        """
        try:
            if not portfolio_data:
                return self._create_empty_chart("No timeline data available")
            
            # Convert to DataFrame
            if isinstance(portfolio_data, list):
                df = pd.DataFrame(portfolio_data)
            else:
                df = portfolio_data.copy()
            
            # Ensure proper column names
            if 'Date' not in df.columns:
                if 'date' in df.columns:
                    df = df.rename(columns={'date': 'Date'})
                else:
                    return self._create_empty_chart("No date column found")
            
            if 'Portfolio Value' not in df.columns:
                if 'Portfolio_Value' in df.columns:
                    df = df.rename(columns={'Portfolio_Value': 'Portfolio Value'})
                elif 'total_value' in df.columns:
                    df = df.rename(columns={'total_value': 'Portfolio Value'})
                elif 'portfolio_value' in df.columns:
                    df = df.rename(columns={'portfolio_value': 'Portfolio Value'})
                else:
                    return self._create_empty_chart("No portfolio value column found")
            
            # Clean and sort data
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date', 'Portfolio Value'])
            df = df.sort_values('Date')
            
            if df.empty:
                return self._create_empty_chart("No valid portfolio timeline data")
            
            # Calculate percentage change from first value
            first_value = df['Portfolio Value'].iloc[0]
            if first_value > 0:
                df['Pct Change'] = ((df['Portfolio Value'] - first_value) / first_value) * 100
            
            fig = go.Figure()
            
            # Main portfolio value line
            fig.add_trace(go.Scatter(
                x=df['Date'], 
                y=df['Portfolio Value'], 
                mode='lines+markers',
                name='Portfolio Value', 
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=4),
                hovertemplate="<b>Portfolio Value</b><br>" +
                            "Date: %{x|%B %d, %Y}<br>" +
                            "Value: $%{y:,.2f}<extra></extra>"
            ))
            
            # Add percentage change on secondary axis if we have multiple points
            if len(df) > 1 and 'Pct Change' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['Date'], 
                    y=df['Pct Change'], 
                    mode='lines',
                    name='% Change from Start', 
                    yaxis='y2',
                    line=dict(color='#ff7f0e', dash='dash', width=2),
                    hovertemplate="<b>Performance</b><br>" +
                                "Date: %{x|%B %d, %Y}<br>" +
                                "Change: %{y:.2f}%<extra></extra>"
                ))
            
            # Add breakdown traces if available
            if 'Cash' in df.columns and 'Holdings Value' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['Cash'],
                    mode='lines',
                    name='Cash',
                    line=dict(color='green', dash='dot'),
                    hovertemplate="<b>Cash</b><br>" +
                                "Date: %{x|%B %d, %Y}<br>" +
                                "Cash: $%{y:,.2f}<extra></extra>"
                ))
                
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['Holdings Value'],
                    mode='lines',
                    name='Holdings Value',
                    line=dict(color='purple', dash='dot'),
                    hovertemplate="<b>Holdings</b><br>" +
                                "Date: %{x|%B %d, %Y}<br>" +
                                "Holdings: $%{y:,.2f}<extra></extra>"
                ))
            
            # Update layout with proper formatting
            fig.update_layout(
                title='Portfolio Value Over Time',
                xaxis_title='Date',
                yaxis_title='Portfolio Value ($)',
                template=self.template,
                width=900,
                height=600,
                hovermode='x unified'
            )
            
            # Add secondary y-axis for percentage change
            if len(df) > 1 and 'Pct Change' in df.columns:
                fig.update_layout(
                    yaxis2=dict(
                        title='% Change from Start',
                        overlaying='y',
                        side='right',
                        ticksuffix='%',
                        showgrid=False
                    )
                )
            
            # Format primary y-axis with proper scaling
            max_val = df['Portfolio Value'].max()
            if max_val > 1000000:
                fig.update_yaxes(tickformat='$,.0f')
            elif max_val > 1000:
                fig.update_yaxes(tickformat='$,.0f')
            else:
                fig.update_yaxes(tickformat='$,.2f')
            
            return fig

        except Exception as e:
            print(f"[ERROR] Timeline chart creation failed: {e}")
            return self._create_empty_chart(f"Timeline Error: {e}")

        
    # -------------------------
    # Current Holdings Chart
    # -------------------------
    def _create_current_holdings_chart(self, portfolio: pd.DataFrame) -> go.Figure:
        """
        Fallback chart when no historical data is available:
        Horizontal bar of holdings by market value (or quantity).
        """
        try:
            fig = go.Figure()
            if 'market_value' in portfolio.columns:
                valid_portfolio = portfolio[
                    (portfolio['market_value'].notna()) & (portfolio['market_value'] > 0)
                ]
                if len(valid_portfolio) == 0:
                    return self._create_empty_chart("No valid market values found")

                portfolio_sorted = valid_portfolio.sort_values('market_value', ascending=True)
                fig.add_trace(go.Bar(
                    x=portfolio_sorted['market_value'],
                    y=portfolio_sorted['symbol'],
                    orientation='h',
                    marker_color='lightblue',
                    hovertemplate="<b>%{y}</b><br>Value: $%{x:,.2f}<extra></extra>"
                ))

                fig.update_layout(
                    title='Current Portfolio Holdings by Value',
                    xaxis_title='Market Value ($)',
                    yaxis_title='Symbol',
                    template=self.template,
                    height=max(400, len(valid_portfolio) * 30)
                )
                fig.update_xaxes(tickformat='$,.0f')
            elif 'quantity' in portfolio.columns:
                valid_portfolio = portfolio[
                    (portfolio['quantity'].notna()) & (portfolio['quantity'] > 0)
                ]
                if len(valid_portfolio) == 0:
                    return self._create_empty_chart("No valid quantities found")

                portfolio_sorted = valid_portfolio.sort_values('quantity', ascending=True)
                fig.add_trace(go.Bar(
                    x=portfolio_sorted['quantity'],
                    y=portfolio_sorted['symbol'],
                    orientation='h',
                    marker_color='lightcoral',
                    hovertemplate="<b>%{y}</b><br>Quantity: %{x}<extra></extra>"
                ))

                fig.update_layout(
                    title='Current Portfolio Holdings by Quantity',
                    xaxis_title='Quantity',
                    yaxis_title='Symbol',
                    template=self.template,
                    height=max(400, len(valid_portfolio) * 30)
                )
            else:
                return self._create_empty_chart("No market_value or quantity data available")

            return fig

        except Exception as e:
            st.error(f"Error creating current holdings chart: {str(e)}")
            print(f"[ERROR] Current holdings chart failed: {e}")
            return self._create_empty_chart(f"Error: {str(e)}")

    # -------------------------
    # Empty Chart Helper
    # -------------------------
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Shows an empty chart with a central message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            template=self.template,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            height=400
        )
        return fig

  
    
    def create_asset_performance_chart(self, tracker, historical_data: pd.DataFrame = None) -> go.Figure:
        """
        Create individual asset performance chart over time
        
        Args:
            tracker: Portfolio tracker with portfolio data
            historical_data: Historical price data
            
        Returns:
            Plotly figure showing individual asset performance
        """
        try:
            if (not hasattr(tracker, 'historical_values') or 
                tracker.historical_values is None or
                len(tracker.historical_values) == 0):
                st.warning("No historical data available for asset performance chart")
                return self._create_empty_chart("No Historical Data Available")
            
            portfolio = tracker.portfolio
            historical_values = tracker.historical_values
            
            # Handle different data structures
            if isinstance(historical_values, dict):
                # Check if it's date-based dictionary
                dates = [k for k in historical_values.keys() if isinstance(k, str) and len(k) >= 8]
                if dates:
                    return self._create_asset_performance_from_dates(portfolio, historical_values, dates)
                elif isinstance(historical_values, dict) and 'date' in historical_values:
                    # Try to handle structured data
                    try:
                        df = pd.DataFrame(historical_values)
                        return self._create_asset_performance_from_df(portfolio, df)
                    except:
                        pass
            elif isinstance(historical_values, pd.DataFrame):
                return self._create_asset_performance_from_df(portfolio, historical_values)
            
            return self._create_empty_chart("Cannot process historical data for asset performance")
            
        except Exception as e:
            st.error(f"Error creating asset performance chart: {str(e)}")
            print(f"Asset performance chart error: {e}")
            return self._create_empty_chart(f"Asset Performance Error: {str(e)}")
    
    def _create_asset_performance_from_dates(self, portfolio, historical_values, dates):
        """Create asset performance chart from date-based historical values"""
        try:
            symbols = portfolio['symbol'].unique()
            fig = go.Figure()
            
            # Create performance lines for each asset
            for symbol in symbols:
                asset_values = []
                valid_dates = []
                
                for date_str in sorted(dates):
                    if symbol in historical_values[date_str]:
                        try:
                            price = float(historical_values[date_str][symbol])
                            asset_values.append(price)
                            parsed_date = pd.to_datetime(date_str, errors='coerce')
                            if not pd.isnull(parsed_date):
                                valid_dates.append(parsed_date)
                        except (ValueError, TypeError):
                            continue
                
                if len(asset_values) > 1 and len(valid_dates) == len(asset_values):
                    # Calculate normalized performance (percentage change from start)
                    start_price = asset_values[0]
                    if start_price > 0:
                        normalized_values = [((price - start_price) / start_price) * 100 for price in asset_values]
                        
                        fig.add_trace(go.Scatter(
                            x=valid_dates,
                            y=normalized_values,
                            mode='lines',
                            name=symbol,
                            hovertemplate=f"<b>{symbol}</b><br>" +
                                        "Date: %{x}<br>" +
                                        "Change: %{y:.2f}%<br>" +
                                        "<extra></extra>"
                        ))
            
            if len(fig.data) == 0:
                return self._create_empty_chart("No valid asset performance data")
            
            fig.update_layout(
                title='Individual Asset Performance (% Change)',
                xaxis_title='Date',
                yaxis_title='Percentage Change from Start (%)',
                template=self.template,
                width=900,
                height=600,
                hovermode='x unified'
            )
            
            fig.update_yaxes(ticksuffix='%')
            return fig
            
        except Exception as e:
            print(f"Error in _create_asset_performance_from_dates: {e}")
            return self._create_empty_chart(f"Asset Performance Error: {str(e)}")
    
    def _create_asset_performance_from_df(self, portfolio, df):
        """Create asset performance chart from DataFrame"""
        try:
            if 'date' not in df.columns or 'symbol' not in df.columns:
                return self._create_empty_chart("DataFrame missing required columns")
            
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            
            symbols = portfolio['symbol'].unique()
            fig = go.Figure()
            
            for symbol in symbols:
                symbol_data = df[df['symbol'] == symbol].sort_values('date')
                if len(symbol_data) > 1:
                    # Use market_value if available, otherwise try to find price column
                    value_col = 'market_value' if 'market_value' in symbol_data.columns else None
                    if value_col and symbol_data[value_col].notna().any():
                        values = symbol_data[value_col].values
                        dates = symbol_data['date'].values
                        
                        # Calculate percentage change from start
                        start_value = values[0]
                        if start_value > 0:
                            pct_changes = [((val - start_value) / start_value) * 100 for val in values]
                            
                            fig.add_trace(go.Scatter(
                                x=dates,
                                y=pct_changes,
                                mode='lines',
                                name=symbol,
                                hovertemplate=f"<b>{symbol}</b><br>" +
                                            "Date: %{x}<br>" +
                                            "Change: %{y:.2f}%<br>" +
                                            "<extra></extra>"
                            ))
            
            if len(fig.data) == 0:
                return self._create_empty_chart("No valid asset performance data found")
            
            fig.update_layout(
                title='Individual Asset Performance (% Change)',
                xaxis_title='Date',
                yaxis_title='Percentage Change from Start (%)',
                template=self.template,
                width=900,
                height=600,
                hovermode='x unified'
            )
            
            fig.update_yaxes(ticksuffix='%')
            return fig
            
        except Exception as e:
            print(f"Error in _create_asset_performance_from_df: {e}")
            return self._create_empty_chart(f"Asset Performance Error: {str(e)}")
    
    def create_portfolio_composition_over_time(self, tracker) -> go.Figure:
        """
        Create stacked area chart showing portfolio composition changes over time
        
        Args:
            tracker: Portfolio tracker with historical data
            
        Returns:
            Plotly figure showing composition changes
        """
        try:
            if (not hasattr(tracker, 'historical_values') or 
                tracker.historical_values is None or
                len(tracker.historical_values) == 0):
                st.warning("No historical data available for composition chart")
                return self._create_empty_chart("No Historical Data Available")
            
            composition_data = self._calculate_proper_composition(tracker)
            
            if not composition_data:
                return self._create_empty_chart("No valid composition data")
            
            return self._create_composition_chart(composition_data)
            
        except Exception as e:
            st.error(f"Error creating composition chart: {str(e)}")
            print(f"Composition chart error: {e}")
            return self._create_empty_chart(f"Composition Error: {str(e)}")

    def _calculate_proper_composition(self, tracker):
        """Calculate proper portfolio composition over time."""
        try:
            portfolio = tracker.portfolio
            historical_values = tracker.historical_values
            composition_data = []
            
            if isinstance(historical_values, dict):
                dates = [k for k in historical_values.keys() if isinstance(k, str) and len(k) >= 8]
                
                for date_str in sorted(dates):
                    try:
                        parsed_date = pd.to_datetime(date_str, errors='coerce')
                        if pd.isnull(parsed_date):
                            continue
                            
                        date_data = historical_values[date_str]
                        
                        total_value = 0.0
                        asset_values = {}
                        
                        # First pass: calculate total portfolio value and individual asset values
                        for _, holding in portfolio.iterrows():
                            symbol = holding['symbol']
                            current_quantity = holding.get('quantity', 0)
                            
                            if symbol and pd.notnull(current_quantity) and current_quantity > 0:
                                if symbol in date_data:
                                    try:
                                        historical_price = float(date_data[symbol])
                                        asset_value = current_quantity * historical_price
                                        asset_values[symbol] = asset_value
                                        total_value += asset_value
                                    except (ValueError, TypeError):
                                        continue
                        
                        # Second pass: calculate percentages
                        if total_value > 0 and asset_values:
                            composition_record = {'date': parsed_date}
                            
                            # Add each asset's percentage
                            for symbol, value in asset_values.items():
                                composition_record[symbol] = (value / total_value) * 100
                            
                            composition_data.append(composition_record)
                    
                    except Exception as e:
                        print(f"Error processing composition for date {date_str}: {e}")
                        continue
            
            return composition_data
            
        except Exception as e:
            print(f"[ERROR] Failed to calculate proper composition: {e}")
            return []
    
    def _create_composition_from_dates(self, portfolio, historical_values, dates):
        """Create composition chart from date-based historical values"""
        try:
            composition_data = []
            
            for date_str in sorted(dates):
                try:
                    parsed_date = pd.to_datetime(date_str, errors='coerce')
                    if pd.isnull(parsed_date):
                        continue
                        
                    date_data = historical_values[date_str]
                    
                    # Check if this is pre-calculated composition data
                    if isinstance(date_data, dict) and any(key for key in date_data.keys() if not isinstance(key, str) or len(key) > 10):
                        # This might be composition percentages already
                        composition_record = {'date': parsed_date}
                        for key, value in date_data.items():
                            if isinstance(key, str) and key != 'total_value':
                                try:
                                    # If value is already a percentage (0-100), use it
                                    # If value is a decimal (0-1), convert to percentage
                                    pct_value = float(value)
                                    if pct_value <= 1.0:  # Assume it's decimal, convert to percentage
                                        pct_value = pct_value * 100
                                    composition_record[key] = pct_value
                                except (ValueError, TypeError):
                                    continue
                        
                        if len(composition_record) > 1:  # More than just 'date'
                            composition_data.append(composition_record)
                    else:
                        # Calculate composition from prices and quantities
                        total_value = 0
                        asset_values = {}
                        
                        # First pass: calculate total portfolio value and individual asset values
                        for _, holding in portfolio.iterrows():
                            symbol = holding['symbol']
                            quantity = holding.get('quantity', 0)
                            
                            if symbol and pd.notnull(quantity) and symbol in date_data:
                                try:
                                    value = float(quantity) * float(date_data[symbol])
                                    asset_values[symbol] = value
                                    total_value += value
                                except (ValueError, TypeError):
                                    continue
                        
                        # Second pass: calculate percentages
                        if total_value > 0:
                            composition_record = {'date': parsed_date}
                            for symbol, value in asset_values.items():
                                composition_record[symbol] = (value / total_value) * 100
                            composition_data.append(composition_record)
                
                except Exception as e:
                    print(f"Error processing composition for date {date_str}: {e}")
                    continue
            
            if not composition_data:
                return self._create_empty_chart("No valid composition data")
            
            return self._create__composition_chart(composition_data)
            
        except Exception as e:
            print(f"Error in _create_composition_from_dates: {e}")
            return self._create_empty_chart(f"Composition Error: {str(e)}")

    def _create__composition_chart(self, composition_data):
        """Create the actual fixed composition stacked area chart."""
        try:
            composition_df = pd.DataFrame(composition_data)
            composition_df = composition_df.sort_values('date')
            
            # Fill NaN values with 0
            composition_df = composition_df.fillna(0)
            
            # Get symbols (exclude date column)
            symbols = [col for col in composition_df.columns if col != 'date']
            
            # Limit to top holdings to avoid overcrowding
            if len(symbols) > 20:
                # Calculate average allocation for each symbol
                avg_allocations = {}
                for symbol in symbols:
                    avg_allocations[symbol] = composition_df[symbol].mean()
                
                # Keep top 15 holdings, group rest as "Others"
                top_symbols = sorted(avg_allocations.items(), key=lambda x: x[1], reverse=True)[:15]
                keep_symbols = [s[0] for s in top_symbols]
                other_symbols = [s for s in symbols if s not in keep_symbols]
                
                if other_symbols:
                    composition_df['Others'] = composition_df[other_symbols].sum(axis=1)
                    composition_df = composition_df[['date'] + keep_symbols + ['Others']]
                    symbols = keep_symbols + ['Others']
                else:
                    symbols = keep_symbols
            
            # Create stacked area chart
            fig = go.Figure()
            
            # Use a better color palette
            colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel + px.colors.qualitative.Set1
            
            for i, symbol in enumerate(symbols):
                fig.add_trace(go.Scatter(
                    x=composition_df['date'],
                    y=composition_df[symbol],
                    fill='tonexty' if i > 0 else 'tozeroy',
                    mode='none',
                    name=symbol,
                    fillcolor=colors[i % len(colors)],
                    stackgroup='one',  # This ensures proper stacking
                    hovertemplate=f"<b>{symbol}</b><br>" +
                                "Date: %{x|%Y-%m-%d}<br>" +
                                "Allocation: %{y:.1f}%<br>" +
                                "<extra></extra>"
                ))
            
            fig.update_layout(
                title='Portfolio Composition Over Time',
                xaxis_title='Date',
                yaxis_title='Allocation (%)',
                template=self.template,
                width=900,
                height=600,
                hovermode='x unified',
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                )
            )
            
            # Fix y-axis to show 0-100%
            fig.update_yaxes(range=[0, 100], ticksuffix='%')
            
            return fig
            
        except Exception as e:
            print(f"[ERROR] Fixed composition chart creation failed: {e}")
            return self._create_empty_chart(f"Composition Chart Error: {str(e)}")

    def _create_composition_from_df(self, portfolio, df):
        """Create composition chart from DataFrame"""
        try:
            if 'date' not in df.columns or 'symbol' not in df.columns or 'market_value' not in df.columns:
                return self._create_empty_chart("DataFrame missing required columns for composition")
            
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            
            # Group by date and calculate total portfolio value and individual percentages
            composition_data = []
            
            for date in df['date'].unique():
                date_data = df[df['date'] == date]
                total_value = date_data['market_value'].sum()
                
                if total_value > 0:
                    composition_record = {'date': date}
                    for _, row in date_data.iterrows():
                        symbol = row['symbol']
                        percentage = (row['market_value'] / total_value) * 100
                        composition_record[symbol] = percentage
                    composition_data.append(composition_record)
            
            if not composition_data:
                return self._create_empty_chart("No valid composition data")
            
            return self._create_composition_chart(composition_data)
            
        except Exception as e:
            print(f"Error in _create_composition_from_df: {e}")
            return self._create_empty_chart(f"Composition Error: {str(e)}")
    
    def _create_composition_chart(self, composition_data):
        """Create the actual composition stacked area chart"""
        try:
            composition_df = pd.DataFrame(composition_data)
            composition_df = composition_df.sort_values('date')
            
            # Fill NaN values with 0
            composition_df = composition_df.fillna(0)
            
            # Create stacked area chart
            fig = go.Figure()
            
            symbols = [col for col in composition_df.columns if col != 'date']
            colors = px.colors.qualitative.Set3[:len(symbols)]
            
            for i, symbol in enumerate(symbols):
                fig.add_trace(go.Scatter(
                    x=composition_df['date'],
                    y=composition_df[symbol],
                    fill='tonexty' if i > 0 else 'tozeroy',
                    mode='none',
                    name=symbol,
                    fillcolor=colors[i % len(colors)],
                    hovertemplate=f"<b>{symbol}</b><br>" +
                                "Date: %{x}<br>" +
                                "Allocation: %{y:.1f}%<br>" +
                                "<extra></extra>"
                ))
            
            fig.update_layout(
                title='Portfolio Composition Over Time',
                xaxis_title='Date',
                yaxis_title='Allocation (%)',
                template=self.template,
                width=900,
                height=600,
                hovermode='x unified'
            )
            
            fig.update_yaxes(range=[0, 100], ticksuffix='%')
            
            return fig
            
        except Exception as e:
            print(f"Error in _create_composition_chart: {e}")
            return self._create_empty_chart(f"Composition Chart Error: {str(e)}")
    
    def create_portfolio_metrics_dashboard(self, tracker) -> go.Figure:
        """
        Create a FIXED dashboard with key portfolio metrics over time.
        """
        try:
            if (not hasattr(tracker, 'historical_values') or 
                tracker.historical_values is None or
                len(tracker.historical_values) == 0):
                st.warning("No historical data available for metrics dashboard")
                return self._create_empty_chart("No Historical Data Available")
            
            # Calculate metrics properly
            metrics_data = self._calculate_proper_metrics(tracker)
            
            if not metrics_data:
                return self._create_empty_chart("No valid metrics data")
            
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df = metrics_df.sort_values('date')
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Portfolio Value ($)', 'Number of Holdings', 
                              'Concentration Index (HHI)', 'Diversification Score'),
                specs=[[{'secondary_y': False}, {'secondary_y': False}],
                       [{'secondary_y': False}, {'secondary_y': False}]],
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            # Portfolio Value - with proper scaling
            fig.add_trace(
                go.Scatter(x=metrics_df['date'], y=metrics_df['total_value'],
                          mode='lines+markers', name='Portfolio Value',
                          line=dict(color='blue', width=2),
                          marker=dict(size=4),
                          hovertemplate="Date: %{x|%Y-%m-%d}<br>Value: $%{y:,.2f}<extra></extra>"),
                row=1, col=1
            )
            
            # Number of Holdings
            fig.add_trace(
                go.Scatter(x=metrics_df['date'], y=metrics_df['num_holdings'],
                          mode='lines+markers', name='Holdings Count',
                          line=dict(color='green', width=2),
                          marker=dict(size=4),
                          hovertemplate="Date: %{x|%Y-%m-%d}<br>Holdings: %{y}<extra></extra>"),
                row=1, col=2
            )
            
            # Concentration Index (bounded between 0 and 1)
            fig.add_trace(
                go.Scatter(x=metrics_df['date'], y=metrics_df['concentration_hhi'],
                          mode='lines+markers', name='HHI',
                          line=dict(color='orange', width=2),
                          marker=dict(size=4),
                          hovertemplate="Date: %{x|%Y-%m-%d}<br>HHI: %{y:.4f}<extra></extra>"),
                row=2, col=1
            )
            
            # Diversification Score (0-1, where 1 is most diversified)
            fig.add_trace(
                go.Scatter(x=metrics_df['date'], y=metrics_df['diversification_score'],
                          mode='lines+markers', name='Diversification',
                          line=dict(color='purple', width=2),
                          marker=dict(size=4),
                          hovertemplate="Date: %{x|%Y-%m-%d}<br>Diversification: %{y:.4f}<extra></extra>"),
                row=2, col=2
            )
            
            fig.update_layout(
                title='Portfolio Metrics Dashboard',
                template=self.template,
                height=800,
                showlegend=False
            )
            
            # Update y-axis formats
            fig.update_yaxes(tickformat='$,.0f', row=1, col=1)  # Portfolio value
            fig.update_yaxes(range=[0, max(metrics_df['num_holdings']) + 1], row=1, col=2)  # Holdings count
            fig.update_yaxes(range=[0, 1], tickformat='.3f', row=2, col=1)  # HHI
            fig.update_yaxes(range=[0, 1], tickformat='.3f', row=2, col=2)  # Diversification
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating metrics dashboard: {str(e)}")
            print(f"Metrics dashboard error: {e}")
            return self._create_empty_chart(f"Metrics Error: {str(e)}")
        
    def _calculate_proper_metrics(self, tracker):
        """Calculate proper portfolio metrics over time."""
        try:
            portfolio = tracker.portfolio
            historical_values = tracker.historical_values
            metrics_data = []
            
            if isinstance(historical_values, dict):
                dates = [k for k in historical_values.keys() if isinstance(k, str) and len(k) >= 8]
                
                for date_str in sorted(dates):
                    try:
                        parsed_date = pd.to_datetime(date_str, errors='coerce')
                        if pd.isnull(parsed_date):
                            continue
                            
                        date_data = historical_values[date_str]
                        
                        total_value = 0.0
                        asset_values = []
                        valid_holdings = 0
                        
                        # Calculate for each holding
                        for _, holding in portfolio.iterrows():
                            symbol = holding['symbol']
                            current_quantity = holding.get('quantity', 0)
                            
                            if symbol and pd.notnull(current_quantity) and current_quantity > 0:
                                if symbol in date_data:
                                    try:
                                        historical_price = float(date_data[symbol])
                                        asset_value = current_quantity * historical_price
                                        total_value += asset_value
                                        asset_values.append(asset_value)
                                        valid_holdings += 1
                                    except (ValueError, TypeError):
                                        continue
                        
                        # Calculate concentration metrics only if we have valid data
                        if total_value > 0 and asset_values and valid_holdings > 0:
                            # Herfindahl-Hirschman Index (concentration)
                            concentrations = [(value / total_value) ** 2 for value in asset_values]
                            hhi = sum(concentrations)
                            
                            # Diversification score (inverse of concentration)
                            diversification_score = 1 - hhi
                            
                            metrics_data.append({
                                'date': parsed_date,
                                'total_value': total_value,
                                'num_holdings': valid_holdings,
                                'concentration_hhi': hhi,
                                'diversification_score': max(0, diversification_score)
                            })
                    
                    except Exception as e:
                        print(f"Error processing metrics for date {date_str}: {e}")
                        continue
            
            return metrics_data
            
        except Exception as e:
            print(f"[ERROR] Failed to calculate proper metrics: {e}")
            return []
    
    def _calculate_metrics_from_dates(self, portfolio, historical_values, dates):
        """Calculate metrics from date-based historical values"""
        try:
            metrics_data = []
            
            for date_str in sorted(dates):
                try:
                    parsed_date = pd.to_datetime(date_str, errors='coerce')
                    if pd.isnull(parsed_date):
                        continue
                        
                    date_data = historical_values[date_str]
                    
                    # Check if we have pre-calculated metrics
                    if isinstance(date_data, dict) and 'total_value' in date_data:
                        total_value = float(date_data['total_value'])
                        # Try to extract other metrics if available
                        num_holdings = date_data.get('num_holdings', len([k for k in date_data.keys() if k != 'total_value']))
                        
                        # If we have individual asset values, calculate concentration
                        asset_values = []
                        for key, value in date_data.items():
                            if key != 'total_value' and isinstance(value, (int, float)):
                                try:
                                    asset_values.append(float(value))
                                except (ValueError, TypeError):
                                    continue
                        
                        if not asset_values:
                            # Calculate from portfolio holdings
                            asset_values = []
                            for _, holding in portfolio.iterrows():
                                symbol = holding['symbol']
                                quantity = holding.get('quantity', 0)
                                
                                if symbol and pd.notnull(quantity) and symbol in date_data:
                                    try:
                                        value = float(quantity) * float(date_data[symbol])
                                        asset_values.append(value)
                                    except (ValueError, TypeError):
                                        continue
                    else:
                        # Calculate everything from scratch
                        total_value = 0
                        asset_values = []
                        
                        for _, holding in portfolio.iterrows():
                            symbol = holding['symbol']
                            quantity = holding.get('quantity', 0)
                            
                            if symbol and pd.notnull(quantity) and symbol in date_data:
                                try:
                                    value = float(quantity) * float(date_data[symbol])
                                    total_value += value
                                    asset_values.append(value)
                                except (ValueError, TypeError):
                                    continue
                    
                    # Calculate concentration metrics (Herfindahl index)
                    if total_value > 0 and asset_values:
                        # Filter out zero values
                        valid_values = [v for v in asset_values if v > 0]
                        if valid_values:
                            concentrations = [(value / total_value) ** 2 for value in valid_values]
                            hhi = sum(concentrations)
                            
                            metrics_data.append({
                                'date': parsed_date,
                                'total_value': total_value,
                                'num_holdings': len(valid_values),
                                'concentration_hhi': hhi,
                                'diversification_score': max(0, 1 - hhi)
                            })
                
                except Exception as e:
                    print(f"Error processing metrics for date {date_str}: {e}")
                    continue
            
            return metrics_data
            
        except Exception as e:
            print(f"Error in _calculate_metrics_from_dates: {e}")
            return []
    
    def _calculate_metrics_from_df(self, portfolio, df):
        """Calculate metrics from DataFrame"""
        try:
            if 'date' not in df.columns or 'market_value' not in df.columns or 'symbol' not in df.columns:
                return []
            
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            
            metrics_data = []
            
            for date in df['date'].unique():
                date_data = df[df['date'] == date]
                total_value = date_data['market_value'].sum()
                asset_values = date_data['market_value'].tolist()
                
                # Calculate concentration (Herfindahl index)
                if total_value > 0 and asset_values:
                    concentrations = [(value / total_value) ** 2 for value in asset_values if value > 0]
                    hhi = sum(concentrations)
                    
                    metrics_data.append({
                        'date': date,
                        'total_value': total_value,
                        'num_holdings': len([v for v in asset_values if v > 0]),
                        'concentration_hhi': hhi,
                        'diversification_score': 1 - hhi
                    })
            
            return metrics_data
            
        except Exception as e:
            print(f"Error in _calculate_metrics_from_df: {e}")
            return []
    
    # Keep all existing optimization-related methods unchanged
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
            st.error(f"Error plotting efficient frontier with multiple portfolios: {str(e)}")
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
            for i, row in enumerate(correlation_matrix.index):
                for j, col in enumerate(correlation_matrix.columns):
                    fig.add_annotation(
                        x=col,
                        y=row,
                        text=str(round(correlation_matrix.iloc[i, j], 3)),
                        showarrow=False,
                        font=dict(color="yellow", size=12)  # Your desired color
                        )
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