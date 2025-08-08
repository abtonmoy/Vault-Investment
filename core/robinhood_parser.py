import re
from datetime import datetime, timedelta
import pdfplumber
import PyPDF2
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from fuzzywuzzy import fuzz
from datetime import date


class InvestmentTracker:
    def __init__(self):
        self.portfolio = pd.DataFrame()
        self.transactions = pd.DataFrame()
        self.historical_values = pd.DataFrame()
        self.risk_free_rate = 0.02  # Default risk-free rate (2%)
    
    def parse_robinhood_csv(self, uploaded_file):
        """Parse Robinhood CSV reports (positions, transactions, and account statements)"""
        debug_info = []
        debug_info.append(f"⏳ Starting Robinhood CSV parse for {uploaded_file.name}")
        
        try:
            # Read CSV with error handling
            try:
                df = pd.read_csv(uploaded_file)
                debug_info.append(f"✅ CSV read successfully. Shape: {df.shape}")
            except Exception as e:
                debug_info.append(f"❌ CSV read error: {str(e)}")
                st.error(f"Error reading CSV: {str(e)}")
                with st.expander("Debug Information"):
                    for entry in debug_info:
                        st.text(entry)
                return False
            
            # Check if DataFrame is empty
            if df.empty:
                debug_info.append("❌ CSV file is empty")
                st.error("CSV file is empty")
                with st.expander("Debug Information"):
                    for entry in debug_info:
                        st.text(entry)
                return False
            
            # Log original columns
            original_columns = df.columns.tolist()
            debug_info.append(f"📋 Original columns: {original_columns}")
            
            # Standardize column names
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            debug_info.append(f"🔄 Standardized columns: {df.columns.tolist()}")
            
            # Show first few rows for debugging
            debug_info.append(f"📊 First 3 rows of data:")
            for i, row in df.head(3).iterrows():
                debug_info.append(f"Row {i}: {dict(row)}")
            
            # Check if this is a positions report, transaction history, or account statement
            is_positions_report = 'symbol' in df.columns and 'quantity' in df.columns
            is_transaction_history = 'instrument' in df.columns and 'trans_code' in df.columns
            is_account_statement = 'activity_date' in df.columns and 'trans_code' in df.columns
            
            debug_info.append(f"🔍 File type detection:")
            debug_info.append(f"  - Positions report: {is_positions_report}")
            debug_info.append(f"  - Transaction history: {is_transaction_history}")
            debug_info.append(f"  - Account statement: {is_account_statement}")
            
            if is_positions_report:
                debug_info.append("🔍 Detected positions report format")
                success = self._process_positions_report(df, debug_info, uploaded_file.name)
            elif is_transaction_history:
                debug_info.append("🔍 Detected transaction history format")
                success = self._process_transaction_history(df, debug_info, uploaded_file.name)
            elif is_account_statement:
                debug_info.append("🔍 Detected account statement format")
                success = self._process_account_statement(df, debug_info, uploaded_file.name)
            else:
                debug_info.append("❌ Unsupported CSV format")
                debug_info.append(f"Available columns: {df.columns.tolist()}")
                st.error("Unsupported Robinhood CSV format. Please upload positions, transactions, or account statements.")
                success = False
            
            # Show debug information
            with st.expander("Robinhood CSV Debug Information"):
                for entry in debug_info:
                    st.text(entry)
            
            return success
        
        except Exception as e:
            debug_info.append(f"❌ Parse error: {str(e)}")
            import traceback
            debug_info.append(f"❌ Full traceback: {traceback.format_exc()}")
            st.error(f"Error parsing Robinhood CSV: {str(e)}")
            with st.expander("Debug Information"):
                for entry in debug_info:
                    st.text(entry)
            return False
    
    def _process_account_statement(self, df, debug_info, file_name):
        """Process Robinhood account statements (cash transfers, dividends, etc.)"""
        try:
            debug_info.append("🔧 Processing account statement")
            
            # Show available columns
            debug_info.append(f"Available columns: {df.columns.tolist()}")
            
            # Filter relevant columns
            keep_cols = ['activity_date', 'process_date', 'settle_date', 
                         'instrument', 'description', 'trans_code', 'amount']
            available_cols = [col for col in keep_cols if col in df.columns]
            debug_info.append(f"Keeping columns: {available_cols}")
            
            if not available_cols:
                debug_info.append("❌ No relevant columns found")
                return False
            
            df_filtered = df[available_cols].copy()
            
            # Convert amount to numeric if it exists
            if 'amount' in df_filtered.columns:
                # Handle different amount formats
                df_filtered['amount'] = df_filtered['amount'].astype(str)
                # Remove $ and commas, handle parentheses for negative values
                df_filtered['amount'] = df_filtered['amount'].str.replace('[\$,]', '', regex=True)
                df_filtered['amount'] = df_filtered['amount'].str.replace(r'\((.*)\)', r'-\1', regex=True)
                
                try:
                    df_filtered['amount'] = pd.to_numeric(df_filtered['amount'], errors='coerce')
                    debug_info.append("🧮 Converted amount to numeric")
                except Exception as e:
                    debug_info.append(f"⚠️ Error converting amount: {str(e)}")
            
            # Add metadata
            df_filtered['source'] = file_name
            df_filtered['brokerage'] = 'Robinhood'
            df_filtered['report_date'] = datetime.now().date()
            
            # Add to transactions
            if self.transactions.empty:
                self.transactions = df_filtered
            else:
                self.transactions = pd.concat([self.transactions, df_filtered], ignore_index=True)
            
            debug_info.append(f"✅ Added {len(df_filtered)} cash transactions")
            
            # Check if we can derive any positions from transactions
            security_transactions = df_filtered[
                df_filtered['trans_code'].str.upper().isin(['BUY', 'SELL']) &
                df_filtered['instrument'].notna() &
                (df_filtered['instrument'] != '')
            ]
            
            if not security_transactions.empty:
                debug_info.append(f"🔍 Found {len(security_transactions)} security transactions, attempting to derive positions")
                return self._derive_positions_from_transactions(security_transactions, debug_info, file_name)
            else:
                debug_info.append("ℹ️ This file contained only cash transactions, no security positions")
                st.info("Note: This file contained cash transactions, but no security positions could be derived")
                return True
                
        except Exception as e:
            debug_info.append(f"❌ Error processing account statement: {str(e)}")
            import traceback
            debug_info.append(f"❌ Full traceback: {traceback.format_exc()}")
            return False
    
    def _derive_positions_from_transactions(self, transactions_df, debug_info, file_name):
        """Derive current positions from transaction history"""
        try:
            debug_info.append("🔧 Deriving positions from transactions")
            
            positions = {}
            
            for _, row in transactions_df.iterrows():
                symbol = str(row.get('instrument', '')).strip()
                if not symbol or symbol.lower() in ['', 'nan', 'none']:
                    continue
                
                trans_code = str(row.get('trans_code', '')).strip().upper()
                if trans_code not in ['BUY', 'SELL']:
                    continue
                
                try:
                    # Get quantity and amount - handle missing quantity column
                    if 'quantity' in row and pd.notna(row['quantity']):
                        quantity = float(row['quantity'])
                    else:
                        quantity = 1.0  # Default quantity if not provided
                    
                    if 'amount' in row and pd.notna(row['amount']):
                        amount = float(row['amount'])
                    else:
                        amount = 0.0
                    
                    # Calculate price if not provided
                    if 'price' in row and pd.notna(row['price']):
                        # Clean price string (remove $ and convert)
                        price_str = str(row['price']).replace('$', '').replace(',', '')
                        try:
                            price = float(price_str)
                        except ValueError:
                            price = abs(amount) / quantity if quantity > 0 else 0.0
                    elif quantity > 0:
                        price = abs(amount) / quantity
                    else:
                        price = 0.0
                    
                except (ValueError, TypeError, ZeroDivisionError):
                    debug_info.append(f"⚠️ Skipping transaction - invalid numeric values for {symbol}")
                    continue
                
                # Initialize position if not exists
                if symbol not in positions:
                    positions[symbol] = {
                        'symbol': symbol,
                        'description': row.get('description', symbol),
                        'quantity': 0.0,
                        'cost_basis': 0.0,
                        'transactions': 0
                    }
                
                # Update position
                if trans_code == 'BUY':
                    positions[symbol]['quantity'] += quantity
                    positions[symbol]['cost_basis'] += abs(amount)
                elif trans_code == 'SELL':
                    positions[symbol]['quantity'] -= quantity
                    # Reduce cost basis proportionally
                    if positions[symbol]['quantity'] + quantity > 0:
                        avg_cost = positions[symbol]['cost_basis'] / (positions[symbol]['quantity'] + quantity)
                        positions[symbol]['cost_basis'] -= quantity * avg_cost
                
                positions[symbol]['transactions'] += 1
            
            # Convert to DataFrame and filter positive positions
            if positions:
                positions_df = pd.DataFrame(list(positions.values()))
                positions_df = positions_df[positions_df['quantity'] > 0]
                
                if not positions_df.empty:
                    # Calculate derived values - Fix dtype warnings by explicitly converting to float
                    positions_df = positions_df.copy()  # Ensure we're working with a copy
                    
                    # Convert numeric columns to float explicitly to avoid dtype warnings
                    numeric_cols = ['quantity', 'cost_basis']
                    for col in numeric_cols:
                        positions_df[col] = pd.to_numeric(positions_df[col], errors='coerce').astype(float)
                    
                    positions_df['average_price'] = positions_df['cost_basis'] / positions_df['quantity']
                    positions_df['current_price'] = positions_df['average_price']  # Placeholder
                    positions_df['market_value'] = positions_df['quantity'] * positions_df['current_price']
                    positions_df['gain_loss'] = 0.0  # No gain/loss data available
                    positions_df['gain_loss_pct'] = 0.0
                    
                    # Add metadata
                    positions_df['source'] = file_name
                    positions_df['brokerage'] = 'Robinhood'
                    positions_df['report_date'] = datetime.now().date()
                    
                    debug_info.append(f"✅ Derived {len(positions_df)} positions from transactions")
                    self._add_to_portfolio(positions_df)
                    st.success(f"Derived {len(positions_df)} positions from transaction history")
                    return True
                else:
                    debug_info.append("ℹ️ No current positions found (all sold)")
                    st.info("No current positions found - all securities appear to have been sold")
                    return True
            else:
                debug_info.append("ℹ️ No valid transactions found for position derivation")
                return True
                
        except Exception as e:
            debug_info.append(f"❌ Error deriving positions: {str(e)}")
            import traceback
            debug_info.append(f"❌ Full traceback: {traceback.format_exc()}")
            return False

    
    def _parse_positions_from_text(self, text, debug_info):
        """Basic text parsing to extract position information from PDF"""
        positions = []
        debug_info.append("🔍 Attempting to parse positions from text")
        
        # Look for common patterns in Robinhood PDFs
        lines = text.split('\n')
        
        # Simple pattern matching for stock symbols and values
        import re
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for patterns like "AAPL 100 shares $150.00 $15,000.00"
            pattern = r'([A-Z]{1,5})\s+(\d+(?:\.\d+)?)\s+shares?\s+\$?([\d,]+\.?\d*)\s+\$?([\d,]+\.?\d*)'
            match = re.search(pattern, line)
            
            if match:
                symbol = match.group(1)
                quantity = float(match.group(2))
                price = float(match.group(3).replace(',', ''))
                value = float(match.group(4).replace(',', ''))
                
                position = {
                    'symbol': symbol,
                    'description': symbol,
                    'quantity': quantity,
                    'current_price': price,
                    'market_value': value,
                    'average_price': price,  # Assumption
                    'cost_basis': value,  # Assumption
                    'gain_loss': 0.0,
                    'gain_loss_pct': 0.0
                }
                
                positions.append(position)
                debug_info.append(f"📊 Found position: {symbol} - {quantity} shares")
        
        debug_info.append(f"✅ Extracted {len(positions)} positions from text")
        return positions
    
    def _process_positions_report(self, df, debug_info, file_name):
        """Process Robinhood positions report"""
        # Filter relevant columns
        required_cols = ['symbol', 'description', 'quantity', 'average_price', 
                         'current_price', 'current_value', 'last_updated']
        available_cols = [col for col in required_cols if col in df.columns]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        debug_info.append(f"🔍 Required position columns: {required_cols}")
        debug_info.append(f"🔍 Available position columns: {available_cols}")
        debug_info.append(f"🔍 Missing position columns: {missing_cols}")
        
        # Keep only available required columns plus any extras
        df = df[available_cols + [col for col in df.columns if col not in required_cols]]
        debug_info.append(f"📊 Filtered DataFrame columns: {df.columns.tolist()}")
        
        # Convert numeric columns to appropriate dtypes to avoid warnings
        numeric_cols = ['quantity', 'average_price', 'current_price', 'current_value']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        
        # Calculate additional metrics
        if 'average_price' in df.columns and 'quantity' in df.columns:
            df['cost_basis'] = df['average_price'] * df['quantity']
            debug_info.append("🧮 Added cost_basis column")
        else:
            debug_info.append("⚠️ Cannot calculate cost_basis - missing columns")
            
        if 'current_price' in df.columns and 'quantity' in df.columns:
            df['market_value'] = df['current_price'] * df['quantity']
            debug_info.append("🧮 Added market_value column")
        elif 'current_value' in df.columns:
            df['market_value'] = df['current_value']
            debug_info.append("🧮 Using current_value as market_value")
        else:
            debug_info.append("⚠️ Cannot determine market_value - no price or value columns")
            
        if 'market_value' in df.columns and 'cost_basis' in df.columns:
            df['gain_loss'] = df['market_value'] - df['cost_basis']
            df['gain_loss_pct'] = (df['gain_loss'] / df['cost_basis']) * 100
            debug_info.append("🧮 Added gain/loss columns")
        
        # Add source information
        df['source'] = file_name
        df['brokerage'] = 'Robinhood'
        df['report_date'] = datetime.now().date()
        debug_info.append(f"🏷️ Added metadata columns")
        
        # Add to portfolio
        debug_info.append(f"📤 Adding to portfolio with columns: {df.columns.tolist()}")
        self._add_to_portfolio(df)
        
        debug_info.append("✅ Position report processed successfully")
        return True
    
    def _process_transaction_history(self, df, debug_info, file_name):
        """Process Robinhood transaction history to derive positions"""
        try:
            debug_info.append("🔧 Processing transaction history to derive positions")
            
            # Filter relevant transactions (buys and sells)
            valid_transactions = ['buy', 'sell']
            if 'trans_code' in df.columns:
                # Clean transaction codes
                df['trans_code'] = df['trans_code'].str.strip().str.lower()
                df = df[df['trans_code'].isin(valid_transactions)]
                debug_info.append(f"🔍 Filtered to {len(df)} buy/sell transactions")
            
            if df.empty:
                debug_info.append("⚠️ No buy/sell transactions found after filtering")
                st.warning("No valid buy/sell transactions found in the history")
                return False
            
            # Create position records from transactions
            positions = {}
            for _, row in df.iterrows():
                # Get symbol from instrument or description
                symbol = row.get('instrument', '')
                if pd.isna(symbol) or symbol == '':
                    symbol = row.get('description', 'UNKNOWN')
                
                # Clean symbol name
                symbol = re.sub(r'\s*Class\s+[A-Z]$', '', symbol).strip()
                symbol = symbol.split('(')[0].strip()  # Remove any text in parentheses
                
                # Get transaction details
                try:
                    # Handle quantity - it might not exist in all rows
                    if 'quantity' in row and pd.notna(row['quantity']):
                        qty = float(row['quantity'])
                    else:
                        qty = 1.0  # Default if quantity not provided
                    
                    # Handle price - clean the $ and convert
                    if 'price' in row and pd.notna(row['price']):
                        price_str = str(row['price']).replace('$', '').replace(',', '')
                        try:
                            price = float(price_str)
                        except ValueError:
                            debug_info.append(f"⚠️ Could not parse price: {row['price']}")
                            continue
                    else:
                        price = 0.0
                    
                    # Handle amount - clean the $ and convert
                    if 'amount' in row and pd.notna(row['amount']):
                        amount_str = str(row['amount']).replace(',', '').replace('$', '')
                        try:
                            amount = float(amount_str)
                        except ValueError:
                            debug_info.append(f"⚠️ Could not parse amount: {row['amount']}")
                            continue
                    else:
                        amount = 0.0
                    
                    # If price is 0 but we have amount and quantity, calculate price
                    if price == 0.0 and qty > 0 and amount != 0:
                        price = abs(amount) / qty
                        
                except (ValueError, KeyError, TypeError) as e:
                    debug_info.append(f"⚠️ Skipping transaction for {symbol} - error reading values: {str(e)}")
                    continue
                
                # Initialize position if not exists
                if symbol not in positions:
                    positions[symbol] = {
                        'symbol': symbol,
                        'description': row.get('description', symbol),
                        'quantity': 0.0,  # Explicitly use float
                        'cost_basis': 0.0,  # Explicitly use float
                        'transactions': 0
                    }
                
                # Update position based on transaction type
                trans_type = row['trans_code'].lower()
                if trans_type == 'buy':
                    positions[symbol]['quantity'] += qty
                    positions[symbol]['cost_basis'] += abs(amount)  # Buy amount is negative
                    positions[symbol]['transactions'] += 1
                elif trans_type == 'sell':
                    positions[symbol]['quantity'] -= qty
                    # For sells, reduce cost basis proportionally
                    current_qty = positions[symbol]['quantity'] + qty  # Quantity before sale
                    if current_qty > 0:
                        avg_price = positions[symbol]['cost_basis'] / current_qty
                        positions[symbol]['cost_basis'] -= qty * avg_price
                    positions[symbol]['transactions'] += 1
            
            # Convert to DataFrame
            if positions:
                positions_df = pd.DataFrame(list(positions.values()))
                
                # Filter out positions with zero or negative quantity
                positions_df = positions_df[positions_df['quantity'] > 0]
                debug_info.append(f"📊 Derived {len(positions_df)} positions with positive quantity")
                
                if positions_df.empty:
                    debug_info.append("⚠️ No positions with positive quantity after processing")
                    st.warning("No current positions found in the transaction history")
                    return False
                
                # Ensure all numeric columns are float type to avoid dtype warnings
                positions_df = positions_df.copy()  # Ensure we're working with a copy
                numeric_cols = ['quantity', 'cost_basis', 'transactions']
                for col in numeric_cols:
                    positions_df[col] = pd.to_numeric(positions_df[col], errors='coerce').astype(float)
                
                # Calculate current values
                positions_df['average_price'] = positions_df.apply(
                    lambda x: x['cost_basis'] / x['quantity'] if x['quantity'] > 0 else 0.0, axis=1)
                
                # Placeholder for current price (requires market data)
                positions_df['current_price'] = positions_df['average_price']
                positions_df['market_value'] = positions_df['quantity'] * positions_df['current_price']
                positions_df['gain_loss'] = 0.0
                positions_df['gain_loss_pct'] = 0.0
                
                # Add source information
                positions_df['source'] = file_name
                positions_df['brokerage'] = 'Robinhood'
                positions_df['report_date'] = datetime.now().date()
                debug_info.append(f"🏷️ Added metadata columns")
                
                # Add to portfolio
                debug_info.append(f"📤 Adding {len(positions_df)} derived positions to portfolio")
                self._add_to_portfolio(positions_df)
                
                debug_info.append("✅ Transaction history processed successfully")
                st.warning("Note: Current prices are not available in transaction history. Market values and gains are calculated using cost basis as placeholder.")
                return True
            else:
                debug_info.append("⚠️ No valid positions found after processing transactions")
                st.warning("No valid positions could be derived from the transaction history")
                return False
            
        except Exception as e:
            debug_info.append(f"❌ Error processing transactions: {str(e)}")
            import traceback
            debug_info.append(traceback.format_exc())
            return False

    def parse_generic_csv(self, uploaded_file, brokerage):
        """Parse generic brokerage CSV reports with detailed debugging"""
        debug_info = []
        debug_info.append(f"⏳ Starting {brokerage} CSV parse for {uploaded_file.name}")
        
        try:
            # Read CSV with error handling
            try:
                df = pd.read_csv(uploaded_file)
                debug_info.append(f"✅ CSV read successfully. Shape: {df.shape}")
            except Exception as e:
                debug_info.append(f"❌ CSV read error: {str(e)}")
                st.error(f"Error reading CSV: {str(e)}")
                return False
            
            # Log original columns
            original_columns = df.columns.tolist()
            debug_info.append(f"📋 Original columns: {original_columns}")
            
            # Standardize column names
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            debug_info.append(f"🔄 Standardized columns: {df.columns.tolist()}")
            
            # Map columns to standard format
            column_mapping = {
                'ticker': 'symbol',
                'ticker_symbol': 'symbol',
                'security': 'description',
                'shares': 'quantity',
                'avg_cost': 'average_price',
                'current_value': 'market_value',
                'price': 'current_price',
                'value': 'market_value',
                'current_price': 'current_price'
            }
            
            # Apply mapping
            df.rename(columns=column_mapping, inplace=True)
            debug_info.append(f"🗺️ Applied column mapping: {column_mapping}")
            debug_info.append(f"🆕 Columns after mapping: {df.columns.tolist()}")
            
            # Convert numeric columns to appropriate dtypes to avoid warnings
            numeric_cols = ['quantity', 'average_price', 'current_price', 'market_value']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            
            # Calculate metrics if columns exist
            if 'average_price' in df.columns and 'quantity' in df.columns:
                df['cost_basis'] = df['average_price'] * df['quantity']
                debug_info.append("🧮 Added cost_basis column")
            else:
                debug_info.append("⚠️ Cannot calculate cost_basis - missing columns")
                
            if 'current_price' in df.columns and 'quantity' in df.columns:
                df['market_value'] = df['current_price'] * df['quantity']
                debug_info.append("🧮 Added market_value column")
            elif 'market_value' in df.columns:
                debug_info.append("ℹ️ market_value column already exists")
            else:
                debug_info.append("⚠️ Cannot determine market_value - no price or value columns")
                
            if 'market_value' in df.columns and 'cost_basis' in df.columns:
                df['gain_loss'] = df['market_value'] - df['cost_basis']
                df['gain_loss_pct'] = (df['gain_loss'] / df['cost_basis']) * 100
                debug_info.append("🧮 Added gain/loss columns")
            
            # Add source information
            df['source'] = uploaded_file.name
            df['brokerage'] = brokerage
            df['report_date'] = datetime.now().date()
            debug_info.append(f"🏷️ Added metadata columns")
            
            # Add to portfolio
            debug_info.append(f"📤 Adding to portfolio with columns: {df.columns.tolist()}")
            self._add_to_portfolio(df)
            
            debug_info.append("✅ Parse completed successfully")
            return True
        except Exception as e:
            debug_info.append(f"❌ Parse error: {str(e)}")
            st.error(f"Error parsing {brokerage} CSV: {str(e)}")
            
            # Show debug info in expander
            with st.expander("Debug Information"):
                for entry in debug_info:
                    st.text(entry)
                    
            return False

    def _add_to_portfolio(self, new_df):
        """Add new holdings to portfolio. Replaces data for existing symbols."""
        debug_info = []
        debug_info.append("\n Starting _add_to_portfolio")
        
        if new_df.empty:
            debug_info.append("⚠️ Empty DataFrame received")
            st.warning("No data to add to portfolio")
            # Still update historical values even if new_df is empty, 
            # to potentially record an empty portfolio state if needed.
            self._update_historical_values() 
            return

        # Ensure required columns exist
        required_columns = ['symbol', 'description', 'quantity', 'market_value']
        missing_columns = [col for col in required_columns if col not in new_df.columns]
        if missing_columns:
            debug_info.append(f"❌ Missing required columns: {missing_columns}")
            st.error(f"Missing required columns: {missing_columns}")
            # Update historical with current state (which might be partial or empty)
            self._update_historical_values() 
            return

        new_df = new_df.copy()
        
        # Convert columns to correct types
        numeric_cols = ['quantity', 'market_value', 'cost_basis', 'current_price', 'average_price', 'gain_loss', 'gain_loss_pct']
        for col in numeric_cols:
            if col in new_df.columns:
                new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
        new_df['report_date'] = pd.to_datetime(new_df.get('report_date', pd.NaT), errors='coerce')

        if self.portfolio.empty:
            debug_info.append("🆕 Portfolio is empty - creating new portfolio")
            self.portfolio = new_df
            st.success(f"Added {len(new_df)} new positions to portfolio")
        else:
            updated_positions = 0
            new_positions = 0
            
            # --- NEW LOGIC: Identify symbols in new data ---
            new_symbols = set(new_df['symbol'])
            
            # --- Keep rows for symbols NOT present in the new file ---
            rows_to_keep = self.portfolio[~self.portfolio['symbol'].isin(new_symbols)]
            
            # --- Add/Replace rows from the new file ---
            # new_df already contains the new or updated data for symbols
            # Combine kept rows with new/updated rows
            self.portfolio = pd.concat([rows_to_keep, new_df], ignore_index=True)
            
            # --- Provide feedback ---
            # Count how many were actually new vs. updated
            # This is a simplification; exact count needs tracking during iteration like before,
            # but the outcome (portfolio updated with new_df data) is achieved.
            # For detailed debug, you could iterate, but the concat approach is cleaner for replacement.
            st.success(f"Portfolio updated with data from {len(new_df)} holdings in the uploaded file.")
            
        self._update_historical_values() # Always call after modifying self.portfolio
        debug_info.append("✅ Portfolio updated successfully")
        
        with st.expander("Portfolio Update Details"):
            for entry in debug_info:
                st.text(entry)


    # --- REPLACE the existing _update_historical_values method in Pasted_Text_1753637807170.txt ---
    def _update_historical_values(self):
        """Update historical portfolio values, avoiding duplicate entries for the same day if portfolio state hasn't changed."""
        
        # --- Record Total Portfolio Value ---
        total_value = 0.0
        total_quantity = 0.0
        if not self.portfolio.empty:
            # Ensure market_value is numeric for sum
            self.portfolio['market_value'] = pd.to_numeric(self.portfolio['market_value'], errors='coerce')
            total_value = float(self.portfolio['market_value'].sum())
            
            # Ensure quantity is numeric for sum (optional)
            if 'quantity' in self.portfolio.columns:
                self.portfolio['quantity'] = pd.to_numeric(self.portfolio['quantity'], errors='coerce')
                total_quantity = float(self.portfolio['quantity'].sum())

        today = date.today()
        new_total_entry = pd.DataFrame({
            'date': [today],
            'symbol': ['PORTFOLIO_TOTAL'],
            'market_value': [total_value],
            'quantity': [total_quantity] # Optional
        })

        # --- Record Individual Asset Values ---
        asset_entries = pd.DataFrame() # Initialize as empty
        if not self.portfolio.empty:
            # Ensure 'symbol' and 'market_value' exist
            if 'symbol' in self.portfolio.columns and 'market_value' in self.portfolio.columns:
                # Prepare asset-specific entries
                asset_columns = ['symbol', 'market_value']
                if 'quantity' in self.portfolio.columns:
                    asset_columns.append('quantity')
                
                asset_entries = self.portfolio[asset_columns].copy()
                asset_entries['date'] = today
                
                # Reorder columns to match expected structure if needed
                final_asset_columns = ['date', 'symbol', 'market_value']
                if 'quantity' in asset_entries.columns:
                    final_asset_columns.append('quantity')
                asset_entries = asset_entries[final_asset_columns]
            else:
                print("Warning: Missing 'symbol' or 'market_value' in portfolio for history update.")

        # Combine total and asset entries for the current snapshot
        new_entries = pd.concat([new_total_entry, asset_entries], ignore_index=True) if not asset_entries.empty else new_total_entry

        # --- Append to historical values with Balanced Approach ---
        if self.historical_values.empty:
            self.historical_values = new_entries
        else:
            # --- Balanced Approach Logic ---
            existing_today_total = self.historical_values[
                (self.historical_values['date'] == today) &
                (self.historical_values['symbol'] == 'PORTFOLIO_TOTAL')
            ]

            # Check if the total portfolio value has changed
            if not existing_today_total.empty:
                existing_total_value = existing_today_total['market_value'].iloc[0]
                # Use np.isclose for safer float comparison, with a small tolerance
                if not np.isclose(existing_total_value, total_value, atol=1e-8): 
                    # Total value changed, update the snapshot for today
                    # 1. Remove old snapshot for today
                    self.historical_values = self.historical_values[
                        self.historical_values['date'] != today
                    ]
                    # 2. Append the new snapshot for today
                    self.historical_values = pd.concat([self.historical_values, new_entries], ignore_index=True)
                    # print(f"Updated historical snapshot for {today} (value changed from {existing_total_value} to {total_value})")
                # else: Total value is the same, do nothing (avoid unnecessary replacement/appending)
                #     print(f"Skipped historical snapshot update for {today} (value unchanged at {total_value})")
            else:
                # No entry for today yet, append the new snapshot
                self.historical_values = pd.concat([self.historical_values, new_entries], ignore_index=True)
                # print(f"Added initial historical snapshot for {today}")

        # Optional: Sort by date for cleaner data (though plotting handles sorting)
        # self.historical_values.sort_values(by=['date', 'symbol'], inplace=True)
        # Optional: Reset index
        # self.historical_values.reset_index(drop=True, inplace=True)

    
    def calculate_portfolio_summary(self):
        """Calculate portfolio summary statistics with error handling"""
        if self.portfolio.empty:
            st.warning("Portfolio is empty")
            return None
        
        # Check for required columns
        required_columns = ['market_value', 'cost_basis', 'gain_loss']
        missing_columns = [col for col in required_columns if col not in self.portfolio.columns]
        
        if missing_columns:
            st.error(f"Portfolio missing required columns: {missing_columns}")
            st.error(f"Available columns: {list(self.portfolio.columns)}")
            return None
        
        try:
            # Ensure numeric columns are float type
            for col in required_columns:
                self.portfolio[col] = pd.to_numeric(self.portfolio[col], errors='coerce').astype(float)
            
            summary = {
                'total_value': float(self.portfolio['market_value'].sum()),
                'total_cost_basis': float(self.portfolio['cost_basis'].sum()),
                'total_gain_loss': float(self.portfolio['gain_loss'].sum()), 
                'num_positions': len(self.portfolio),
                'num_brokerages': self.portfolio['brokerage'].nunique() if 'brokerage' in self.portfolio.columns else 1
            }
            
            summary['gain_loss_pct'] = (
                (summary['total_gain_loss'] / summary['total_cost_basis']) * 100 
                if summary['total_cost_basis'] > 0 else 0.0
            )
            
            return summary
            
        except Exception as e:
            st.error(f"Error calculating portfolio summary: {str(e)}")
            st.error(f"Portfolio columns: {list(self.portfolio.columns)}")
            st.error(f"Portfolio shape: {self.portfolio.shape}")
            return None
    
    def calculate_advanced_metrics(self):
        """Calculate advanced portfolio metrics"""
        if self.historical_values.empty or len(self.historical_values) < 2:
            return None
        
        # Calculate returns
        returns = self.historical_values.sort_values('date').copy()
        returns['daily_return'] = returns['total_value'].pct_change()
        
        # Remove first row (no return)
        returns = returns.iloc[1:]
        
        if returns.empty:
            return None
        
        # Calculate metrics
        avg_daily_return = float(returns['daily_return'].mean())
        std_daily_return = float(returns['daily_return'].std())
        total_return = float((returns['total_value'].iloc[-1] / returns['total_value'].iloc[0]) - 1)
        
        # Annualize metrics
        trading_days = 252
        annualized_return = float((1 + avg_daily_return) ** trading_days - 1)
        annualized_volatility = float(std_daily_return * np.sqrt(trading_days))
        
        # Calculate Sharpe ratio
        sharpe_ratio = float((annualized_return - self.risk_free_rate) / annualized_volatility) if annualized_volatility > 0 else 0.0
        
        # Calculate Sortino ratio (only downside volatility)
        downside_returns = returns[returns['daily_return'] < 0]['daily_return']
        if not downside_returns.empty:
            downside_volatility = float(downside_returns.std() * np.sqrt(trading_days))
            sortino_ratio = float((annualized_return - self.risk_free_rate) / downside_volatility) if downside_volatility > 0 else 0.0
        else:
            sortino_ratio = 0.0
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + returns['daily_return']).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = float(drawdown.min())
        
        # Calculate Calmar ratio
        calmar_ratio = float(annualized_return / abs(max_drawdown)) if max_drawdown < 0 else 0.0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'risk_free_rate': self.risk_free_rate
        }
    
    def get_sector_breakdown(self):
        """Group holdings by sector (requires sector data)"""
        if 'sector' in self.portfolio.columns:
            return self.portfolio.groupby('sector').agg({
                'market_value': 'sum',
                'gain_loss': 'sum'
            }).reset_index()
        return None
    
    def get_asset_class_breakdown(self):
        """Group holdings by asset class (stock, ETF, crypto)"""
        if self.portfolio.empty:
            return None
            
        # Simple classifier based on description
        def classify_asset(description):
            if pd.isna(description):
                return 'Unknown'
            desc = description.lower()
            if 'etf' in desc:
                return 'ETF'
            if 'fund' in desc:
                return 'Mutual Fund'
            if 'bitcoin' in desc or 'crypto' in desc or 'coin' in desc:
                return 'Crypto'
            return 'Stock'
        
        self.portfolio['asset_class'] = self.portfolio['description'].apply(classify_asset)
        return self.portfolio.groupby('asset_class').agg({
            'market_value': 'sum',
            'quantity': 'sum',
            'symbol': 'count'
        }).reset_index().rename(columns={'symbol': 'holdings'})
    
    def get_top_performers(self, n=5):
        """Get top performing positions"""
        if self.portfolio.empty:
            return pd.DataFrame()

        # Convert gain_loss_pct to numeric in case it's of object/string dtype
        self.portfolio['gain_loss_pct'] = pd.to_numeric(self.portfolio['gain_loss_pct'], errors='coerce')

        # Drop rows where conversion failed (optional, but safer)
        self.portfolio = self.portfolio.dropna(subset=['gain_loss_pct'])

        return self.portfolio.nlargest(n, 'gain_loss_pct')

    
    def get_bottom_performers(self, n=5):
        """Get bottom performing positions"""
        if self.portfolio.empty:
            return pd.DataFrame()
        return self.portfolio.nsmallest(n, 'gain_loss_pct')


def parse_investment_documents(uploaded_files, doc_type, brokerage):
    """
    Main function to parse investment documents with debugging
    """
    debug_info = []
    debug_info.append(f"🚀 Starting document parsing for {len(uploaded_files)} files")
    debug_info.append(f"Document type: {doc_type}, Brokerage: {brokerage}")
    
    tracker = InvestmentTracker()
    
    if not uploaded_files:
        debug_info.append("⚠️ No files uploaded")
        st.warning("No files uploaded")
        return None
    
    successful_parses = 0
    
    for uploaded_file in uploaded_files:
        file_info = []
        file_info.append(f"\n📄 Processing {uploaded_file.name}...")
        
        try:
            if doc_type == "CSV":
                if brokerage == "Robinhood":
                    file_info.append("🔧 Using Robinhood CSV parser")
                    success = tracker.parse_robinhood_csv(uploaded_file)
                else:
                    file_info.append(f"🔧 Using Generic CSV parser for {brokerage}")
                    success = tracker.parse_generic_csv(uploaded_file, brokerage)
                    
            # elif doc_type == "PDF":
            #     if brokerage == "Robinhood":
            #         file_info.append("🔧 Using Robinhood PDF parser")
            #         success, debug_log = tracker.parse_robinhood_pdf(uploaded_file)
            #         if not success:
            #             file_info.extend(debug_log)
            #     else:
            #         file_info.append(f"⚠️ PDF parsing for {brokerage} not implemented")
            #         success = False
            else:
                file_info.append(f"❌ Unsupported document type: {doc_type}")
                success = False
                
            if success:
                file_info.append("✅ Parse successful")
                successful_parses += 1
                st.success(f"✅ Successfully parsed {uploaded_file.name}")
            else:
                file_info.append("❌ Parse failed")
                st.error(f"❌ Failed to parse {uploaded_file.name}")
                
        except Exception as e:
            file_info.append(f"🔥 Critical error: {str(e)}")
            import traceback
            file_info.append(traceback.format_exc())
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        # Add file info to main debug log
        debug_info.extend(file_info)
    
    debug_info.append(f"\n📊 Parse results: {successful_parses} successful, {len(uploaded_files) - successful_parses} failed")
    
    if successful_parses == 0:
        debug_info.append("❌ No files successfully parsed")
        st.error("No files were successfully parsed")
    else:
        debug_info.append(f"✅ Successfully parsed {successful_parses} files")
        st.success(f"Successfully parsed {successful_parses} out of {len(uploaded_files)} files")
    
    # Validate portfolio data
    if tracker.portfolio.empty and tracker.transactions.empty:
        debug_info.append("❌ Portfolio is empty after parsing")
        st.error("No investment data found in the uploaded files")
        tracker = None
    elif tracker.portfolio.empty and not tracker.transactions.empty:
        debug_info.append("💵 Found cash transactions but no security positions")
        st.warning("Only cash transactions found - no security positions in portfolio")
    else:
        debug_info.append(f"📊 Portfolio contains {len(tracker.portfolio)} positions")
        st.success(f"Portfolio created with {len(tracker.portfolio)} positions")
    
    # Show comprehensive debug info
    with st.expander("Comprehensive Parsing Debug Information"):
        for entry in debug_info:
            st.text(entry)
    
    return tracker