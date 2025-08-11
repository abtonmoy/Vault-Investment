import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def nonlinear_beta_analysis(portfolio_df, historical_values, benchmark_symbol='SPY'):
    """
    Comprehensive nonlinear beta analysis using multiple approaches:
    1. Time-varying rolling beta
    2. Regime-dependent beta (bull/bear markets)  
    3. Polynomial beta (quadratic/cubic relationships)
    4. Conditional beta based on market volatility
    5. Threshold beta models
    """
    if historical_values.empty or portfolio_df.empty:
        return None

    # Get portfolio returns
    portfolio_data = historical_values[
        historical_values['symbol'] == 'PORTFOLIO_TOTAL'
    ].copy()
    
    if portfolio_data.empty or len(portfolio_data) < 60:
        return None

    portfolio_data['date'] = pd.to_datetime(portfolio_data['date'])
    portfolio_data = portfolio_data.sort_values('date')
    portfolio_data['portfolio_returns'] = portfolio_data['market_value'].pct_change()
    
    # Create synthetic benchmark returns (in practice, fetch real market data)
    np.random.seed(42)
    benchmark_returns = np.random.normal(0.0005, 0.012, len(portfolio_data))
    portfolio_data['benchmark_returns'] = benchmark_returns
    
    # Remove NaN values
    clean_data = portfolio_data.dropna()
    
    if len(clean_data) < 30:
        return None

    # 1. TIME-VARYING ROLLING BETA
    rolling_betas = calculate_rolling_beta(clean_data, window=30)
    
    # 2. REGIME-DEPENDENT BETA
    regime_betas = calculate_regime_beta(clean_data)
    
    # 3. POLYNOMIAL BETA (Nonlinear relationship)
    polynomial_beta = calculate_polynomial_beta(clean_data)
    
    # 4. CONDITIONAL BETA (Based on market volatility)
    conditional_beta = calculate_conditional_beta(clean_data)
    
    # 5. THRESHOLD BETA MODEL
    threshold_beta = calculate_threshold_beta(clean_data)
    
    # Create comprehensive visualization
    fig = create_nonlinear_beta_visualization(
        clean_data, rolling_betas, regime_betas, 
        polynomial_beta, conditional_beta, threshold_beta
    )
    
    return fig

def calculate_rolling_beta(data, window=30):
    """Calculate time-varying beta using rolling windows"""
    rolling_betas = []
    
    for i in range(window, len(data)):
        window_data = data.iloc[i-window:i]
        
        # Calculate rolling beta
        covariance = np.cov(window_data['portfolio_returns'], 
                           window_data['benchmark_returns'])[0][1]
        benchmark_variance = np.var(window_data['benchmark_returns'])
        
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
        
        rolling_betas.append({
            'date': window_data['date'].iloc[-1],
            'beta': beta,
            'volatility': window_data['portfolio_returns'].std()
        })
    
    return pd.DataFrame(rolling_betas)

def calculate_regime_beta(data):
    """Calculate beta for different market regimes (bull/bear/neutral)"""
    # Define regimes based on benchmark returns
    data['cumulative_benchmark'] = (1 + data['benchmark_returns']).cumprod()
    data['benchmark_trend'] = data['cumulative_benchmark'].rolling(20).mean()
    
    # Classify regimes
    data['regime'] = 'neutral'
    benchmark_20_change = data['benchmark_trend'].pct_change(20)
    
    data.loc[benchmark_20_change > 0.05, 'regime'] = 'bull'
    data.loc[benchmark_20_change < -0.05, 'regime'] = 'bear'
    
    regime_betas = {}
    for regime in ['bull', 'bear', 'neutral']:
        regime_data = data[data['regime'] == regime]
        
        if len(regime_data) > 10:  # Need sufficient data
            covariance = np.cov(regime_data['portfolio_returns'], 
                               regime_data['benchmark_returns'])[0][1]
            benchmark_variance = np.var(regime_data['benchmark_returns'])
            
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
            regime_betas[regime] = {
                'beta': beta,
                'observations': len(regime_data),
                'avg_return': regime_data['portfolio_returns'].mean(),
                'volatility': regime_data['portfolio_returns'].std()
            }
    
    return regime_betas

def calculate_polynomial_beta(data, degree=3):
    """Calculate polynomial (nonlinear) beta relationship with better detection"""
    X = data['benchmark_returns'].values.reshape(-1, 1)
    y = data['portfolio_returns'].values
    
    # Remove extreme outliers that might skew the relationship
    from scipy import stats
    z_scores = np.abs(stats.zscore(np.column_stack([X.flatten(), y])))
    mask = (z_scores < 3).all(axis=1)
    X_clean = X[mask]
    y_clean = y[mask]
    
    if len(X_clean) < 20:  # Fall back to original data if too much removed
        X_clean, y_clean = X, y
    
    # Create polynomial features up to specified degree
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X_clean)
    
    # Try different models and pick the best one
    models = [
        ('Linear', LinearRegression()),
        ('Ridge_low', Ridge(alpha=0.001)),  # Lower regularization
        ('Ridge_med', Ridge(alpha=0.01)),
        ('Ridge_high', Ridge(alpha=0.1))
    ]
    
    best_model = None
    best_score = -np.inf
    best_name = 'Linear'
    
    for name, model in models:
        try:
            model.fit(X_poly, y_clean)
            score = model.score(X_poly, y_clean)
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
        except:
            continue
    
    if best_model is None:
        return None
    
    # Generate predictions for visualization
    X_range = np.linspace(X_clean.min(), X_clean.max(), 200).reshape(-1, 1)
    X_range_poly = poly_features.transform(X_range)
    y_pred = best_model.predict(X_range_poly)
    
    # Calculate nonlinearity measure
    # Compare polynomial model to simple linear model
    linear_model = LinearRegression()
    linear_model.fit(X_clean, y_clean)
    linear_pred = linear_model.predict(X_range)
    
    # Measure how much the polynomial deviates from linear
    nonlinearity_score = np.std(y_pred - linear_pred) / np.std(y_clean)
    
    # Test for significant nonlinear terms
    linear_r2 = linear_model.score(X_clean, y_clean)
    poly_r2 = best_model.score(X_poly, y_clean)
    r2_improvement = poly_r2 - linear_r2
    
    return {
        'coefficients': best_model.coef_,
        'X_range': X_range.flatten(),
        'y_pred': y_pred,
        'linear_pred': linear_pred,  # Add linear comparison
        'r_squared': poly_r2,
        'linear_r2': linear_r2,
        'r2_improvement': r2_improvement,
        'nonlinearity_score': nonlinearity_score,
        'linear_beta': best_model.coef_[0],
        'nonlinear_terms': best_model.coef_[1:] if len(best_model.coef_) > 1 else [],
        'best_model_type': best_name,
        'degree': degree,
        'significant_nonlinearity': r2_improvement > 0.01 and nonlinearity_score > 0.1
    }

def create_realistic_benchmark_returns(portfolio_data):
    """Create more realistic benchmark returns that might show nonlinear relationships"""
    np.random.seed(42)
    n = len(portfolio_data)
    
    # Create market returns with some autocorrelation and volatility clustering
    market_returns = np.zeros(n)
    volatility = np.zeros(n)
    
    # Initial values
    volatility[0] = 0.015
    market_returns[0] = np.random.normal(0, volatility[0])
    
    # GARCH-like process for volatility clustering
    for i in range(1, n):
        # Volatility clustering
        volatility[i] = 0.01 + 0.05 * market_returns[i-1]**2 + 0.9 * volatility[i-1]
        volatility[i] = np.clip(volatility[i], 0.005, 0.05)  # Keep reasonable bounds
        
        # Autocorrelated returns
        market_returns[i] = (0.02 * market_returns[i-1] + 
                           np.random.normal(0.0005, volatility[i]))
    
    return market_returns

def calculate_conditional_beta(data, volatility_window=20):
    """Calculate beta conditional on market volatility levels"""
    # Calculate rolling benchmark volatility
    data['benchmark_volatility'] = data['benchmark_returns'].rolling(
        volatility_window
    ).std() * np.sqrt(252)  # Annualized
    
    # Define volatility regimes
    vol_33rd = data['benchmark_volatility'].quantile(0.33)
    vol_67th = data['benchmark_volatility'].quantile(0.67)
    
    data['vol_regime'] = 'medium'
    data.loc[data['benchmark_volatility'] <= vol_33rd, 'vol_regime'] = 'low'
    data.loc[data['benchmark_volatility'] >= vol_67th, 'vol_regime'] = 'high'
    
    conditional_betas = {}
    for vol_regime in ['low', 'medium', 'high']:
        regime_data = data[data['vol_regime'] == vol_regime].dropna()
        
        if len(regime_data) > 10:
            covariance = np.cov(regime_data['portfolio_returns'], 
                               regime_data['benchmark_returns'])[0][1]
            benchmark_variance = np.var(regime_data['benchmark_returns'])
            
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
            conditional_betas[vol_regime] = {
                'beta': beta,
                'avg_volatility': regime_data['benchmark_volatility'].mean(),
                'observations': len(regime_data)
            }
    
    return conditional_betas

def calculate_threshold_beta(data, threshold_percentile=50):
    """Calculate threshold beta model (different betas above/below threshold)"""
    # Use median benchmark return as threshold
    threshold = data['benchmark_returns'].quantile(threshold_percentile / 100)
    
    # Split data
    above_threshold = data[data['benchmark_returns'] > threshold]
    below_threshold = data[data['benchmark_returns'] <= threshold]
    
    threshold_betas = {}
    
    for name, subset in [('above', above_threshold), ('below', below_threshold)]:
        if len(subset) > 5:
            covariance = np.cov(subset['portfolio_returns'], 
                               subset['benchmark_returns'])[0][1]
            benchmark_variance = np.var(subset['benchmark_returns'])
            
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
            threshold_betas[name] = {
                'beta': beta,
                'threshold': threshold,
                'observations': len(subset),
                'avg_benchmark_return': subset['benchmark_returns'].mean()
            }
    
    return threshold_betas

def create_polynomial_comparison_plot(data, polynomial_beta):
    """Create a plot showing linear vs polynomial fit"""
    if not polynomial_beta:
        return None
        
    fig = go.Figure()
    
    # Scatter plot of actual data
    fig.add_trace(
        go.Scatter(
            x=data['benchmark_returns'],
            y=data['portfolio_returns'],
            mode='markers',
            name='Actual Data',
            marker=dict(
                color='rgba(52,152,219,0.6)', 
                size=6,
                opacity=0.7
            ),
            hovertemplate="<b>Returns</b><br>Benchmark: %{x:.4f}<br>Portfolio: %{y:.4f}<extra></extra>"
        )
    )
    
    # Linear fit line
    fig.add_trace(
        go.Scatter(
            x=polynomial_beta['X_range'],
            y=polynomial_beta['linear_pred'],
            mode='lines',
            name=f'Linear Fit (R²={polynomial_beta["linear_r2"]:.3f})',
            line=dict(color='#95a5a6', width=2, dash='dash'),
            hovertemplate="<b>Linear Fit</b><br>Benchmark: %{x:.4f}<br>Predicted: %{y:.4f}<extra></extra>"
        )
    )
    
    # Polynomial fit line
    line_color = '#e74c3c' if polynomial_beta['significant_nonlinearity'] else '#f39c12'
    line_width = 3 if polynomial_beta['significant_nonlinearity'] else 2
    
    fig.add_trace(
        go.Scatter(
            x=polynomial_beta['X_range'],
            y=polynomial_beta['y_pred'],
            mode='lines',
            name=f'Polynomial Fit (Degree {polynomial_beta["degree"]}, R²={polynomial_beta["r_squared"]:.3f})',
            line=dict(color=line_color, width=line_width),
            hovertemplate="<b>Polynomial Fit</b><br>Benchmark: %{x:.4f}<br>Predicted: %{y:.4f}<extra></extra>"
        )
    )
    
    # Add annotation about nonlinearity
    significance_text = "Significant" if polynomial_beta['significant_nonlinearity'] else "Minimal"
    
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"Nonlinearity: {significance_text}<br>R² Improvement: {polynomial_beta['r2_improvement']:.4f}<br>Nonlinearity Score: {polynomial_beta['nonlinearity_score']:.3f}",
        showarrow=False,
        bgcolor="rgba(0,0,0,0.7)",
        bordercolor="white",
        borderwidth=1,
        font=dict(color="white", size=12),
        align="left"
    )
    
    fig.update_layout(
        title='Polynomial vs Linear Beta Relationship',
        xaxis_title='Benchmark Returns',
        yaxis_title='Portfolio Returns',
        paper_bgcolor='#2c3e50',
        plot_bgcolor='#34495e',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="white",
            borderwidth=1
        )
    )
    
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.2)",
        tickfont=dict(color='white')
    )
    
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.2)",
        tickfont=dict(color='white')
    )
    
    return fig


def create_nonlinear_beta_visualization(data, rolling_betas, regime_betas, 
                                       polynomial_beta, conditional_beta, threshold_beta):
    """Create comprehensive nonlinear beta visualization"""
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Time-Varying Rolling Beta',
            'Regime-Dependent Beta',
            'Polynomial Beta Relationship', 
            'Conditional Beta (by Volatility)',
            'Threshold Beta Model',
            'Beta Summary Statistics'
        ],
        specs=[[{"secondary_y": False}, {"type": "bar"}],
               [{"secondary_y": False}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "table"}]],
        vertical_spacing=0.12
    )
    
    # 1. Time-varying rolling beta
    if not rolling_betas.empty:
        fig.add_trace(
            go.Scatter(
                x=rolling_betas['date'],
                y=rolling_betas['beta'],
                mode='lines',
                name='Rolling Beta',
                line=dict(color='#3498db', width=2),
                hovertemplate="<b>Rolling Beta</b><br>Date: %{x}<br>Beta: %{y:.3f}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Add beta = 1 reference line
        fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(255,255,255,0.5)",
                     row=1, col=1)
    
    # 2. Regime-dependent beta
    if regime_betas:
        regimes = list(regime_betas.keys())
        regime_beta_values = [regime_betas[r]['beta'] for r in regimes]
        regime_colors = ['#e74c3c', '#f39c12', '#2ecc71']  # Red, Orange, Green
        
        fig.add_trace(
            go.Bar(
                x=regimes,
                y=regime_beta_values,
                name='Regime Beta',
                marker_color=regime_colors,
                hovertemplate="<b>%{x} Market</b><br>Beta: %{y:.3f}<extra></extra>"
            ),
            row=1, col=2
        )
    
    # 3. Polynomial beta relationship - UPDATED VERSION
    if polynomial_beta:
        # Scatter plot of actual data
        fig.add_trace(
            go.Scatter(
                x=data['benchmark_returns'],
                y=data['portfolio_returns'],
                mode='markers',
                name='Actual Returns',
                marker=dict(color='rgba(52,152,219,0.6)', size=4, opacity=0.7),
                hovertemplate="<b>Return Relationship</b><br>Benchmark: %{x:.3f}<br>Portfolio: %{y:.3f}<extra></extra>"
            ),
            row=2, col=1
        )
        
        # Linear fit line (reference)
        if 'linear_pred' in polynomial_beta:
            fig.add_trace(
                go.Scatter(
                    x=polynomial_beta['X_range'],
                    y=polynomial_beta['linear_pred'],
                    mode='lines',
                    name=f'Linear (R²={polynomial_beta["linear_r2"]:.3f})',
                    line=dict(color='#95a5a6', width=2, dash='dash'),
                    opacity=0.7,
                    hovertemplate="<b>Linear Fit</b><br>Benchmark: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>"
                ),
                row=2, col=1
            )
        
        # Polynomial fit line
        line_color = '#e74c3c' if polynomial_beta.get('significant_nonlinearity', False) else '#f39c12'
        line_width = 3 if polynomial_beta.get('significant_nonlinearity', False) else 2
        
        nonlinearity_text = ""
        if 'significant_nonlinearity' in polynomial_beta:
            nonlinearity_text = " (Nonlinear)" if polynomial_beta['significant_nonlinearity'] else " (Linear)"
        
        fig.add_trace(
            go.Scatter(
                x=polynomial_beta['X_range'],
                y=polynomial_beta['y_pred'],
                mode='lines',
                name=f'Polynomial (R²={polynomial_beta["r_squared"]:.3f}){nonlinearity_text}',
                line=dict(color=line_color, width=line_width),
                hovertemplate="<b>Polynomial Fit</b><br>Benchmark: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>"
            ),
            row=2, col=1
        )
    
    # 4. Conditional beta by volatility
    if conditional_beta:
        vol_regimes = list(conditional_beta.keys())
        vol_beta_values = [conditional_beta[r]['beta'] for r in vol_regimes]
        vol_colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
        
        fig.add_trace(
            go.Bar(
                x=vol_regimes,
                y=vol_beta_values,
                name='Volatility Conditional Beta',
                marker_color=vol_colors,
                hovertemplate="<b>%{x} Volatility</b><br>Beta: %{y:.3f}<extra></extra>"
            ),
            row=2, col=2
        )
    
    # 5. Threshold beta model
    if threshold_beta:
        threshold_regimes = list(threshold_beta.keys())
        threshold_beta_values = [threshold_beta[r]['beta'] for r in threshold_regimes]
        threshold_colors = ['#e74c3c', '#2ecc71']  # Red for below, Green for above
        
        fig.add_trace(
            go.Bar(
                x=[f"{r.title()} Threshold" for r in threshold_regimes],
                y=threshold_beta_values,
                name='Threshold Beta',
                marker_color=threshold_colors,
                hovertemplate="<b>%{x}</b><br>Beta: %{y:.3f}<extra></extra>"
            ),
            row=3, col=1
        )
    
    # 6. Summary statistics table
    summary_data = prepare_summary_table(rolling_betas, regime_betas, 
                                        polynomial_beta, conditional_beta, threshold_beta)
    
    if summary_data:
        fig.add_trace(
            go.Table(
                header=dict(values=['Model', 'Beta Range', 'Key Insight'],
                          fill_color='#34495e',
                          font=dict(color='white', size=12)),
                cells=dict(values=[summary_data['model'], 
                                 summary_data['beta_range'],
                                 summary_data['insight']],
                          fill_color='#2c3e50',
                          font=dict(color='white', size=11))
            ),
            row=3, col=2
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Nonlinear Beta Analysis: Multiple Modeling Approaches',
            font=dict(size=20, color='white'),
            x=0.5
        ),
        paper_bgcolor='#2c3e50',
        plot_bgcolor='#34495e',
        font=dict(color='white'),
        height=900,
        showlegend=False
    )
    
    # Update axes
    fig.update_xaxes(
        title_text='Date',
        title_font=dict(color='white'),
        tickfont=dict(color='white'),
        gridcolor="rgba(255,255,255,0.2)",
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text='Beta',
        title_font=dict(color='white'),
        tickfont=dict(color='white'),
        gridcolor="rgba(255,255,255,0.2)",
        row=1, col=1
    )
    
    fig.update_xaxes(
        title_text='Benchmark Returns',
        title_font=dict(color='white'),
        tickfont=dict(color='white'),
        gridcolor="rgba(255,255,255,0.2)",
        row=2, col=1
    )
    
    fig.update_yaxes(
        title_text='Portfolio Returns',
        title_font=dict(color='white'),
        tickfont=dict(color='white'),
        gridcolor="rgba(255,255,255,0.2)",
        row=2, col=1
    )
    
    return fig

def prepare_summary_table(rolling_betas, regime_betas, polynomial_beta, 
                         conditional_beta, threshold_beta):
    """Prepare summary statistics for the table"""
    summary = {
        'model': [],
        'beta_range': [],
        'insight': []
    }
    
    # Rolling beta summary
    if not rolling_betas.empty:
        beta_min, beta_max = rolling_betas['beta'].min(), rolling_betas['beta'].max()
        summary['model'].append('Time-Varying')
        summary['beta_range'].append(f'{beta_min:.2f} - {beta_max:.2f}')
        summary['insight'].append('Beta changes over time')
    
    # Regime beta summary
    if regime_betas:
        regime_values = [v['beta'] for v in regime_betas.values()]
        summary['model'].append('Regime-Dependent')
        summary['beta_range'].append(f'{min(regime_values):.2f} - {max(regime_values):.2f}')
        summary['insight'].append('Different risk in bull/bear markets')
    
    # Polynomial beta
    if polynomial_beta:
        linear_beta = polynomial_beta['linear_beta']
        summary['model'].append('Polynomial')
        summary['beta_range'].append(f'Linear: {linear_beta:.2f}')
        summary['insight'].append(f'R² = {polynomial_beta["r_squared"]:.3f}')
    
    # Conditional beta
    if conditional_beta:
        cond_values = [v['beta'] for v in conditional_beta.values()]
        summary['model'].append('Volatility-Conditional')
        summary['beta_range'].append(f'{min(cond_values):.2f} - {max(cond_values):.2f}')
        summary['insight'].append('Beta varies with market volatility')
    
    # Threshold beta
    if threshold_beta and len(threshold_beta) == 2:
        thresh_values = [v['beta'] for v in threshold_beta.values()]
        summary['model'].append('Threshold')
        summary['beta_range'].append(f'{min(thresh_values):.2f} - {max(thresh_values):.2f}')
        summary['insight'].append('Asymmetric response to market moves')
    
    return summary if summary['model'] else None

# Integration function for your existing beta_analysis_chart
def nonlinear_beta_analysis_chart(portfolio_df, historical_values, benchmark_symbol='SPY'):
    """beta analysis with nonlinear modeling - replaces existing function"""
    return nonlinear_beta_analysis(portfolio_df, historical_values, benchmark_symbol)