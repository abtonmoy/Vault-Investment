import streamlit as st
from interfaces import investment_tracker
from interfaces import portfolio_optimizer_interface
from interfaces import risk_analysis_interface
import base64


# global theme configuration with the specified color palette
def apply_global_theme():
    st.markdown("""
    <style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Color Variables */
    :root {
        --pakistan-green: #023411;
        --rojo: #DE2C2C;
        --midnight-green: #0C4A57;
        --lapis-lazuli: #15577A;
        --dark-green: #053B2A;
        --light-bg: #f8fafc;
        --card-bg: #ffffff;
        --text-primary: #1a1a1a;
        --text-secondary: #6b7280;
        --border-light: #e5e7eb;
        --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Base styling */
    .stApp {
        background: linear-gradient(135deg, var(--light-bg) 0%, #f1f5f9 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Headers and Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: var(--text-primary);
        letter-spacing: -0.025em;
    }
    
    h1 { font-size: 2.5rem; }
    h2 { font-size: 2rem; }
    h3 { font-size: 1.5rem; }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--card-bg);
        padding: 6px;
        border-radius: 12px;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-light);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        padding: 0 24px;
        background: transparent;
        border-radius: 8px !important;
        border: none !important;
        color: var(--text-secondary);
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(21, 87, 122, 0.1);
        color: var(--lapis-lazuli);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--lapis-lazuli), var(--midnight-green)) !important;
        color: white !important;
        font-weight: 600;
        box-shadow: var(--shadow-sm);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--lapis-lazuli), var(--midnight-green));
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 500;
        padding: 12px 24px;
        font-size: 14px;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--midnight-green), var(--dark-green));
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Download Button Styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--pakistan-green), var(--dark-green));
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 500;
        padding: 12px 24px;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, var(--dark-green), var(--pakistan-green));
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }
    
    /* Form Elements */
    .stSelectbox > div > div {
        background: var(--card-bg);
        border: 2px solid var(--border-light);
        border-radius: 10px;
        transition: all 0.2s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: var(--lapis-lazuli);
        box-shadow: 0 0 0 3px rgba(21, 87, 122, 0.1);
    }
    
    .stTextInput > div > div > input {
        background: var(--card-bg);
        border: 2px solid var(--border-light);
        border-radius: 10px;
        padding: 12px 16px;
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--lapis-lazuli);
        box-shadow: 0 0 0 3px rgba(21, 87, 122, 0.1);
    }
    
    .stNumberInput > div > div > input {
        background: var(--card-bg);
        border: 2px solid var(--border-light);
        border-radius: 10px;
        padding: 12px 16px;
    }
    
    /* Slider Styling */
    .stSlider > div > div > div > div {
        background: var(--lapis-lazuli);
    }
    
    /* File Uploader */
    .stFileUploader > section {
        border: 2px dashed var(--lapis-lazuli);
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(21, 87, 122, 0.05), rgba(12, 74, 87, 0.03));
        padding: 2rem;
        transition: all 0.2s ease;
    }
    
    .stFileUploader > section:hover {
        border-color: var(--midnight-green);
        background: linear-gradient(135deg, rgba(21, 87, 122, 0.08), rgba(12, 74, 87, 0.05));
    }
    
    /* Data Display */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border-light);
        background: var(--card-bg);
    }
    
    /* Metrics */
    .stMetric {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid var(--border-light);
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    
    .stMetric > div > div > div[data-testid="metric-container"] > div[data-testid="metric-container"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    /* Alerts */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: var(--shadow-sm);
    }
    
    .stAlert[data-baseweb="notification"] {
        background: linear-gradient(135deg, rgba(21, 87, 122, 0.1), rgba(12, 74, 87, 0.05));
        border-left: 4px solid var(--lapis-lazuli);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: var(--card-bg);
        border-radius: 12px 12px 0 0;
        border: 1px solid var(--border-light);
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .streamlit-expanderContent {
        background: var(--card-bg);
        border: 1px solid var(--border-light);
        border-top: none;
        border-radius: 0 0 12px 12px;
        box-shadow: var(--shadow-sm);
    }
    
    /* Progress Bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--lapis-lazuli), var(--midnight-green));
        border-radius: 10px;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: var(--card-bg);
        border-right: 1px solid var(--border-light);
    }
    
    /* Custom Card Components */
    .gradient-header {
        background: linear-gradient(135deg, var(--pakistan-green), var(--dark-green));
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
    }
    
    .gradient-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, transparent, rgba(255, 255, 255, 0.1));
        pointer-events: none;
    }
    
    .gradient-header h1,
    .gradient-header h2,
    .gradient-header h3 {
        color: white;
        margin: 0;
        position: relative;
        z-index: 1;
    }
    
    .gradient-header p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        position: relative;
        z-index: 1;
    }
    
    .gradient-card {
        background: linear-gradient(135deg, var(--lapis-lazuli), var(--midnight-green));
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
    }
    
    .gradient-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, transparent, rgba(255, 255, 255, 0.1));
        pointer-events: none;
    }
    
    .gradient-metric {
        background: linear-gradient(135deg, var(--midnight-green), var(--lapis-lazuli));
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: var(--shadow-md);
        transition: all 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    
    .gradient-metric::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, transparent, rgba(255, 255, 255, 0.1));
        pointer-events: none;
    }
    
    .gradient-metric:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-lg);
    }
    
    .error-gradient {
        background: linear-gradient(135deg, var(--rojo), #b91c1c);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: var(--shadow-md);
        margin-bottom: 1.5rem;
    }
    
    .warning-gradient {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: var(--shadow-sm);
        margin-bottom: 1rem;
    }
    
    .settings-gradient {
        background: linear-gradient(135deg, var(--lapis-lazuli), var(--midnight-green));
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-md);
    }
    
    .feature-card {
        background: var(--card-bg);
        padding: 2rem;
        border-radius: 16px;
        border-left: 4px solid var(--lapis-lazuli);
        box-shadow: var(--shadow-md);
        transition: all 0.2s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
        border-left-color: var(--midnight-green);
    }
    
    .feature-card h3 {
        color: var(--lapis-lazuli);
        margin-top: 0;
        margin-bottom: 1rem;
    }
    
    .feature-card p {
        color: var(--text-secondary);
        margin-bottom: 1rem;
        line-height: 1.6;
    }
    
    .feature-card ul {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .feature-card li {
        color: var(--text-secondary);
        padding: 0.25rem 0;
        position: relative;
        padding-left: 1.5rem;
    }
    
    .feature-card li::before {
        content: '✓';
        position: absolute;
        left: 0;
        color: var(--lapis-lazuli);
        font-weight: bold;
    }
    
    .file-info {
        background: var(--card-bg);
        border: 1px solid var(--border-light);
        border-left: 4px solid var(--lapis-lazuli);
        color: var(--text-primary);
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        box-shadow: var(--shadow-sm);
    }
    
    .progress-step {
        background: linear-gradient(135deg, var(--midnight-green), var(--lapis-lazuli));
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        margin: 0 0.5rem 0.5rem 0;
        display: inline-block;
        font-size: 0.9rem;
        font-weight: 500;
        box-shadow: var(--shadow-sm);
    }
    
    /* Data Editor Styling */
    .stDataEditor {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: var(--shadow-md);
    }
    
    /* Checkbox Styling */
    .stCheckbox > label {
        font-weight: 500;
        color: var(--text-primary);
    }
    
    /* Success/Info Messages */
    .stSuccess {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(22, 163, 74, 0.05));
        border-left: 4px solid #22c55e;
        border-radius: 8px;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(21, 87, 122, 0.1), rgba(12, 74, 87, 0.05));
        border-left: 4px solid var(--lapis-lazuli);
        border-radius: 8px;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        .gradient-header,
        .gradient-card {
            padding: 1.5rem;
        }
        
        .feature-card {
            padding: 1.5rem;
        }
        
        h1 { font-size: 2rem; }
        h2 { font-size: 1.75rem; }
        h3 { font-size: 1.25rem; }
    }
    
    /* Animation Classes */
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .slide-up {
        animation: slideUp 0.3s ease-out;
    }
    
    @keyframes slideUp {
        from { transform: translateY(10px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)

def image_to_base64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Vault Investment",
        page_icon="assets\profile.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply global theme
    apply_global_theme()

    # Initialize session state
    if 'transactions_df' not in st.session_state:
        st.session_state['transactions_df'] = None

    # Option 1: Using HTML with flexbox (Recommended)
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
        <img src="data:image/png;base64,{image_to_base64('assets/logo1.png')}" width="60" style="margin: 0;">
        <div>
            <h1 style="color: black; margin: 0;">Vault Investment</h1>
        </div>
    </div>
    <h2 style="color: gray; margin: 1px;"> Portfolio and Risk Analyzer with Optimization</h2>
    <h5 style="color: red; margin: 10px;"> Disclaimer: This tool is for informational purposes only. Don't use it as a financial advisor</h5>
""", unsafe_allow_html=True)
        
    tab1, tab2, tab3 = st.tabs([ 
        "Portfolio Analysis", "Portfolio Optimization", "Risk Analysis"
    ])
    

    with tab1:
        investment_tracker.show_investment_dashboard()
    with tab2:
        portfolio_optimizer_interface.show_portfolio_optimization_tab()
    with tab3:
        risk_analysis_interface.show_risk_analysis_dashboard()
    

if __name__ == "__main__":
    main()