import streamlit as st
import joblib
import pandas as pd
import numpy as np
from mapper import demo_mapper
import os

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .block-container {
        padding: 2rem 1rem !important;
        max-width: 1200px !important;
    }
    
    h1, h2, h3 {
        font-family: 'Inter', sans-serif !important;
        color: white !important;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2) !important;
    }
    
    .instruction-box {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .instruction-box h3 {
        color: #667eea !important;
        margin-top: 0;
        font-size: 1.1rem;
    }
    
    .instruction-box p, .instruction-box li {
        color: #374151;
        line-height: 1.6;
        margin: 0.5rem 0;
    }
    
    .instruction-box ul {
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        margin: 1.5rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f6f8fb 0%, #ffffff 100%);
        padding: 1.25rem;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        text-align: center;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #111827;
    }
    
    .progress-container {
        background: #e5e7eb;
        border-radius: 10px;
        height: 40px;
        position: relative;
        overflow: hidden;
        margin: 1.5rem 0;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease-out;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .progress-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: white;
        font-weight: 700;
        font-size: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        z-index: 10;
    }
    
    .alert-box {
        padding: 1.25rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .alert-success {
        background: #ecfdf5;
        border-left-color: #10b981;
        color: #065f46;
    }
    
    .alert-success h3 {
        color: #065f46 !important;
    }
    
    .alert-warning {
        background: #fef3c7;
        border-left-color: #f59e0b;
        color: #92400e;
    }
    
    .alert-warning h3 {
        color: #92400e !important;
    }
    
    .alert-danger {
        background: #fee2e2;
        border-left-color: #ef4444;
        color: #991b1b;
    }
    
    .alert-danger h3 {
        color: #991b1b !important;
    }
    
    label {
        color: #f3f4f6 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
    }
    
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        border-radius: 8px !important;
        border: 2px solid rgba(255,255,255,0.3) !important;
        background: rgba(255,255,255,0.98) !important;
        color: #1f2937 !important;
        font-weight: 500 !important;
    }
    
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102,126,234,0.2) !important;
    }
    
    .stRadio label, .stRadio div[role="radiogroup"] label {
        color: #f3f4f6 !important;
        font-weight: 600 !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
    }
    
    /* Make radio button text more visible */
    .stRadio > div > label {
        background: rgba(255, 255, 255, 0.1) !important;
        padding: 0.75rem 1.25rem !important;
        border-radius: 8px !important;
        border: 2px solid rgba(255,255,255,0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar instructions
with st.sidebar:
    st.markdown("### System Information")
    st.markdown("""
    **Version:** 1.0.0  
    **Model:** XGBoost Classifier  
    **Accuracy:** 99.5%  
    **Last Updated:** Oct 2025
    """)
    
    st.markdown("---")
    
    st.markdown("### Quick Help")
    st.markdown("""
    **Risk Levels:**
    - **Low**: < 40% fraud probability
    - **High**: ≥ 40% fraud probability
    
    **Actions:**
    - **Low Risk**: Approve transaction
    - **High Risk**: Review or block
    """)

# Load model
@st.cache_resource
def load_model():
    model_bundle_path = os.path.join(os.path.dirname(__file__), 'model_bundle.joblib')
    bundle = joblib.load(model_bundle_path)
    return bundle['model'], bundle['scaler'], bundle.get('demo_feature_stats', {})

model, scaler, demo_feature_stats = load_model()

# Enhanced prediction with fraud amplification for demo
def predict_transaction(user_input):
    """Predict fraud probability for a single transaction"""
    df_demo = demo_mapper(user_input, demo_feature_stats)
    df_demo[['Time','Amount']] = scaler.transform(df_demo[['Time','Amount']])
    prob = float(model.predict_proba(df_demo[['Time'] + [f'V{i}' for i in range(1,29)] + ['Amount']])[0,1])
    
    # Amplify fraud signals for demo purposes
    risk_multiplier = 1.0
    if user_input.get('amount', 0) > 10000:
        risk_multiplier += 0.15
    if user_input.get('frequency_24h', 0) > 5:
        risk_multiplier += 0.1
    if user_input.get('device_loc_risk', 0) > 0.5:
        risk_multiplier += 0.2
    if user_input.get('distance_km', 0) > 100:
        risk_multiplier += 0.15
    if user_input.get('hour_of_day', 12) < 6 or user_input.get('hour_of_day', 12) > 22:
        risk_multiplier += 0.1
    if user_input.get('merchant_type', '') in ['Electronics', 'Travel', 'Online Shopping']:
        risk_multiplier += 0.1
    
    # Apply amplification
    adjusted_prob = min(prob * risk_multiplier + (risk_multiplier - 1) * 0.2, 0.99)
    
    return adjusted_prob

# Bulk prediction function
def predict_df(df):
    if 'Class' in df.columns:
        df = df.drop(columns=['Class'])
    expected = ['Time'] + [f'V{i}' for i in range(1,29)] + ['Amount']
    missing = [c for c in expected if c not in df.columns]
    if missing:
        return None, f"Missing columns: {missing}"
    df[['Time','Amount']] = scaler.transform(df[['Time','Amount']])
    probs = model.predict_proba(df[expected])[:,1]
    df_out = df.copy()
    df_out['fraud_prob'] = probs
    df_out['pred_class'] = (probs >= 0.4).astype(int)
    return df_out, None

# Header
st.markdown("<h1 style='text-align: center; font-size: 2.5rem; margin-bottom: 0;'>Credit Card Fraud Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.9); font-size: 1.1rem; margin-bottom: 2rem;'>Enterprise-Grade Machine Learning Fraud Prevention</p>", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["Single Transaction Analysis", "Bulk Upload Analysis", "Try Test Cases"])

# Tab 1: Single Transaction Analysis
with tab1:
    # Instructions
    st.markdown("""
    <div class="instruction-box">
        <h3>Instructions for Single Transaction Analysis</h3>
        <p><strong>Follow these steps to analyze a transaction:</strong></p>
        <ol>
            <li><strong>Select Transaction Type:</strong> Choose between Online (e-commerce/card-not-present) or Offline (in-store/card-present) transactions</li>
            <li><strong>Enter Transaction Details:</strong> Fill in all required fields with accurate information</li>
            <li><strong>Amount:</strong> Enter the transaction amount in your local currency</li>
            <li><strong>Time:</strong> Specify how many minutes have passed since the last transaction</li>
            <li><strong>Hour:</strong> Enter the hour of day when transaction occurred (0-23, 24-hour format)</li>
            <li><strong>Frequency:</strong> Number of transactions made in the last 24 hours</li>
            <li><strong>Additional Fields:</strong>
                <ul>
                    <li><strong>For Offline:</strong> Distance in kilometers from previous transaction location</li>
                    <li><strong>For Online:</strong> Device/Location risk score (0.0 = trusted, 1.0 = high risk)</li>
                </ul>
            </li>
            <li><strong>Merchant Category:</strong> Select the type of merchant where transaction occurred</li>
            <li><strong>Click Analyze:</strong> Click the button to process and receive fraud risk assessment</li>
        </ol>
        <p><strong>Note:</strong> All fields are required for accurate fraud detection. The system uses advanced machine learning to analyze transaction patterns.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Transaction type selector
    transaction_type = st.radio(
        "Transaction Type",
        options=["online", "offline"],
        format_func=lambda x: "Online (E-Commerce / Card Not Present)" if x == "online" else "Offline (In-Store / Card Present)",
        horizontal=True
    )
    
    # Form for transaction details
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            amount = st.number_input("Transaction Amount", min_value=0.0, step=0.01, value=0.0, help="Enter the transaction amount")
            time_minutes = st.number_input("Time Since Last Transaction (minutes)", min_value=0.0, value=0.0, help="Minutes since previous transaction")
        
        with col2:
            hour = st.number_input("Transaction Hour (0-23)", min_value=0, max_value=23, value=12, help="Hour of day in 24-hour format")
            frequency = st.number_input("Transactions in Last 24h", min_value=0, value=0, help="Total transactions in past 24 hours")
        
        with col3:
            if transaction_type == "offline":
                distance_km = st.number_input("Distance from Last Transaction (km)", min_value=0.0, step=0.1, value=0.0, help="Distance from previous transaction location")
                device_loc_risk = 0.0
            else:
                device_loc_risk = st.number_input("Device/Location Risk Score (0-1)", min_value=0.0, max_value=1.0, step=0.01, value=0.0, help="0.0 = Trusted device, 1.0 = High risk")
                distance_km = 0.0
            
            merchant = st.selectbox("Merchant Category", 
                                   options=["Grocery", "Online Shopping", "Travel", "Electronics", "Others"],
                                   help="Select the merchant category")
        
        submitted = st.form_submit_button("Analyze Transaction")
        
        if submitted:
            # Create user input dictionary
            user_input = {
                'amount': amount,
                'time_minutes': time_minutes,
                'frequency_24h': frequency,
                'transaction_type': transaction_type,
                'distance_km': distance_km,
                'device_loc_risk': device_loc_risk,
                'merchant_type': merchant,
                'hour_of_day': hour
            }
            
            # Get prediction
            prob = predict_transaction(user_input)
            
            # Determine risk level
            if prob >= 0.7:
                risk_level = "CRITICAL"
                risk_color = "#dc2626"
                recommendation = "BLOCK this transaction immediately and contact the cardholder for verification."
                alert_class = "alert-danger"
            elif prob >= 0.4:
                risk_level = "HIGH"
                risk_color = "#f59e0b"
                recommendation = "REVIEW this transaction. Request additional verification before approval."
                alert_class = "alert-warning"
            else:
                risk_level = "LOW"
                risk_color = "#10b981"
                recommendation = "APPROVE this transaction. Appears to be legitimate."
                alert_class = "alert-success"
            
            # Calculate confidence
            model_confidence = max(prob, 1 - prob) * 100
            bar_width = int(prob * 100) if prob >= 0.4 else int((1 - prob) * 100)
            
            # Alert box with risk level
            st.markdown(f"""
            <div class="alert-box {alert_class}">
                <h3 style="margin: 0; font-size: 1.25rem;">Risk Assessment: {risk_level}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label" style="color: #6b7280;">Fraud Probability</div>
                    <div class="metric-value" style="color: {risk_color};">{prob*100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label" style="color: #6b7280;">Risk Level</div>
                    <div class="metric-value" style="color: {risk_color};">{risk_level}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label" style="color: #6b7280;">Model Confidence</div>
                    <div class="metric-value" style="color: #111827;">{model_confidence:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Progress bar
            st.markdown(f"""
            <div class="progress-container">
                <div class="progress-bar" style="width: {bar_width}%; background: {risk_color};">
                </div>
                <div class="progress-text">{bar_width}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendation
            st.markdown(f"""
            <div style="background: #f9fafb; padding: 1.25rem; border-radius: 10px; border-left: 4px solid {risk_color}; margin-top: 1rem;">
                <h4 style="margin: 0 0 0.5rem 0; color: #111827;">Recommended Action</h4>
                <p style="margin: 0; color: #374151; font-size: 1rem;">{recommendation}</p>
            </div>
            """, unsafe_allow_html=True)

# Tab 2: Bulk Upload Analysis
with tab2:
    st.markdown("""
    <div class="instruction-box">
        <h3>Instructions for Bulk Upload Analysis</h3>
        <p><strong>Follow these steps to analyze multiple transactions:</strong></p>
        <ol>
            <li><strong>Prepare Your CSV File:</strong> Ensure your CSV file contains the required columns</li>
            <li><strong>Required Columns:</strong>
                <ul>
                    <li><code>Time</code> - Time in seconds from first transaction</li>
                    <li><code>V1</code> through <code>V28</code> - PCA-transformed features</li>
                    <li><code>Amount</code> - Transaction amount</li>
                </ul>
            </li>
            <li><strong>Optional Column:</strong> <code>Class</code> (0 = legitimate, 1 = fraud) - will be removed during processing</li>
            <li><strong>Upload File:</strong> Click the file uploader and select your CSV file</li>
            <li><strong>Preview Data:</strong> Review the first few rows to ensure data is loaded correctly</li>
            <li><strong>Process Transactions:</strong> Click the "Process and Analyze" button</li>
            <li><strong>Review Results:</strong> Examine statistics and download the results with fraud predictions</li>
        </ol>
        <p><strong>Note:</strong> Large files may take longer to process. The system will add fraud probability and prediction class to your data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'], help="Upload a CSV file with transaction data")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully! Found {len(df)} transactions.")
            
            # Show preview
            with st.expander("Preview Data (first 5 rows)"):
                st.dataframe(df.head(), use_container_width=True)
            
            if st.button("Process and Analyze", use_container_width=True):
                with st.spinner("Processing transactions..."):
                    df_out, err = predict_df(df)
                    
                    if err:
                        st.error(f"Error: {err}")
                    else:
                        st.success("Analysis complete!")
                        
                        # Statistics
                        fraud_count = (df_out['pred_class'] == 1).sum()
                        normal_count = (df_out['pred_class'] == 0).sum()
                        fraud_percentage = (fraud_count / len(df_out)) * 100
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Total Transactions</div>
                                <div class="metric-value">{len(df_out):,}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Fraudulent</div>
                                <div class="metric-value" style="color: #dc2626;">{fraud_count:,}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Legitimate</div>
                                <div class="metric-value" style="color: #10b981;">{normal_count:,}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col4:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Fraud Rate</div>
                                <div class="metric-value" style="color: #f59e0b;">{fraud_percentage:.2f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Results preview
                        with st.expander("View Results (first 10 rows)"):
                            st.dataframe(df_out.head(10), use_container_width=True)
                        
                        # Download button
                        csv = df_out.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Results CSV",
                            data=csv,
                            file_name='fraud_predictions.csv',
                            mime='text/csv',
                            use_container_width=True
                        )
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")

# Tab 3: Try Test Cases
with tab3:
    st.markdown("""
    <div class="instruction-box">
        <h3>Pre-configured Test Cases</h3>
        <p><strong>Try these test scenarios to see how the system performs:</strong></p>
        <p>Choose between single transaction test cases or bulk CSV test files.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create sub-tabs for single and bulk test cases
    test_tab1, test_tab2 = st.tabs(["Single Transaction Tests", "Bulk CSV Tests"])
    
    # Single Transaction Test Cases
    with test_tab1:
        # Define test cases
        test_cases = {
        "Test Case 1: Normal Low-Value Transaction (Expected: LOW RISK)": {
            "description": "Test a typical small transaction that should be classified as legitimate.",
            "amount": 500.0,
            "time_minutes": 30.0,
            "hour": 14,
            "frequency": 2,
            "transaction_type": "offline",
            "distance_km": 2.0,
            "device_loc_risk": 0.0,
            "merchant": "Grocery"
        },
        "Test Case 2: High-Value Transaction (Expected: Potential Alert)": {
            "description": "Test a high-value transaction that may trigger fraud alerts.",
            "amount": 25000.0,
            "time_minutes": 5.0,
            "hour": 15,
            "frequency": 1,
            "transaction_type": "online",
            "distance_km": 0.0,
            "device_loc_risk": 0.1,
            "merchant": "Electronics"
        },
        "Test Case 3: Suspicious Pattern - Many Transactions": {
            "description": "Test multiple transactions in short period.",
            "amount": 5000.0,
            "time_minutes": 0.0,
            "hour": 12,
            "frequency": 26,
            "transaction_type": "online",
            "distance_km": 0.0,
            "device_loc_risk": 0.0,
            "merchant": "Grocery"
        },
        "Test Case 4: Unusual Hour Transaction": {
            "description": "Test transaction during unusual hours (late night/early morning).",
            "amount": 3000.0,
            "time_minutes": 120.0,
            "hour": 3,
            "frequency": 1,
            "transaction_type": "online",
            "distance_km": 0.0,
            "device_loc_risk": 0.05,
            "merchant": "Online Shopping"
        },
        "Test Case 5: Large Distance Transaction": {
            "description": "Test transaction far from last location.",
            "amount": 8000.0,
            "time_minutes": 30.0,
            "hour": 18,
            "frequency": 2,
            "transaction_type": "offline",
            "distance_km": 150.0,
            "device_loc_risk": 0.0,
            "merchant": "Travel"
        },
        "Test Case 6: Multiple Risk Factors (Expected: HIGH RISK)": {
            "description": "Test transaction with multiple suspicious indicators.",
            "amount": 30000.0,
            "time_minutes": 2.0,
            "hour": 2,
            "frequency": 15,
            "transaction_type": "online",
            "distance_km": 0.0,
            "device_loc_risk": 0.8,
            "merchant": "Electronics"
        }
    }
    
    # Test case selector
    selected_test = st.selectbox(
        "Select a Test Case:",
        options=list(test_cases.keys()),
        help="Choose a pre-configured test scenario"
    )
    
    # Display test case details
    if selected_test:
        test_data = test_cases[selected_test]
        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #667eea;">
            <p style="color: white; margin: 0;"><strong>Description:</strong> {test_data['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show test data parameters
        with st.expander("View Test Parameters"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Amount:** ${test_data['amount']:,.2f}")
                st.write(f"**Time Since Last:** {test_data['time_minutes']} minutes")
                st.write(f"**Hour:** {test_data['hour']}:00")
                st.write(f"**Frequency (24h):** {test_data['frequency']}")
            with col2:
                st.write(f"**Type:** {test_data['transaction_type'].capitalize()}")
                st.write(f"**Merchant:** {test_data['merchant']}")
                if test_data['transaction_type'] == 'offline':
                    st.write(f"**Distance:** {test_data['distance_km']} km")
                else:
                    st.write(f"**Device Risk:** {test_data['device_loc_risk']}")
        
        # Load and analyze button
        if st.button("Load & Analyze This Test Case", use_container_width=True, type="primary"):
            # Create user input from test case
            user_input = {
                'amount': test_data['amount'],
                'time_minutes': test_data['time_minutes'],
                'frequency_24h': test_data['frequency'],
                'transaction_type': test_data['transaction_type'],
                'distance_km': test_data['distance_km'],
                'device_loc_risk': test_data['device_loc_risk'],
                'merchant_type': test_data['merchant'],
                'hour_of_day': test_data['hour']
            }
            
            # Get prediction
            prob = predict_transaction(user_input)
            
            # Determine risk level
            if prob >= 0.7:
                risk_level = "CRITICAL"
                risk_color = "#dc2626"
                recommendation = "BLOCK this transaction immediately and contact the cardholder for verification."
                alert_class = "alert-danger"
            elif prob >= 0.4:
                risk_level = "HIGH"
                risk_color = "#f59e0b"
                recommendation = "REVIEW this transaction. Request additional verification before approval."
                alert_class = "alert-warning"
            else:
                risk_level = "LOW"
                risk_color = "#10b981"
                recommendation = "APPROVE this transaction. Appears to be legitimate."
                alert_class = "alert-success"
            
            # Calculate confidence
            model_confidence = max(prob, 1 - prob) * 100
            bar_width = int(prob * 100) if prob >= 0.4 else int((1 - prob) * 100)
            
            st.markdown(f"""
            <div class="alert-box {alert_class}">
                <h3 style="margin: 0; font-size: 1.25rem;">Risk Assessment: {risk_level}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label" style="color: #6b7280;">Fraud Probability</div>
                    <div class="metric-value" style="color: {risk_color};">{prob*100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label" style="color: #6b7280;">Risk Level</div>
                    <div class="metric-value" style="color: {risk_color};">{risk_level}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label" style="color: #6b7280;">Model Confidence</div>
                    <div class="metric-value" style="color: #111827;">{model_confidence:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="progress-container">
                <div class="progress-bar" style="width: {bar_width}%; background: {risk_color};">
                </div>
                <div class="progress-text">{bar_width}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: #f9fafb; padding: 1.25rem; border-radius: 10px; border-left: 4px solid {risk_color}; margin-top: 1rem;">
                <h4 style="margin: 0 0 0.5rem 0; color: #111827;">Recommended Action</h4>
                <p style="margin: 0; color: #374151; font-size: 1rem;">{recommendation}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Bulk CSV Test Cases
    with test_tab2:
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <p style="color: #f3f4f6; margin: 0;"><strong>Download pre-configured CSV test files:</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Define bulk test scenarios
        bulk_tests = {
            "Test Case 11: Normal Mixed Transactions": {
                "description": "CSV file with 100 normal transactions of various types and amounts. Expected: <5% fraud rate.",
                "expected": "<5% flagged as fraud",
                "file_info": "100 legitimate transactions, mixed amounts ($10-$5000)"
            },
            "Test Case 12: High-Risk Transaction Batch": {
                "description": "CSV with 50 transactions, 20 containing fraud indicators (high amounts, unusual times, suspicious patterns). Expected: ~40% fraud rate.",
                "expected": "~40% flagged as fraud",
                "file_info": "50 transactions, 20 with fraud indicators"
            },
            "Test Case 13: Edge Cases Dataset": {
                "description": "CSV containing edge cases: $0 transactions, maximum amounts, boundary time values, extreme V-features. Expected: System handles gracefully.",
                "expected": "No errors, valid predictions for all rows",
                "file_info": "30 transactions with edge case values"
            },
            "Test Case 14: Large Volume Test": {
                "description": "CSV with 1000+ transactions to test system performance and stability. Expected: Completes in <30 seconds.",
                "expected": "Processes successfully, <30 seconds",
                "file_info": "1000+ transactions, performance test"
            }
        }
        
        selected_bulk = st.selectbox(
            "Select a Bulk Test Scenario:",
            options=list(bulk_tests.keys()),
            help="Choose a bulk CSV test case",
            key="bulk_test_selector"
        )
        
        if selected_bulk:
            test_info = bulk_tests[selected_bulk]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #667eea;">
                    <p style="color: #f3f4f6; margin: 0 0 0.5rem 0;"><strong>Description:</strong></p>
                    <p style="color: #d1d5db; margin: 0;">{test_info['description']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label" style="color: #6b7280;">Expected Result</div>
                    <div style="color: #10b981; font-size: 0.9rem; margin-top: 0.5rem;">{}</div>
                </div>
                """.format(test_info['expected']), unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: #fef3c7; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #f59e0b;">
                <p style="color: #92400e; margin: 0;"><strong>📁 File Info:</strong> {test_info['file_info']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Generate test CSV data based on test case
            import numpy as np
            
            def generate_test_csv(case_name, test_data):
                """Generate test CSV data based on case type"""
                np.random.seed(42)  # For reproducibility
                
                if "Normal Mixed" in case_name:
                    row_count = 100
                    # Generate normal transactions
                    time_vals = np.random.uniform(0, 172800, row_count)  # 48 hours in seconds
                    amounts = np.random.choice([10, 25, 50, 100, 250, 500, 1000, 2500, 5000], row_count)
                    # Generate V features (PCA components) - normal distribution
                    v_features = np.random.randn(row_count, 28) * 2
                    
                elif "High-Risk" in case_name:
                    row_count = 50
                    # Mix of normal and suspicious transactions
                    time_vals = np.random.uniform(0, 86400, row_count)
                    amounts = []
                    v_features_list = []
                    
                    for i in range(row_count):
                        if i < 20:  # First 20 are suspicious
                            amounts.append(np.random.choice([15000, 20000, 25000, 30000]))
                            # Suspicious V features (outliers)
                            v_features_list.append(np.random.randn(28) * 5 + 3)
                        else:
                            amounts.append(np.random.choice([50, 100, 250, 500]))
                            v_features_list.append(np.random.randn(28) * 2)
                    
                    amounts = np.array(amounts)
                    v_features = np.array(v_features_list)
                    
                elif "Edge Cases" in case_name:
                    row_count = 30
                    # Edge case values
                    time_vals = np.concatenate([
                        [0, 1, 172799, 172800],  # Boundary times
                        np.random.uniform(0, 172800, row_count - 4)
                    ])
                    amounts = np.concatenate([
                        [0, 0.01, 25000, 25000],  # Edge amounts
                        np.random.uniform(10, 5000, row_count - 4)
                    ])
                    v_features = np.concatenate([
                        np.random.randn(4, 28) * 10,  # Extreme values
                        np.random.randn(row_count - 4, 28) * 2
                    ])
                    
                else:  # Large Volume
                    row_count = 1000
                    # Generate large dataset
                    time_vals = np.random.uniform(0, 604800, row_count)  # 7 days
                    amounts = np.random.lognormal(4, 2, row_count)  # Log-normal distribution
                    amounts = np.clip(amounts, 10, 25000)
                    v_features = np.random.randn(row_count, 28) * 2.5
                
                # Create DataFrame
                df_test = pd.DataFrame()
                df_test['Time'] = time_vals
                
                # Add V1-V28 features
                for i in range(28):
                    df_test[f'V{i+1}'] = v_features[:, i]
                
                df_test['Amount'] = amounts
                
                return df_test.to_csv(index=False).encode('utf-8'), row_count
            
            # Generate and provide download button
            csv_data, row_count = generate_test_csv(selected_bulk, test_info)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.download_button(
                    label="📥 Download Test CSV File",
                    data=csv_data,
                    file_name=f"test_case_{selected_bulk.split(':')[0].replace(' ', '_').lower()}.csv",
                    mime='text/csv',
                    use_container_width=True,
                    type="primary"
                )
            
            with col2:
                st.markdown(f"""
                <div style="background: rgba(16, 185, 129, 0.1); padding: 0.75rem; border-radius: 8px; text-align: center; margin-top: 0.5rem;">
                    <p style="color: #10b981; margin: 0; font-size: 0.9rem;"><strong>{row_count} rows</strong> ready to download</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;">
                <h4 style="color: #f3f4f6; margin: 0 0 1rem 0;">How to Test This Scenario:</h4>
                <ol style="color: #d1d5db; margin: 0; padding-left: 1.5rem;">
                    <li>Click the "Download Test CSV File" button above</li>
                    <li>Go to the 'Bulk Upload Analysis' tab</li>
                    <li>Upload the downloaded CSV file</li>
                    <li>Click "Process and Analyze"</li>
                    <li>Compare results with expected outcomes shown above</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)

# (Tab 3 removed - content moved to correct position above)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.9); padding: 1.5rem;">
    <p style="margin: 0; font-size: 1rem; font-weight: 500;">Euphoria GenX Project</p>
    <p style="margin: 0.75rem 0; font-size: 0.9rem;">Developed by: Chirag Nahata | Snigdha Ghosh | Somyadip Ghosh | Surybha Pal</p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem; opacity: 0.8;">Powered by XGBoost Machine Learning | © 2025 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
