import streamlit as st
import joblib
import pandas as pd
import numpy as np
from mapper import demo_mapper
import os
import io

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to replicate the glassmorphism design
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    /* Main background with gradient animation */
    .stApp {
        background: linear-gradient(-45deg, #0ea5e9, #06b6d4, #14b8a6, #10b981);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        font-family: 'Poppins', sans-serif;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container styling */
    .block-container {
        padding: 1.5rem 1rem !important;
        max-width: 1100px !important;
    }
    
    /* Header styling */
    h1 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: white !important;
        text-align: center !important;
        text-shadow: 0 0 30px rgba(255, 255, 255, 0.5), 0 3px 5px rgba(0, 0, 0, 0.3) !important;
        font-size: 2rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Card styling with glassmorphism */
    .css-1r6slb0, .css-12oz5g7, div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-radius: 20px !important;
        border: 2px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3) !important;
        padding: 1.5rem !important;
    }
    
    /* Input fields styling */
    .stTextInput input, .stNumberInput input, .stSelectbox select, .stTextArea textarea {
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        background: rgba(255, 255, 255, 0.95) !important;
        color: #1f2937 !important;
        font-weight: 500 !important;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stTextInput input:focus, .stNumberInput input:focus, .stSelectbox select:focus, .stTextArea textarea:focus {
        border-color: #14b8a6 !important;
        box-shadow: 0 0 0 3px rgba(20, 184, 166, 0.2) !important;
    }
    
    /* Labels styling */
    label {
        color: white !important;
        font-weight: 600 !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #0ea5e9 0%, #14b8a6 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.85rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        border-radius: 14px !important;
        box-shadow: 0 8px 24px rgba(14, 165, 233, 0.5) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        transition: all 0.4s ease !important;
        width: 100% !important;
    }
    
    .stButton button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 12px 32px rgba(20, 184, 166, 0.6) !important;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 15px !important;
        padding: 1.25rem !important;
        border: 2px solid rgba(255, 255, 255, 0.25) !important;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.2) !important;
    }
    
    .stRadio label {
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(15px) !important;
        border-radius: 18px !important;
        padding: 1.75rem !important;
        border: 2px dashed rgba(255, 255, 255, 0.4) !important;
    }
    
    /* Success/Error message styling */
    .stSuccess, .stError, .stWarning, .stInfo {
        background: rgba(255, 255, 255, 0.25) !important;
        backdrop-filter: blur(15px) !important;
        border-radius: 15px !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        padding: 1.15rem !important;
        color: white !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Metric styling */
    .stMetric {
        background: rgba(255, 255, 255, 0.25) !important;
        backdrop-filter: blur(15px) !important;
        padding: 1.15rem !important;
        border-radius: 15px !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2) !important;
    }
    
    .stMetric label {
        color: rgba(255, 255, 255, 0.8) !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .stMetric > div {
        color: white !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.5rem !important;
        font-weight: 800 !important;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Divider */
    hr {
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent) !important;
        border: none !important;
        height: 1px !important;
        margin: 2rem 0 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center !important;
        margin-top: 2rem !important;
        color: white !important;
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 50px !important;
        border: 2px solid rgba(255, 255, 255, 0.2) !important;
        display: inline-block !important;
        font-weight: 600 !important;
    }
    
    /* Section titles */
    .section-title {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: white !important;
        margin-bottom: 1rem !important;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 12px !important;
        color: white !important;
        font-weight: 600 !important;
        border: 2px solid rgba(255, 255, 255, 0.25) !important;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model_bundle_path = os.path.join(os.path.dirname(__file__), 'model_bundle.joblib')
    bundle = joblib.load(model_bundle_path)
    return bundle['model'], bundle['scaler'], bundle.get('demo_feature_stats', {})

model, scaler, demo_feature_stats = load_model()

# Prediction function
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
    df_out['pred_class'] = (probs >= 0.6).astype(int)
    return df_out, None

# Header
st.markdown("<h1>🛡️ Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white; font-size: 0.95rem; opacity: 0.95; margin-bottom: 2rem;'>🚀 Advanced ML-powered fraud detection system for real-time transaction analysis</p>", unsafe_allow_html=True)

# Create tabs for different sections
tab1, tab2 = st.tabs(["🔍 Single Transaction Analysis", "📊 Bulk Upload Analysis"])

# Tab 1: Single Transaction Analysis
with tab1:
    st.markdown("<div class='section-title'>⚙️ Transaction Configuration</div>", unsafe_allow_html=True)
    
    # Transaction type selector
    transaction_type = st.radio(
        "Transaction Type",
        options=["offline", "online"],
        format_func=lambda x: "🏪 Offline (In-Store)" if x == "offline" else "🌐 Online (E-Commerce)",
        horizontal=True
    )
    
    # Form for transaction details
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            amount = st.number_input("💰 Transaction Amount (₹)", min_value=0.0, step=0.01, value=0.0)
            time_minutes = st.number_input("⏱️ Time Since Last Transaction (min)", min_value=0.0, value=0.0)
        
        with col2:
            hour = st.number_input("🕐 Transaction Hour (0-23)", min_value=0, max_value=23, value=12)
            frequency = st.number_input("🔄 Transactions in Last 24h", min_value=0, value=0)
        
        with col3:
            if transaction_type == "offline":
                distance_km = st.number_input("📍 Distance from Last Transaction (km)", min_value=0.0, step=0.1, value=0.0)
                device_loc_risk = 0.0
            else:
                device_loc_risk = st.number_input("📱 Device/Location Risk Score (0-1)", min_value=0.0, max_value=1.0, step=0.01, value=0.0)
                distance_km = 0.0
            
            merchant = st.selectbox("🛍️ Merchant Category", 
                                   options=["Grocery", "Online Shopping", "Travel", "Electronics", "Others"])
        
        message = st.text_area("💬 Additional Notes / Message (Optional)", 
                              placeholder="Enter any additional information or notes about this transaction...")
        
        submitted = st.form_submit_button("🧠 Analyze Transaction Risk", use_container_width=True)
        
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
            
            # Manual suspicious input check
            suspicious = False
            suspicious_reasons = []
            
            if amount > 20000:
                suspicious = True
                suspicious_reasons.append("High amount")
            if frequency > 10:
                suspicious = True
                suspicious_reasons.append("Many transactions in 24h")
            if distance_km > 100:
                suspicious = True
                suspicious_reasons.append("Large distance")
            if merchant in ["Electronics", "Travel", "Online Shopping"]:
                suspicious = True
                suspicious_reasons.append("High-risk merchant")
            if hour < 6 or hour > 22:
                suspicious = True
                suspicious_reasons.append("Unusual hour")
            
            # Get prediction
            df_demo = demo_mapper(user_input, demo_feature_stats)
            df_demo[['Time','Amount']] = scaler.transform(df_demo[['Time','Amount']])
            prob = float(model.predict_proba(df_demo[['Time'] + [f'V{i}' for i in range(1,29)] + ['Amount']])[0,1])
            
            # Determine label and advice
            label = "High Fraud Risk" if prob >= 0.4 or suspicious else "Low Fraud Risk"
            
            if prob >= 0.85:
                advice = "🚨 Block and contact bank immediately."
            elif prob >= 0.4 or suspicious:
                advice = "⚠️ Review and confirm this transaction."
                if suspicious:
                    advice += f" (Suspicious input detected: {', '.join(suspicious_reasons)})"
            else:
                advice = "✅ Looks normal. Transaction approved."
            
            confidence = int(prob * 100)
            
            # Display results
            st.markdown("---")
            
            if "High" in label:
                st.error(f"⚠️ {label}")
                result_color = "#ef4444"
            else:
                st.success(f"✅ {label}")
                result_color = "#22c55e"
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Fraud Probability", f"{prob*100:.2f}%")
            with col2:
                st.metric("Risk Level", "HIGH" if "High" in label else "LOW")
            with col3:
                st.metric("Confidence Level", f"{confidence}%")
            
            # Confidence bar
            st.markdown(f"""
            <div style="background: rgba(255, 255, 255, 0.2); border-radius: 10px; height: 30px; overflow: hidden; margin: 1.5rem 0;">
                <div style="background: linear-gradient(90deg, {result_color}, {result_color}); height: 100%; width: {confidence}%; 
                    border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; 
                    font-weight: bold; font-size: 0.85rem; transition: width 1s ease-out;">
                    {confidence}% Sure
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display message if provided
            if message:
                st.info(f"📝 **Your Message:** {message}")
            
            # Recommendation
            st.markdown(f"""
            <div style="background: rgba(255, 255, 255, 0.25); backdrop-filter: blur(15px); padding: 1.15rem; 
                border-radius: 15px; border: 2px solid rgba(255, 255, 255, 0.3); border-left: 4px solid #6366f1; 
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2); color: white;">
                <strong style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                    💡 Recommendation
                </strong>
                <div>{advice}</div>
            </div>
            """, unsafe_allow_html=True)

# Tab 2: Bulk Upload Analysis
with tab2:
    st.markdown("<div class='section-title'>☁️ Bulk Transaction Analysis</div>", unsafe_allow_html=True)
    st.markdown("<p style='color: white; margin-bottom: 1.5rem;'>Upload a CSV file containing multiple transactions for batch processing and analysis</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose CSV File", type=['csv'], help="Upload a CSV file with transaction data")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! {len(df)} transactions found.")
            
            # Show preview
            with st.expander("📋 Preview Data (first 5 rows)"):
                st.dataframe(df.head(), use_container_width=True)
            
            if st.button("🔍 Process and Analyze", use_container_width=True):
                with st.spinner("Processing transactions..."):
                    df_out, err = predict_df(df)
                    
                    if err:
                        st.error(f"❌ Error: {err}")
                    else:
                        st.success("✅ Analysis complete!")
                        
                        # Show statistics
                        fraud_count = (df_out['pred_class'] == 1).sum()
                        normal_count = (df_out['pred_class'] == 0).sum()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Transactions", len(df_out))
                        with col2:
                            st.metric("Fraudulent", fraud_count, delta=f"{fraud_count/len(df_out)*100:.1f}%")
                        with col3:
                            st.metric("Normal", normal_count, delta=f"{normal_count/len(df_out)*100:.1f}%")
                        
                        # Show results preview
                        with st.expander("📊 View Results (first 10 rows)"):
                            st.dataframe(df_out.head(10), use_container_width=True)
                        
                        # Download results
                        csv = df_out.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv,
                            file_name='fraud_predictions.csv',
                            mime='text/csv',
                            use_container_width=True
                        )
        except Exception as e:
            st.error(f"❌ Error reading CSV: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <div class="footer">
        🛡️ Powered by ML | 🧠 Euphoria GenX | 🔒 Secure
    </div>
</div>
""", unsafe_allow_html=True)
