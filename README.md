# 🛡️ Credit Card Fraud Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://creditcardfrauddetection-euphoriagenxproject.streamlit.app/)

## 🚀 Live Demo
**Access the application here:** [https://creditcardfrauddetection-euphoriagenxproject.streamlit.app/](https://creditcardfrauddetection-euphoriagenxproject.streamlit.app/)

## 👥 Authors
- **Chirag Nahata**
- **Snigdha Ghosh**
- **Somyadip Ghosh**
- **Surybha Pal**

## 📊 Overview
Advanced ML-powered fraud detection system using XGBoost trained on the Kaggle Credit Card Fraud Detection dataset. The system provides:
- 🔍 **Single Transaction Analysis** - Real-time fraud detection for individual transactions
- 📊 **Bulk Upload Analysis** - Process multiple transactions from CSV files
- 🎨 **Modern UI** - Glassmorphism design with animated gradients
- 🌐 **Online/Offline Detection** - Support for both e-commerce and in-store transactions

## 📁 Project Files
- `streamlit_app.py` - Streamlit web application
- `model_bundle.joblib` - Trained XGBoost model + scaler + feature statistics
- `mapper.py` - Deterministic demo feature mapper
- `creditcard.csv` - Training dataset (tracked with Git LFS)
- `Credit_Card_Fraud_Detection.ipynb` - Model training notebook
- `requirements.txt` - Python dependencies

## 🏃 Run Locally

### Prerequisites
- Python 3.8 or higher
- Git LFS (for dataset)

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/chiragnahata/Euphoria-GenX-Project-Credit-Card-Fraud-Detection.git
cd Euphoria-GenX-Project-Credit-Card-Fraud-Detection

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

## 🌐 Deploy on Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository, branch (`main`), and main file (`streamlit_app.py`)
6. Click "Deploy"

## 🎯 Features

### Single Transaction Analysis
- Input transaction details (amount, time, merchant, etc.)
- Choose between online and offline transaction types
- Get real-time fraud probability and risk assessment
- Receive actionable recommendations

### Bulk Upload Analysis
- Upload CSV files with multiple transactions
- Process hundreds of transactions at once
- Download results with fraud predictions
- View statistics and summaries

## 🔧 Technology Stack
- **Machine Learning**: XGBoost, scikit-learn, imbalanced-learn
- **Web Framework**: Streamlit
- **Data Processing**: pandas, numpy
- **Model Explainability**: SHAP
- **Version Control**: Git, Git LFS

## 📝 Notes
- Demo mapping is synthetic and for presentation purposes
- The model achieves high accuracy on the imbalanced dataset
- Transaction features are PCA-transformed for privacy

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

This project is part of the Euphoria GenX initiative.
