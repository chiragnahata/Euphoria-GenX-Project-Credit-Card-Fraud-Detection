
# Credit Card Fraud Detection — Demo

## Authors
- Chirag Nahata
- Snigdha Ghosh
- Somyadip Ghosh
- Surybha Pal

## Overview
Train XGBoost on Kaggle `creditcard.csv`. Provide Flask demo that supports single-user (online/offline) deterministic mapping and bulk CSV upload.

## Files
- model_bundle.joblib - saved model + scaler + demo stats
- mapper.py - deterministic demo mapper
- app.py - Flask app
- templates/index.html - UI template
- requirements.txt

## Run locally
1. Ensure you have the necessary files in the same directory.
2. python3 -m venv venv
3. source venv/bin/activate
4. pip install -r requirements.txt
5. python app.py
6. Open http://localhost:5000

## Deploy
Upload files to PythonAnywhere or similar. Ensure model_bundle.joblib, mapper.py, app.py, requirements.txt and the 'templates' folder (containing index.html) are in your project root directory.

## Notes
Demo mapping is synthetic and for presentation only.
