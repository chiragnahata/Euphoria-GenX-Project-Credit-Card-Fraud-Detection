
from flask import Flask, render_template, request, send_file
import joblib, io, pandas as pd
from mapper import demo_mapper
import os # Import os

app = Flask(__name__)
# Define the path to the model bundle relative to where the app is run
# Assuming model_bundle.joblib is in the same directory as the app.py
model_bundle_path = os.path.join(os.path.dirname(__file__), 'model_bundle.joblib')
bundle = joblib.load(model_bundle_path) # Load from the relative path
model = bundle['model']
scaler = bundle['scaler']
demo_feature_stats = bundle.get('demo_feature_stats', {})

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    f = request.files.get('file')
    if not f:
        return render_template('index.html', csv_error="No file uploaded.")
    try:
        df = pd.read_csv(f)
    except Exception as e:
        return render_template('index.html', csv_error=f"Error reading CSV: {str(e)}")
    df_out, err = predict_df(df)
    if err:
        return render_template('index.html', csv_error=err)
    # Download the CSV
    output = io.StringIO()
    df_out.to_csv(output, index=False)
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()), as_attachment=True, download_name='predictions.csv', mimetype='text/csv')

@app.route('/predict_single', methods=['POST'])
def predict_single():
    trans_type = request.form.get('transaction_type','offline')
    amount = float(request.form.get('amount',0.0))
    time_min = float(request.form.get('time',0.0))
    frequency = int(request.form.get('frequency',0))
    merchant = request.form.get('merchant','Others')
    hour_of_day = int(request.form.get('hour',12))
    if trans_type == 'online':
        device_loc_risk = float(request.form.get('device_loc_risk',0.0))
        distance_km = 0.0
    else:
        device_loc_risk = 0.0
        distance_km = float(request.form.get('distance_km',0.0))
    user_input = {'amount': amount, 'time_minutes': time_min, 'frequency_24h': frequency,
                  'transaction_type': trans_type, 'distance_km': distance_km,
                  'device_loc_risk': device_loc_risk, 'merchant_type': merchant,
                  'hour_of_day': hour_of_day}
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
    if hour_of_day < 6 or hour_of_day > 22:
        suspicious = True
        suspicious_reasons.append("Unusual hour")
    df_demo = demo_mapper(user_input, demo_feature_stats)
    df_demo[['Time','Amount']] = scaler.transform(df_demo[['Time','Amount']])
    prob = float(model.predict_proba(df_demo[['Time'] + [f'V{i}' for i in range(1,29)] + ['Amount']])[0,1])
    label = "High Fraud Risk" if prob >= 0.4 or suspicious else "Low Fraud Risk"
    advice = ""
    if prob >= 0.85:
        advice = "Block and contact bank."
    elif prob >= 0.4 or suspicious:
        advice = "Review and confirm."
        if suspicious:
            advice += " (Suspicious input: " + ", ".join(suspicious_reasons) + ")"
    else:
        advice = "Looks normal."
    return render_template('index.html', result=label, probability=f"{prob*100:.2f}%", advice=advice, form_data=request.form)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
