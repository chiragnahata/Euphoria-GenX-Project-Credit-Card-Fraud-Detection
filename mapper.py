
import numpy as np
import pandas as pd

def demo_mapper(user_input, demo_feature_stats):
    # user_input keys: amount, time_minutes, frequency_24h, transaction_type,
    # distance_km, device_loc_risk, merchant_type, hour_of_day
    t = float(user_input.get('time_minutes', 0.0)) * 60.0
    merchant_map = {'Grocery':0.0, 'Online Shopping':0.8, 'Travel':1.2, 'Electronics':0.5, 'Others':0.0}
    merchant_val = merchant_map.get(user_input.get('merchant_type','Others'), 0.0)
    amt = float(user_input.get('amount', 0.0))
    freq = float(user_input.get('frequency_24h', 0.0))
    hour = float(user_input.get('hour_of_day', 12.0))
    dlrisk = float(user_input.get('device_loc_risk', 0.0))
    distance = float(user_input.get('distance_km', 0.0))
    base = (amt / 100.0) + freq*0.3 + merchant_val + dlrisk*2.0 + distance/50.0 + (abs(12-hour)/12.0)
    V = []
    for i in range(1,29):
        stats = demo_feature_stats.get(f'V{i}', {'mean':0.0,'std':1.0})
        mean_i = stats['mean']
        std_i = stats['std'] if stats['std']>0 else 1.0
        val = mean_i + (np.sin(i + base) * (0.5 + (i%5)/5.0)) * std_i + ( (amt%100) / 100.0 - 0.5 ) * 0.2
        V.append(val)
    data = {'Time': [t], **{f'V{i}': [V[i-1]] for i in range(1,29)}, 'Amount': [amt]}
    df = pd.DataFrame(data)
    return df
