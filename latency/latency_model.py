import joblib
import time
import pandas as pd
import numpy as np

model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

sample_data = pd.DataFrame([[2, 110, 64, 20, 0, 25.5, 0.126, 28]], 
                           columns=feature_names)

sample_scaled = scaler.transform(sample_data)

start_time = time.time()
model.predict(sample_scaled)
end_time = time.time()

latency = (end_time - start_time) * 1000
print(f"Latency Prediksi: {latency:.4f} ms")

latencies = []
for _ in range(100):
    start = time.time()
    model.predict(sample_scaled)
    latencies.append(time.time() - start)

avg_latency = (sum(latencies) / len(latencies)) * 1000
print(f"Rata-rata Latency (100 iterasi): {avg_latency:.4f} ms")
