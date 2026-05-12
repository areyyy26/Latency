import pytest
import joblib
import pandas as pd
import numpy as np

COLUMNS = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

@pytest.fixture
def model_and_scaler():
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

def test_model_prediction_output(model_and_scaler):
    model, scaler = model_and_scaler
    
    dummy_input = pd.DataFrame([[6, 148, 72, 35, 0, 33.6, 0.627, 50]], 
                               columns=COLUMNS)
    
    dummy_input_scaled = scaler.transform(dummy_input)
    
    prediction = model.predict(dummy_input_scaled)
    
    assert prediction[0] in [0, 1]

def test_model_structure(model_and_scaler):
    model, _ = model_and_scaler
    assert hasattr(model, "predict")
