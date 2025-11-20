# app/ml_model.py

import joblib
import os
import pandas as pd
from django.conf import settings

class FraudModel:
    def __init__(self):
        # --- THIS IS THE MODIFIED PATH ---
        # It now looks for the file directly in the 'app' folder
        model_path = os.path.join(settings.BASE_DIR, 'app', 'fraud_model_pipeline.pkl')
        
        try:
            self.pipeline = joblib.load(model_path)
            print("Fraud model pipeline loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}")
            self.pipeline = None

    def predict(self, data):
        if self.pipeline is None:
            return False, 0.0

        input_df = pd.DataFrame([data])
        probabilities = self.pipeline.predict_proba(input_df)
        fraud_probability = probabilities[0][1]
        
        # Use the 0.75 threshold you found in your notebook
        is_fraud = bool(fraud_probability >= 0.75)
        
        return is_fraud, fraud_probability

# Create a single instance to be loaded when the server starts
model = FraudModel()