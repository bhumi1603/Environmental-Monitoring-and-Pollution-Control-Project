import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import joblib
import os

class AnomalyDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_path = "models/anomaly_detector.pkl"
        self.scaler_path = "models/anomaly_scaler.pkl"
        
    def prepare_data(self, df):
        """Prepare data for anomaly detection"""
        df = df.copy()
        
        # Features for anomaly detection
        features = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
        
        # Only use features that exist in the dataframe
        features = [f for f in features if f in df.columns]
        
        return df, features
    
    def train_anomaly_detector(self, df, contamination=0.05):
        """Train anomaly detection model"""
        df_processed, features = self.prepare_data(df)
        
        # Scale the data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(df_processed[features])
        
        # Train Isolation Forest for anomaly detection
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        self.model.fit(X_scaled)
        
        # Detect anomalies in training data
        anomalies = self.model.predict(X_scaled)
        anomaly_percentage = (anomalies == -1).sum() / len(anomalies) * 100
        
        print(f"Anomaly detector trained. Found {anomaly_percentage:.2f}% anomalies in training data.")
        
        # Save model
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        return anomaly_percentage
    
    def load_anomaly_detector(self):
        """Load trained anomaly detector"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            return True
        return False
    
    def detect_anomalies(self, input_data):
        """Detect anomalies in input data"""
        if self.model is None:
            if not self.load_anomaly_detector():
                raise Exception("Anomaly detector not trained yet")
        
        # Convert input to DataFrame if it's a dictionary
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Features used in training
        features = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
        features = [f for f in features if f in df.columns]
        
        # Scale the data
        X_scaled = self.scaler.transform(df[features])
        
        # Detect anomalies
        anomalies = self.model.predict(X_scaled)
        anomaly_scores = self.model.decision_function(X_scaled)
        
        # Convert to more intuitive format (1 = normal, -1 = anomaly)
        results = []
        for i, (anomaly, score) in enumerate(zip(anomalies, anomaly_scores)):
            results.append({
                'is_anomaly': anomaly == -1,
                'anomaly_score': score,
                'confidence': 1 - (1 / (1 + np.exp(-np.abs(score))))  # Convert to probability-like value
            })
        
        return results[0] if len(results) == 1 else results
    
    def find_historical_anomalies(self, historical_data):
        """Find anomalies in historical data"""
        if self.model is None:
            if not self.load_anomaly_detector():
                raise Exception("Anomaly detector not trained yet")
        
        # Prepare data
        df_processed, features = self.prepare_data(historical_data)
        
        # Scale the data
        X_scaled = self.scaler.transform(df_processed[features])
        
        # Detect anomalies
        anomalies = self.model.predict(X_scaled)
        anomaly_scores = self.model.decision_function(X_scaled)
        
        # Add results to dataframe
        result_df = historical_data.copy()
        result_df['is_anomaly'] = anomalies == -1
        result_df['anomaly_score'] = anomaly_scores
        result_df['anomaly_confidence'] = 1 - (1 / (1 + np.exp(-np.abs(anomaly_scores))))
        
        return result_df