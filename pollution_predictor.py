import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

class PollutionPredictor:
    def __init__(self):
        self.model = None
        self.model_path = "models/pollution_model.pkl"
        
    def prepare_data(self, df):
        """Prepare data for training"""
        # Create features from date
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Create lag features
        for lag in range(1, 8):
            df[f'pm25_lag_{lag}'] = df['pm25'].shift(lag)
            df[f'pm10_lag_{lag}'] = df['pm10'].shift(lag)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        return df
    
    def train_model(self, df, target='pm25'):
        """Train the prediction model"""
        # Prepare data
        df_processed = self.prepare_data(df)
        
        # Define features and target
        features = ['year', 'month', 'day', 'day_of_week', 'temperature', 'humidity']
        
        # Add lag features
        for lag in range(1, 8):
            features.append(f'pm25_lag_{lag}')
            features.append(f'pm10_lag_{lag}')
        
        X = df_processed[features]
        y = df_processed[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model trained with MAE: {mae:.2f}, R2: {r2:.2f}")
        
        # Save model
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, self.model_path)
        
        return mae, r2
    
    def load_model(self):
        """Load trained model"""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return True
        return False
    
    def predict(self, input_data):
        """Make prediction using trained model"""
        if self.model is None:
            if not self.load_model():
                raise Exception("Model not trained yet")
        
        # Convert input to DataFrame and prepare features
        df = pd.DataFrame([input_data])
        df_processed = self.prepare_data(df)
        
        # Ensure we have the same features as training
        features = ['year', 'month', 'day', 'day_of_week', 'temperature', 'humidity']
        for lag in range(1, 8):
            features.append(f'pm25_lag_{lag}')
            features.append(f'pm10_lag_{lag}')
        
        # Make sure we have all required features
        for feature in features:
            if feature not in df_processed.columns:
                df_processed[feature] = 0  # Default value for missing features
        
        prediction = self.model.predict(df_processed[features])
        return prediction[0]