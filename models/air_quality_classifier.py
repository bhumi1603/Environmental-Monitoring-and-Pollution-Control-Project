import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib
import os

class AirQualityClassifier:
    def __init__(self):
        self.model = None
        self.model_path = "models/air_quality_classifier.pkl"
        
    def aqi_to_category(self, aqi):
        """Convert AQI value to category"""
        if aqi <= 50:
            return 'Good'
        elif aqi <= 100:
            return 'Moderate'
        elif aqi <= 150:
            return 'Unhealthy for Sensitive Groups'
        elif aqi <= 200:
            return 'Unhealthy'
        elif aqi <= 300:
            return 'Very Unhealthy'
        else:
            return 'Hazardous'
    
    def prepare_data(self, df):
        """Prepare data for classification"""
        df = df.copy()
        
        # Create target variable (AQI category)
        df['aqi_category'] = df['aqi'].apply(self.aqi_to_category)
        
        # Features for classification
        features = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 'temperature', 'humidity']
        
        # Only use features that exist in the dataframe
        features = [f for f in features if f in df.columns]
        
        return df, features
    
    def train_classifier(self, df):
        """Train air quality classification model"""
        df_processed, features = self.prepare_data(df)
        
        # Split data
        X = df_processed[features]
        y = df_processed['aqi_category']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Classifier trained with accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, self.model_path)
        
        return accuracy
    
    def load_classifier(self):
        """Load trained classifier"""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return True
        return False
    
    def predict_category(self, input_data):
        """Predict air quality category"""
        if self.model is None:
            if not self.load_classifier():
                raise Exception("Classifier not trained yet")
        
        # Convert input to DataFrame if it's a dictionary
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Features used in training
        features = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 'temperature', 'humidity']
        features = [f for f in features if f in df.columns]
        
        # Make prediction
        prediction = self.model.predict(df[features])
        return prediction[0] if len(prediction) == 1 else prediction
    
    def predict_proba(self, input_data):
        """Predict probability for each air quality category"""
        if self.model is None:
            if not self.load_classifier():
                raise Exception("Classifier not trained yet")
        
        # Convert input to DataFrame if it's a dictionary
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Features used in training
        features = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 'temperature', 'humidity']
        features = [f for f in features if f in df.columns]
        
        # Make prediction
        probabilities = self.model.predict_proba(df[features])
        categories = self.model.classes_
        
        return dict(zip(categories, probabilities[0]))