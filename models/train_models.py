import pandas as pd
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collector import DataCollector
from pollution_predictor import PollutionPredictor
from air_quality_classifier import AirQualityClassifier
from anomaly_detector import AnomalyDetector
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train environmental monitoring models')
    parser.add_argument('--days', type=int, default=365, help='Number of days of historical data to use for training')
    parser.add_argument('--cities', nargs='+', default=['Delhi', 'Mumbai', 'Kolkata'], help='Cities to include in training data')
    parser.add_argument('--target', type=str, default='pm25', choices=['pm25', 'pm10', 'aqi'], help='Target variable for prediction')
    parser.add_argument('--forecast-days', type=int, default=1, help='Number of days ahead to forecast')
    parser.add_argument('--force-retrain', action='store_true', help='Force retraining even if models exist')
    
    args = parser.parse_args()
    
    # Initialize data collector
    collector = DataCollector()
    
    # Load or generate historical data
    print("Loading historical data...")
    historical_data = collector.load_historical_data()
    
    if historical_data is None or historical_data.empty:
        print("No historical data found. Generating sample data...")
        historical_data = collector.generate_sample_historical_data()
    
    print(f"Loaded historical data with {len(historical_data)} records")
    
    # Filter data for selected cities if specified
    if args.cities:
        historical_data = historical_data[historical_data['city'].isin(args.cities)]
        print(f"Filtered data for cities: {args.cities}. Now {len(historical_data)} records")
    
    # Train pollution prediction model
    print("\n" + "="*50)
    print("TRAINING POLLUTION PREDICTION MODEL")
    print("="*50)
    
    predictor = PollutionPredictor()
    
    # Check if model already exists
    model_exists = os.path.exists(predictor.model_path) and os.path.exists(predictor.preprocessor_path)
    
    if args.force_retrain or not model_exists:
        print(f"Training pollution prediction model for target: {args.target}")
        print(f"Forecast horizon: {args.forecast_days} day(s)")
        
        mae, rmse, r2 = predictor.train_model(
            historical_data, 
            target=args.target, 
            forecast_days=args.forecast_days
        )
        
        print(f"Training completed: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.2f}")
    else:
        print("Pollution prediction model already exists. Use --force-retrain to retrain.")
    
    # Test the prediction model
    try:
        predictor.load_model()
        # Get the latest data point for testing
        test_data = historical_data.iloc[-1:].copy()
        prediction = predictor.predict(test_data)
        print(f"Test prediction for {args.target}: {prediction:.2f}")
    except Exception as e:
        print(f"Error testing prediction model: {e}")
    
    # Train air quality classifier
    print("\n" + "="*50)
    print("TRAINING AIR QUALITY CLASSIFIER")
    print("="*50)
    
    classifier = AirQualityClassifier()
    classifier_exists = os.path.exists(classifier.model_path)
    
    if args.force_retrain or not classifier_exists:
        print("Training air quality classification model...")
        accuracy = classifier.train_classifier(historical_data)
        print(f"Classification accuracy: {accuracy:.2f}")
    else:
        print("Air quality classifier already exists. Use --force-retrain to retrain.")
    
    # Test the classifier
    try:
        classifier.load_classifier()
        test_data = historical_data.iloc[-1:].copy()
        category = classifier.predict_category(test_data)
        probabilities = classifier.predict_proba(test_data)
        print(f"Test classification: {category}")
        print("Category probabilities:")
        for cat, prob in probabilities.items():
            print(f"  {cat}: {prob:.3f}")
    except Exception as e:
        print(f"Error testing classifier: {e}")
    
    # Train anomaly detector
    print("\n" + "="*50)
    print("TRAINING ANOMALY DETECTOR")
    print("="*50)
    
    detector = AnomalyDetector()
    detector_exists = os.path.exists(detector.model_path) and os.path.exists(detector.scaler_path)
    
    if args.force_retrain or not detector_exists:
        print("Training anomaly detection model...")
        anomaly_percentage = detector.train_anomaly_detector(historical_data)
        print(f"Anomalies detected in training data: {anomaly_percentage:.2f}%")
    else:
        print("Anomaly detector already exists. Use --force-retrain to retrain.")
    
    # Test the anomaly detector
    try:
        detector.load_anomaly_detector()
        test_data = historical_data.iloc[-1:].copy()
        anomaly_result = detector.detect_anomalies(test_data)
        print(f"Test anomaly detection: {'Anomaly' if anomaly_result['is_anomaly'] else 'Normal'}")
        print(f"Anomaly score: {anomaly_result['anomaly_score']:.3f}")
        print(f"Confidence: {anomaly_result['confidence']:.3f}")
    except Exception as e:
        print(f"Error testing anomaly detector: {e}")
    
    print("\n" + "="*50)
    print("MODEL TRAINING COMPLETED")
    print("="*50)
    print("Models saved in the 'models' directory:")
    print(f"  - Pollution predictor: {predictor.model_path}")
    print(f"  - Air quality classifier: {classifier.model_path}")
    print(f"  - Anomaly detector: {detector.model_path}")
    
    # Generate sample predictions for the next 7 days
    print("\nGenerating sample forecasts for the next 7 days...")
    try:
        future_predictions = predictor.predict_future(historical_data, days=7)
        print("7-day forecast:")
        for pred in future_predictions:
            print(f"  {pred['date']}: {pred['prediction']:.2f} {args.target}")
    except Exception as e:
        print(f"Error generating forecasts: {e}")

if __name__ == "__main__":
    main()