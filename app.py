import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import time
import sys
import os

# Add the current directory to Python path to allow importing from models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from data_collector import DataCollector
from pollution_predictor import PollutionPredictor
from data_analyzer import DataAnalyzer

# Import from models directory
try:
    from models.air_quality_classifier import AirQualityClassifier
    from models.anomaly_detector import AnomalyDetector
except ImportError:
    # Fallback: try to import from current directory
    try:
        from air_quality_classifier import AirQualityClassifier
        from anomaly_detector import AnomalyDetector
    except ImportError:
        st.error("Required modules not found. Please make sure air_quality_classifier.py and anomaly_detector.py are in the models folder or main directory.")
        st.stop()

# ... rest of your app.py code ...
# Set page configuration
st.set_page_config(
    page_title="Environmental Monitoring & Pollution Control",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize classes
data_collector = DataCollector()
predictor = PollutionPredictor()
analyzer = DataAnalyzer()
classifier = AirQualityClassifier()
anomaly_detector = AnomalyDetector()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.3rem;
        margin-top: 2rem;
    }
    .recommendation-box {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stButton button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    .anomaly-alert {
        background-color: #ffcccc;
        border-left: 5px solid #ff0000;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">Environmental Monitoring & Pollution Control</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for user inputs
with st.sidebar:
    st.header("Configuration")
    
    # City selection
    cities = ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bangalore", "Hyderabad", "New York", "London", "Beijing", "Tokyo"]
    selected_city = st.selectbox("Select City", cities, index=0)
    
    # Data refresh options
    refresh_interval = st.slider("Data Refresh Interval (seconds)", min_value=10, max_value=300, value=60)
    
    # Model training options
    st.subheader("ML Model Options")
    train_model = st.checkbox("Train Prediction Model", value=False)
    
    if train_model:
        historical_data = data_collector.load_historical_data()
        if not historical_data.empty:
            if st.button("Train Model Now"):
                with st.spinner("Training model..."):
                    try:
                        mae, rmse, r2 = predictor.train_model(historical_data)
                        st.success(f"Model trained! MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
                    except Exception as e:
                        st.error(f"Training failed: {e}")
        else:
            st.warning("No historical data available for training")
    
    # About section
    st.markdown("---")
    st.subheader("About")
    st.info("""
    This application monitors environmental parameters and predicts pollution levels using machine learning.
    Data is collected from various sources including AQICN API.
    """)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<h2 class="sub-header">Real-time Air Quality</h2>', unsafe_allow_html=True)
    
    # Placeholder for current data
    current_data_placeholder = st.empty()
    
    # Pollutant levels
    st.markdown('<h2 class="sub-header">Pollutant Levels</h2>', unsafe_allow_html=True)
    pollutant_placeholder = st.empty()
    
    # Historical trends
    st.markdown('<h2 class="sub-header">Historical Trends</h2>', unsafe_allow_html=True)
    trend_placeholder = st.empty()

with col2:
    st.markdown('<h2 class="sub-header">AQI Category</h2>', unsafe_allow_html=True)
    category_placeholder = st.empty()
    
    st.markdown('<h2 class="sub-header">Recommendations</h2>', unsafe_allow_html=True)
    recommendation_placeholder = st.empty()
    
    # Prediction section
    st.markdown('<h2 class="sub-header">Pollution Prediction</h2>', unsafe_allow_html=True)
    
    # Inputs for prediction
    temp = st.slider("Temperature (°C)", -10.0, 50.0, 25.0)
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0)
    pm25_lag1 = st.slider("PM2.5 (previous day)", 0.0, 300.0, 50.0)
    pm10_lag1 = st.slider("PM10 (previous day)", 0.0, 500.0, 70.0)
    
    if st.button("Predict PM2.5 for Tomorrow"):
        # Create input data for prediction
        tomorrow = datetime.now() + timedelta(days=1)
        input_data = {
            'date': tomorrow.strftime("%Y-%m-%d"),
            'city': selected_city,  # Added city field
            'temperature': temp,
            'humidity': humidity,
            'pm25': pm25_lag1,  # Use correct field name
            'pm10': pm10_lag1,  # Use correct field name
            'pm25_lag_1': pm25_lag1,
            'pm10_lag_1': pm10_lag1
        }
        
        # Add more lag features with default values
        for lag in range(2, 8):
            input_data[f'pm25_lag_{lag}'] = pm25_lag1  # Use the provided value
            input_data[f'pm10_lag_{lag}'] = pm10_lag1  # Use the provided value
        
        # Add other required fields with default values
        input_data['o3'] = 30.0
        input_data['no2'] = 20.0
        input_data['so2'] = 10.0
        input_data['co'] = 1.0
        input_data['pressure'] = 1013.0
        
        try:
            # Check if model is trained first
            if not predictor.load_model():
                st.error("Model not trained yet. Please train the model first using the sidebar options.")
            else:
                prediction = predictor.predict(input_data)
                st.metric("Predicted PM2.5", f"{prediction:.2f} μg/m³")
                
                # Categorize prediction
                category = analyzer.get_aqi_category(prediction * 2)  # Rough conversion from PM2.5 to AQI
                st.info(f"Expected Air Quality: {category}")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.info("Please train the model first using the sidebar options.")
    
    # Anomaly detection section
    st.markdown('<h2 class="sub-header">Anomaly Detection</h2>', unsafe_allow_html=True)
    anomaly_placeholder = st.empty()

# Function to update the dashboard
def update_dashboard():
    # Fetch current data
    current_data = data_collector.fetch_air_quality_data(selected_city)
    
    # Load historical data
    historical_data = data_collector.load_historical_data()
    
    # Generate visualizations
    fig_gauge, fig_pollutants, fig_trend = analyzer.create_pollution_dashboard(current_data, historical_data)
    
    # Update current data display
    with current_data_placeholder.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("AQI", current_data['aqi'])
        with col2:
            st.metric("Temperature", f"{current_data['pollutants']['temperature']:.1f} °C")
        with col3:
            st.metric("Humidity", f"{current_data['pollutants']['humidity']:.1f} %")
        
        # Display gauge chart
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Update pollutant levels
    pollutant_placeholder.plotly_chart(fig_pollutants, use_container_width=True)
    
    # Update historical trends
    trend_placeholder.plotly_chart(fig_trend, use_container_width=True)
    
    # Update AQI category
    category = analyzer.get_aqi_category(current_data['aqi'])
    with category_placeholder.container():
        st.markdown(f'<div class="metric-card" style="background-color: {analyzer.color_map[category]};">'
                   f'<h3>{category}</h3></div>', unsafe_allow_html=True)
    
    # Update recommendations
    recommendations = analyzer.generate_recommendations(current_data['aqi'], current_data['pollutants'])
    with recommendation_placeholder.container():
        st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
        st.subheader("General Recommendations")
        for rec in recommendations['general_recommendations']:
            st.write(f"• {rec}")
        
        st.subheader("Specific Recommendations")
        for rec in recommendations['specific_recommendations']:
            st.write(f"• {rec}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Update anomaly detection
    try:
        # Check if anomaly detector is trained
        if anomaly_detector.load_anomaly_detector():
            anomaly_result = anomaly_detector.detect_anomalies(current_data['pollutants'])
            with anomaly_placeholder.container():
                if anomaly_result['is_anomaly']:
                    st.markdown('<div class="anomaly-alert">', unsafe_allow_html=True)
                    st.subheader("🚨 Anomaly Detected!")
                    st.write(f"Anomaly score: {anomaly_result['anomaly_score']:.3f}")
                    st.write(f"Confidence: {anomaly_result['confidence']:.3f}")
                    st.write("This reading appears unusual compared to historical patterns.")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.success("✅ No anomalies detected. Air quality readings are within expected ranges.")
        else:
            st.info("ℹ️ Anomaly detector not trained yet. Train models in sidebar.")
    except Exception as e:
        st.warning(f"Anomaly detection error: {e}")

# Initial dashboard update
update_dashboard()

# Auto-refresh logic
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

current_time = time.time()
if current_time - st.session_state.last_refresh > refresh_interval:
    st.session_state.last_refresh = current_time
    # Use the new Streamlit rerun method
    if hasattr(st, 'rerun'):
        st.rerun()
    else:
        # Fallback for older versions
        st.experimental_rerun()

# Refresh button
if st.sidebar.button("Refresh Data Now"):
    # Use the new Streamlit rerun method
    if hasattr(st, 'rerun'):
        st.rerun()
    else:
        # Fallback for older versions
        st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Environmental Monitoring System | Powered by Machine Learning</p>
        <p>Data sources: AQICN API and synthetic data</p>
    </div>
    """, 
    unsafe_allow_html=True
)