import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

class DataAnalyzer:
    def __init__(self):
        self.color_map = {
            'Good': '#00E400',
            'Moderate': '#FFFF00',
            'Unhealthy for Sensitive Groups': '#FF7E00',
            'Unhealthy': '#FF0000',
            'Very Unhealthy': '#8F3F97',
            'Hazardous': '#7E0023'
        }
    
    def get_aqi_category(self, aqi):
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
    
    def create_pollution_dashboard(self, current_data, historical_data):
        """Create pollution dashboard with various visualizations"""
        # Current AQI Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = current_data['aqi'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Current AQI - {current_data['city']}", 'font': {'size': 24}},
            delta = {'reference': 50, 'increasing': {'color': "RebeccaPurple"}},
            gauge = {
                'axis': {'range': [None, 500], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': self.color_map['Good']},
                    {'range': [50, 100], 'color': self.color_map['Moderate']},
                    {'range': [100, 150], 'color': self.color_map['Unhealthy for Sensitive Groups']},
                    {'range': [150, 200], 'color': self.color_map['Unhealthy']},
                    {'range': [200, 300], 'color': self.color_map['Very Unhealthy']},
                    {'range': [300, 500], 'color': self.color_map['Hazardous']}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': current_data['aqi']}}))
        
        fig_gauge.update_layout(height=400)
        
        # Pollutant breakdown
        pollutants = current_data['pollutants']
        pollutant_names = list(pollutants.keys())
        pollutant_values = list(pollutants.values())
        
        fig_pollutants = go.Figure([go.Bar(
            x=pollutant_names, 
            y=pollutant_values,
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
        )])
        
        fig_pollutants.update_layout(
            title="Current Pollutant Levels",
            xaxis_title="Pollutant",
            yaxis_title="Concentration",
            height=400
        )
        
        # Historical trends
        if not historical_data.empty:
            # Make sure we have a datetime index
            historical_data = historical_data.copy()
            
            # Convert date column to datetime if it's not already
            if 'date' in historical_data.columns:
                historical_data['date'] = pd.to_datetime(historical_data['date'])
                historical_data = historical_data.set_index('date')
            
            # Select only numeric columns for resampling
            numeric_columns = historical_data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_columns) > 0:
                # Resample to weekly averages for numeric columns only
                weekly_data = historical_data[numeric_columns].resample('W').mean()
                
                fig_trend = go.Figure()
                
                # Add traces for key pollutants if they exist
                if 'pm25' in numeric_columns:
                    fig_trend.add_trace(go.Scatter(
                        x=weekly_data.index, 
                        y=weekly_data['pm25'],
                        mode='lines',
                        name='PM2.5',
                        line=dict(color='#1f77b4', width=2)
                    ))
                
                if 'pm10' in numeric_columns:
                    fig_trend.add_trace(go.Scatter(
                        x=weekly_data.index, 
                        y=weekly_data['pm10'],
                        mode='lines',
                        name='PM10',
                        line=dict(color='#ff7f0e', width=2)
                    ))
                
                if 'aqi' in numeric_columns:
                    fig_trend.add_trace(go.Scatter(
                        x=weekly_data.index, 
                        y=weekly_data['aqi'],
                        mode='lines',
                        name='AQI',
                        line=dict(color='#d62728', width=2)
                    ))
                
                fig_trend.update_layout(
                    title="Historical Pollution Trends (Weekly Averages)",
                    xaxis_title="Date",
                    yaxis_title="Concentration",
                    height=400
                )
            else:
                fig_trend = self.create_empty_plot("No numeric data available for trends")
        else:
            fig_trend = self.create_empty_plot("No historical data available")
        
        return fig_gauge, fig_pollutants, fig_trend
    
    def create_empty_plot(self, message):
        """Create an empty plot with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            height=400,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig
    
    def generate_recommendations(self, aqi, pollutants):
        """Generate recommendations based on pollution levels"""
        category = self.get_aqi_category(aqi)
        
        recommendations = {
            'Good': [
                "Air quality is satisfactory",
                "Ideal time for outdoor activities",
                "No health impacts expected"
            ],
            'Moderate': [
                "Acceptable air quality",
                "Unusually sensitive people should consider reducing prolonged outdoor exertion",
                "Generally acceptable for most activities"
            ],
            'Unhealthy for Sensitive Groups': [
                "Members of sensitive groups may experience health effects",
                "General public is not likely to be affected",
                "People with heart or lung disease, older adults, and children should reduce prolonged outdoor exertion"
            ],
            'Unhealthy': [
                "Everyone may begin to experience health effects",
                "Members of sensitive groups may experience more serious health effects",
                "Reduce outdoor activities, especially for sensitive groups"
            ],
            'Very Unhealthy': [
                "Health alert: everyone may experience more serious health effects",
                "Avoid outdoor activities",
                "Use air purifiers indoors and keep windows closed"
            ],
            'Hazardous': [
                "Health warning of emergency conditions",
                "Everyone is likely to be affected",
                "Stay indoors and keep activity levels low"
            ]
        }
        
        # Add specific recommendations based on dominant pollutants
        dominant_pollutant = max(pollutants.items(), key=lambda x: x[1])[0]
        
        specific_recommendations = {
            'pm25': [
                "Use high-efficiency particulate air (HEPA) filters",
                "Avoid burning candles, smoking, or frying food",
                "Consider wearing N95 masks outdoors"
            ],
            'pm10': [
                "Damp dust and vacuum regularly with HEPA filter",
                "Keep windows closed during high pollen seasons",
                "Use air purifiers with pre-filters"
            ],
            'o3': [
                "Schedule outdoor activities for morning or evening when ozone levels are lower",
                "Avoid strenuous outdoor activities during afternoon hours",
                "Be aware of ozone alert days"
            ],
            'no2': [
                "Ensure proper ventilation when using gas stoves or heaters",
                "Avoid idling vehicles in enclosed spaces",
                "Consider using electric appliances instead of gas"
            ],
            'so2': [
                "Stay indoors during high SO2 episodes",
                "Use air conditioning with clean filters",
                "Be aware of industrial emissions in your area"
            ],
            'co': [
                "Ensure proper ventilation when using fuel-burning appliances",
                "Have heating systems professionally inspected annually",
                "Never use generators or grills indoors"
            ]
        }
        
        return {
            'category': category,
            'general_recommendations': recommendations[category],
            'specific_recommendations': specific_recommendations.get(dominant_pollutant, [])
        }