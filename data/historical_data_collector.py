import requests
import pandas as pd
import os
import time
from datetime import datetime, timedelta
import json
import numpy as np 

class HistoricalDataCollector:
    def __init__(self):
        self.api_key = os.getenv('AQICN_API_KEY', 'demo')
        self.data_path = "data/historical/"
        os.makedirs(self.data_path, exist_ok=True)
    
    def fetch_historical_data(self, city="Delhi", days=30):
        """Fetch historical data for a city"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        all_data = []
        
        # Get station ID for the city
        station_id = self.get_station_id(city)
        if not station_id:
            print(f"No station found for {city}")
            return None
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            filename = f"{self.data_path}{city.lower()}_{date_str}.json"
            
            # Check if we already have this data
            if os.path.exists(filename):
                print(f"Data for {city} on {date_str} already exists")
                current_date += timedelta(days=1)
                continue
                
            try:
                url = f"https://api.waqi.info/historical/@/{station_id}?token={self.api_key}&date={date_str}"
                response = requests.get(url)
                data = response.json()
                
                if data['status'] == 'ok':
                    # Save raw data
                    with open(filename, 'w') as f:
                        json.dump(data, f)
                    
                    # Extract and process data
                    daily_data = self.process_daily_data(data, city, date_str)
                    if daily_data:
                        all_data.append(daily_data)
                    
                    print(f"Fetched data for {city} on {date_str}")
                else:
                    print(f"Failed to fetch data for {city} on {date_str}: {data.get('data', 'Unknown error')}")
                
                # Be respectful to the API - add delay
                time.sleep(1)
                
            except Exception as e:
                print(f"Error fetching data for {city} on {date_str}: {e}")
            
            current_date += timedelta(days=1)
        
        return all_data
    
    def get_station_id(self, city):
        """Get station ID for a city"""
        # This is a simplified approach - in reality, you'd need to map cities to station IDs
        city_station_map = {
            "Delhi": "1451",
            "Mumbai": "1449", 
            "Kolkata": "1455",
            "Chennai": "1453",
            "Bangalore": "1457",
            "Hyderabad": "1463",
            "New York": "3851",
            "London": "london",
            "Beijing": "1450",
            "Tokyo": "1409"
        }
        return city_station_map.get(city, "")
    
    def process_daily_data(self, data, city, date):
        """Process daily data into a structured format"""
        try:
            # Extract relevant data
            day_data = data['data']
            hourly_data = day_data.get('forecast', {}).get('daily', {})
            
            processed = {
                'city': city,
                'date': date,
                'aqi': day_data.get('aqi', 0),
                'pm25': hourly_data.get('pm25', [{}])[0].get('avg', 0),
                'pm10': hourly_data.get('pm10', [{}])[0].get('avg', 0),
                'o3': hourly_data.get('o3', [{}])[0].get('avg', 0),
                'no2': hourly_data.get('no2', [{}])[0].get('avg', 0),
                'so2': hourly_data.get('so2', [{}])[0].get('avg', 0),
                'co': hourly_data.get('co', [{}])[0].get('avg', 0),
                'temperature': day_data.get('iaqi', {}).get('t', {}).get('v', 0),
                'humidity': day_data.get('iaqi', {}).get('h', {}).get('v', 0),
                'pressure': day_data.get('iaqi', {}).get('p', {}).get('v', 0)
            }
            
            return processed
        except Exception as e:
            print(f"Error processing data for {city} on {date}: {e}")
            return None
    
    def build_historical_dataset(self, cities, days=365):
        """Build a complete historical dataset for multiple cities"""
        all_data = []
        
        for city in cities:
            print(f"Fetching data for {city}...")
            city_data = self.fetch_historical_data(city, days)
            if city_data:
                all_data.extend(city_data)
        
        # Convert to DataFrame and save
        if all_data:
            df = pd.DataFrame(all_data)
            df.to_csv(f"{self.data_path}historical_dataset.csv", index=False)
            print(f"Saved historical dataset with {len(df)} records")
            return df
        
        return None
    
    def load_historical_data(self):
        """Load historical data from CSV"""
        csv_path = f"{self.data_path}historical_dataset.csv"
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        
        # If no CSV exists, check for JSON files
        json_files = [f for f in os.listdir(self.data_path) if f.endswith('.json')]
        if json_files:
            all_data = []
            for file in json_files:
                with open(f"{self.data_path}{file}", 'r') as f:
                    data = json.load(f)
                    city = file.split('_')[0]
                    date = file.split('_')[1].replace('.json', '')
                    processed = self.process_daily_data(data, city, date)
                    if processed:
                        all_data.append(processed)
            
            if all_data:
                df = pd.DataFrame(all_data)
                df.to_csv(csv_path, index=False)
                return df
        
        # If no data exists, generate sample data
        return self.generate_sample_historical_data()
    
    def generate_sample_historical_data(self):
        """Generate sample historical data for demonstration"""
        print("Generating sample historical data...")
        
        dates = pd.date_range(start='2020-01-01', end=datetime.now(), freq='D')
        cities = ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bangalore"]
        
        all_data = []
        for city in cities:
            for date in dates:
                # Seasonal pattern with some randomness
                day_of_year = date.timetuple().tm_yday
                seasonal_factor = 0.5 + 0.5 * np.sin(2 * np.pi * day_of_year / 365)
                
                # City-specific baseline pollution
                city_factors = {
                    "Delhi": 1.5,
                    "Mumbai": 1.2,
                    "Kolkata": 1.4,
                    "Chennai": 1.1,
                    "Bangalore": 1.0
                }
                
                base_pm25 = 50 * city_factors[city]
                base_pm10 = 70 * city_factors[city]
                
                # Add some randomness and weekend effect
                weekday = date.weekday()
                weekend_factor = 0.8 if weekday >= 5 else 1.0
                random_factor = np.random.uniform(0.8, 1.2)
                
                pm25 = base_pm25 * seasonal_factor * weekend_factor * random_factor
                pm10 = base_pm10 * seasonal_factor * weekend_factor * random_factor
                
                # Other pollutants correlated with PM levels
                o3 = np.random.uniform(20, 80) * (1 + 0.2 * seasonal_factor)
                no2 = pm25 * 0.7 + np.random.uniform(5, 15)
                so2 = pm25 * 0.3 + np.random.uniform(2, 8)
                co = pm25 * 0.05 + np.random.uniform(0.5, 1.5)
                
                # Weather data
                temperature = 15 + 20 * seasonal_factor + np.random.uniform(-5, 5)
                humidity = 50 + 30 * (1 - seasonal_factor) + np.random.uniform(-10, 10)
                pressure = 1010 + np.random.uniform(-10, 10)
                
                # AQI calculation (simplified)
                aqi = max(pm25, pm10 * 0.8, o3 * 1.2, no2 * 1.5, so2 * 2, co * 10)
                
                all_data.append({
                    'city': city,
                    'date': date.strftime("%Y-%m-%d"),
                    'aqi': aqi,
                    'pm25': pm25,
                    'pm10': pm10,
                    'o3': o3,
                    'no2': no2,
                    'so2': so2,
                    'co': co,
                    'temperature': temperature,
                    'humidity': humidity,
                    'pressure': pressure
                })
        
        df = pd.DataFrame(all_data)
        df.to_csv(f"{self.data_path}historical_dataset.csv", index=False)
        return df