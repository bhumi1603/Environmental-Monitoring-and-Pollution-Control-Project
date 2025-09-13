import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json
import time

load_dotenv()

class DataCollector:
    def __init__(self):
        self.api_key = os.getenv('AQICN_API_KEY', 'demo')
        self.data_path = "data/"
        self.historical_path = f"{self.data_path}historical/"
        os.makedirs(self.historical_path, exist_ok=True)
        
    def fetch_air_quality_data(self, city="Delhi"):
        """Fetch real-time air quality data from AQICN API"""
        try:
            url = f"https://api.waqi.info/feed/{city}/?token={self.api_key}"
            response = requests.get(url)
            data = response.json()
            
            if data['status'] == 'ok':
                aqi = data['data']['aqi']
                iaqi = data['data']['iaqi']
                
                # Extract pollutant data
                pollutants = {
                    'pm25': iaqi.get('pm25', {}).get('v', np.nan),
                    'pm10': iaqi.get('pm10', {}).get('v', np.nan),
                    'o3': iaqi.get('o3', {}).get('v', np.nan),
                    'no2': iaqi.get('no2', {}).get('v', np.nan),
                    'so2': iaqi.get('so2', {}).get('v', np.nan),
                    'co': iaqi.get('co', {}).get('v', np.nan),
                    'temperature': iaqi.get('t', {}).get('v', np.nan),
                    'humidity': iaqi.get('h', {}).get('v', np.nan),
                    'pressure': iaqi.get('p', {}).get('v', np.nan)
                }
                
                return {
                    'city': city,
                    'aqi': aqi,
                    'pollutants': pollutants,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            else:
                return self.generate_sample_data(city)
                
        except Exception as e:
            print(f"Error fetching data: {e}")
            return self.generate_sample_data(city)
    
    def generate_sample_data(self, city):
        """Generate sample data if API fails"""
        return {
            'city': city,
            'aqi': np.random.randint(50, 300),
            'pollutants': {
                'pm25': np.random.uniform(10, 200),
                'pm10': np.random.uniform(20, 300),
                'o3': np.random.uniform(20, 150),
                'no2': np.random.uniform(10, 100),
                'so2': np.random.uniform(5, 80),
                'co': np.random.uniform(0.5, 15),
                'temperature': np.random.uniform(15, 35),
                'humidity': np.random.uniform(30, 90),
                'pressure': np.random.uniform(1000, 1020)
            },
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
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
            filename = f"{self.historical_path}{city.lower()}_{date_str}.json"
            
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
            df.to_csv(f"{self.historical_path}historical_dataset.csv", index=False)
            print(f"Saved historical dataset with {len(df)} records")
            return df
        
        return None
    
    def load_historical_data(self):
        """Load historical environmental data"""
        try:
            # First try to load from CSV
            csv_path = f"{self.historical_path}historical_dataset.csv"
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # Ensure numeric columns are properly typed
                numeric_columns = ['aqi', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 
                                  'temperature', 'humidity', 'pressure']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                return df

            # If no CSV exists, check for JSON files
            json_files = [f for f in os.listdir(self.historical_path) if f.endswith('.json')]
            if json_files:
                all_data = []
                for file in json_files:
                    with open(f"{self.historical_path}{file}", 'r') as f:
                        data = json.load(f)
                        # Process the JSON data based on its structure
                        processed = self.process_historical_json(data, file)
                        if processed:
                            all_data.append(processed)

                if all_data:
                    df = pd.DataFrame(all_data)
                    # Ensure numeric columns are properly typed
                    numeric_columns = ['aqi', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 
                                      'temperature', 'humidity', 'pressure']
                    for col in numeric_columns:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    df.to_csv(csv_path, index=False)
                    return df

            # If no data exists, generate sample data
            return self.generate_sample_historical_data()

        except Exception as e:
            print(f"Error loading historical data: {e}")
            return self.generate_sample_historical_data()

    def process_historical_json(self, data, filename):
        """Process historical JSON data into a standardized format"""
        try:
            # Extract city and date from filename
            name_parts = filename.replace('.json', '').split('_')
            city = name_parts[0]
            date = name_parts[1] if len(name_parts) > 1 else datetime.now().strftime("%Y-%m-%d")
            
            # Process based on data structure
            if 'data' in data and 'aqi' in data['data']:
                # AQICN format
                day_data = data['data']
                hourly_data = day_data.get('forecast', {}).get('daily', {})
                
                return {
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
            else:
                # Other formats
                return None
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            return None
    
    def generate_sample_historical_data(self):
        """Generate sample historical data for demonstration"""
        print("Generating sample historical data...")
        
        dates = pd.date_range(start='2020-01-01', end=datetime.now(), freq='D')
        cities = ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bangalore", "Hyderabad"]
        
        all_data = []
        for city in cities:
            for date in dates:
                # Create realistic seasonal patterns
                day_of_year = date.timetuple().tm_yday
                seasonal_factor = 0.5 + 0.5 * np.sin(2 * np.pi * day_of_year / 365)
                
                # City-specific baseline pollution
                city_factors = {
                    "Delhi": 1.5,
                    "Mumbai": 1.2,
                    "Kolkata": 1.4,
                    "Chennai": 1.1,
                    "Bangalore": 1.0,
                    "Hyderabad": 1.3
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
        df.to_csv(f"{self.historical_path}historical_dataset.csv", index=False)
        return df

# Example usage
if __name__ == "__main__":
    collector = DataCollector()
    
    # Test real-time data
    current_data = collector.fetch_air_quality_data("Delhi")
    print("Current data:", current_data)
    
    # Test historical data loading
    historical_data = collector.load_historical_data()
    print("Historical data shape:", historical_data.shape if historical_data is not None else "No data")
    
    # To fetch new historical data (uncomment to use)
    # cities = ["Delhi", "Mumbai", "Kolkata"]
    # collector.build_historical_dataset(cities, days=30)