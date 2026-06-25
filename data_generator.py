import numpy as np
import pandas as pd
import datetime
import os
import requests

def get_real_rainfall_data(year=2026):
    """
    Fetches real historical rainfall data for Chennai (13.08 N, 80.27 E) for the year 2025
    from the Open-Meteo Archive API and maps it to the simulation year.
    Caches the results to 'chennai_rainfall_real.csv'.
    If the API fetch fails and there is no cache, returns (None, None) to fall back to synthetic.
    """
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chennai_rainfall_real.csv')
    
    # Try reading from cache first
    if os.path.exists(cache_path):
        try:
            cache_df = pd.read_csv(cache_path)
            if len(cache_df) == 8760:
                print(f"[Rainfall] Loaded cached real rainfall data from {cache_path}")
                return cache_df['rainfall_mmph'].values.tolist(), cache_df['season'].values.tolist()
        except Exception as e:
            print(f"[Rainfall] Error reading cache file: {e}. Re-fetching...")
            
    # Fetch from Open-Meteo Archive API for 2025 (a complete 365-day year)
    print("[Rainfall] Fetching real historical 2025 rainfall data from Open-Meteo API...")
    url = "https://archive-api.open-meteo.com/v1/archive?latitude=13.08&longitude=80.27&start_date=2025-01-01&end_date=2025-12-31&daily=precipitation_sum&timezone=Asia%2FKolkata"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            daily_rain = data['daily']['precipitation_sum']
            daily_times = data['daily']['time']
            
            # Map daily rainfall to hourly using diurnal patterns
            hourly_rain = []
            seasons_list = []
            
            # Helper arrays for diurnal weight patterns
            h_arr = np.arange(24)
            # SW Monsoon: peaks in afternoon/evening (centered at 5 PM / hour 17)
            W_sw = np.exp(-0.5 * ((h_arr - 17) / 3.0) ** 2)
            W_sw = W_sw / W_sw.sum()
            
            # NE Monsoon: peaks at night (centered at 2 AM / hour 2, with circular wrap-around)
            diff_ne = np.minimum(np.abs(h_arr - 2), 24 - np.abs(h_arr - 2))
            W_ne = np.exp(-0.5 * (diff_ne / 4.0) ** 2)
            W_ne = W_ne / W_ne.sum()
            
            # Winter & Summer: flat
            W_flat = np.ones(24) / 24.0
            
            start_date = datetime.datetime(year, 1, 1, 0, 0)
            
            for i, daily_val in enumerate(daily_rain):
                dt_str = daily_times[i]
                m = int(dt_str.split('-')[1])
                
                if m in [3, 4, 5]:
                    season = "Summer"
                    weights = W_flat
                elif m in [6, 7, 8, 9]:
                    season = "SW Monsoon"
                    weights = W_sw
                elif m in [10, 11, 12]:
                    season = "NE Monsoon"
                    weights = W_ne
                else:
                    season = "Winter"
                    weights = W_flat
                
                # Distribute daily rain into 24 hours
                day_rain_hourly = daily_val * weights
                hourly_rain.extend(day_rain_hourly.tolist())
                seasons_list.extend([season] * 24)
                
            # Build and cache DataFrame
            timestamps = [start_date + datetime.timedelta(hours=j) for j in range(8760)]
            cache_df = pd.DataFrame({
                'timestamp': [t.strftime('%Y-%m-%d %H:%M:%S') for t in timestamps],
                'rainfall_mmph': hourly_rain,
                'season': seasons_list
            })
            cache_df.to_csv(cache_path, index=False)
            print(f"[Rainfall] Saved real rainfall data to cache: {cache_path}")
            return hourly_rain, seasons_list
        else:
            print(f"[Rainfall] Open-Meteo API returned status code {r.status_code}. Using synthetic fallback.")
            return None, None
    except Exception as e:
        print(f"[Rainfall] API fetch failed: {e}. Using synthetic fallback.")
        return None, None

def generate_chennai_weather_and_load(year=2026):
    """
    Generates a realistic 1-year (8760 hours) dataset of weather variables
    (irradiance, temperature, rainfall, wind speed) and power load for a school campus
    in Chennai, India (13.08 N, 80.27 E).
    """
    # Start and end timestamps for the year
    start_date = datetime.datetime(year, 1, 1, 0, 0)
    timestamps = [start_date + datetime.timedelta(hours=i) for i in range(8760)]
    
    df = pd.DataFrame(index=timestamps)
    df['timestamp'] = df.index
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Initialize weather lists
    irradiance = []
    temperature = []
    rainfall = []
    grid_available = []
    seasons = []
    wind_speeds = []
    
    # Seed for reproducibility
    np.random.seed(42)
    
    # Get real rainfall data
    real_rain_list, real_season_list = get_real_rainfall_data(year)
    
    for i, dt in enumerate(timestamps):
        m = dt.month
        h = dt.hour
        is_day = 6 <= h <= 18
        
        # Calculate daily solar cycle profile (normalized 0 to 1)
        if is_day:
            solar_factor = np.sin(np.pi * (h - 6) / 12)
        else:
            solar_factor = 0.0
            
        temp = 25.0
        irr = 0.0
        rain = 0.0
        grid = 1.0 # 1 = available, 0 = cut
        
        # Determine season string and wind speed base ranges
        if m in [3, 4, 5]:
            season_str = "Summer"
            wind_speed = np.random.uniform(3.0, 6.0)
        elif m in [6, 7, 8, 9]:
            season_str = "SW Monsoon"
            wind_speed = np.random.uniform(5.0, 9.0)
        elif m in [10, 11, 12]:
            season_str = "NE Monsoon"
            wind_speed = np.random.uniform(7.0, 14.0)
        else:
            season_str = "Winter"
            wind_speed = np.random.uniform(2.0, 5.0)
            
        # Model temperature and clear-sky irradiance
        if season_str == "Summer":
            peak_irr = np.random.uniform(900, 1000)
            irr = solar_factor * peak_irr if is_day else 0.0
            temp_range = np.random.uniform(35, 42) - 28.0
            temp = 28.0 + temp_range * solar_factor + np.random.normal(0, 0.5)
            
        elif season_str == "SW Monsoon":
            peak_irr = np.random.uniform(200, 400)
            irr = solar_factor * peak_irr if is_day else 0.0
            temp_range = np.random.uniform(32, 34) - 27.0
            temp = 27.0 + temp_range * solar_factor + np.random.normal(0, 0.5)
            
        elif season_str == "NE Monsoon":
            peak_irr = np.random.uniform(100, 250)
            irr = solar_factor * peak_irr if is_day else 0.0
            temp_range = np.random.uniform(28, 30) - 23.0
            temp = 23.0 + temp_range * solar_factor + np.random.normal(0, 0.5)
            
        else: # Winter
            peak_irr = np.random.uniform(700, 900)
            irr = solar_factor * peak_irr if is_day else 0.0
            temp_range = np.random.uniform(26, 28) - 21.0
            temp = 21.0 + temp_range * solar_factor + np.random.normal(0, 0.5)
            
        # Apply rainfall
        if real_rain_list is not None:
            rain = real_rain_list[i]
            # Couple real rainfall to irradiance reduction
            if rain > 0.1:
                if season_str == "SW Monsoon":
                    irr *= 0.25
                elif season_str == "NE Monsoon":
                    irr *= 0.10
                elif season_str == "Winter":
                    irr *= 0.80
            
            # Severe Grid Outages based on real rain
            if season_str == "NE Monsoon":
                if rain > 20.0:
                    grid = 0.0
                elif np.random.rand() < 0.10:
                    grid = 0.0
        else:
            # Fallback to synthetic rain
            if season_str == "Summer":
                rain = 0.0
            elif season_str == "SW Monsoon":
                if np.random.rand() < 0.15:
                    rain = np.random.uniform(5, 30)
                    irr *= 0.25
            elif season_str == "NE Monsoon":
                if np.random.rand() < 0.35:
                    rain = np.random.uniform(10, 80)
                    irr *= 0.10
                if rain > 20.0:
                    grid = 0.0
                elif np.random.rand() < 0.10:
                    grid = 0.0
            else: # Winter
                if np.random.rand() < 0.01:
                    rain = np.random.uniform(0.5, 3.0)
                    irr *= 0.8
                    
        irradiance.append(irr)
        temperature.append(temp)
        rainfall.append(rain)
        grid_available.append(grid)
        seasons.append(season_str)
        wind_speeds.append(wind_speed)
        
    df['solar_irradiance'] = irradiance
    df['ambient_temp'] = temperature
    df['rainfall_rate'] = rainfall
    df['rainfall_mmph'] = rainfall
    df['season'] = seasons
    df['wind_speed_mps'] = wind_speeds
    df['grid_available'] = grid_available
    
    # --- School Load Profile Modeling ---
    # School load depends on:
    # 1. Day of the week (weekend vs weekday)
    # 2. Hour of the day (school hours: 8 AM - 4 PM)
    # 3. Season (vacations: Summer in May, Winter Break in mid-to-late Dec)
    # 4. Ambient Temperature (AC load scaling)
    
    loads = []
    for idx, row in df.iterrows():
        h = row['hour']
        m = row['month']
        is_we = row['is_weekend']
        t_amb = row['ambient_temp']
        d = row['day']
        
        # Base load (security, servers, standby)
        load = np.random.uniform(6.0, 9.0)
        
        # Check if school is in vacation
        is_summer_vacation = (m == 5)
        is_winter_vacation = (m == 12 and d >= 15) or (m == 1 and d <= 5)
        
        if is_we:
            # Weekends have minimal load, slightly higher than base due to security
            load += np.random.uniform(1.0, 3.0)
        elif is_summer_vacation or is_winter_vacation:
            # Vacation weekdays: admin staff only (reduced load)
            if 9 <= h <= 15:
                load += np.random.uniform(10.0, 15.0) # Admin load
                # Moderate AC for admin offices
                ac_load = max(0.0, (t_amb - 28.0) * 0.5)
                load += ac_load
        else:
            # Active school days
            if 8 <= h <= 16:
                # Main academic hours (classes, labs, lighting, fans)
                load += np.random.uniform(45.0, 60.0)
                # AC cooling load depending on ambient temperature
                # In Chennai, high temperatures (T > 27) demand substantial AC
                ac_load = max(0.0, (t_amb - 27.0) * 1.8)
                load += ac_load
            elif h == 7 or h == 17:
                # Arrival/departure hours (moderate load)
                load += np.random.uniform(15.0, 25.0)
                
        loads.append(load)
        
    df['school_load'] = loads
    
    # Let's clean up grid availability transitions to make grid cuts continuous (e.g. 2-6 hour blocks)
    # This prevents grid flipping on and off every hour in the NE Monsoon.
    consecutive_cuts = 0
    grid_states = df['grid_available'].values.copy()
    for idx in range(1, len(grid_states)):
        # If previous hour was a cut and we have a cut trigger, let's extend the cut
        if grid_states[idx-1] == 0.0 and consecutive_cuts < 4:
            # 70% probability of continuing the cut
            if np.random.rand() < 0.7:
                grid_states[idx] = 0.0
                consecutive_cuts += 1
            else:
                consecutive_cuts = 0
        elif grid_states[idx] == 0.0:
            consecutive_cuts = 1
            
    df['grid_available'] = grid_states
    
    return df

if __name__ == "__main__":
    # Test generation and print summary stats
    print("Testing Chennai Weather and Load Generation...")
    data = generate_chennai_weather_and_load()
    print(data.describe())
    print("\nNortheast Monsoon (Oct-Dec) stats:")
    ne_monsoon = data[data['month'].isin([10, 11, 12])]
    print(ne_monsoon[['solar_irradiance', 'rainfall_rate', 'grid_available', 'school_load']].describe())
    print(f"Total simulated grid cuts: {int((data['grid_available'] == 0).sum())} hours out of 8760.")
