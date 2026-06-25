import numpy as np
import pandas as pd
import datetime

def generate_chennai_weather_and_load(year=2026):
    """
    Generates a realistic 1-year (8760 hours) dataset of weather variables
    (irradiance, temperature, rainfall) and power load for a school campus
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
    
    # Seed for reproducibility
    np.random.seed(42)
    
    for dt in timestamps:
        m = dt.month
        h = dt.hour
        is_day = 6 <= h <= 18
        
        # Calculate daily solar cycle profile (normalized 0 to 1)
        if is_day:
            # Simple sine wave representation of daytime solar path
            solar_factor = np.sin(np.pi * (h - 6) / 12)
        else:
            solar_factor = 0.0
            
        # Initialize hourly state variables
        temp = 25.0
        irr = 0.0
        rain = 0.0
        grid = 1.0 # 1 = available, 0 = cut
        
        # --- Seasonal Weather Modeling ---
        
        # 1. Summer (March - May: months 3, 4, 5)
        if m in [3, 4, 5]:
            # Peak irradiance: 900 - 1000 W/m²
            peak_irr = np.random.uniform(900, 1000)
            irr = solar_factor * peak_irr if is_day else 0.0
            
            # Temperature: 35 - 42 °C in afternoon, cooler at night (27-30 °C)
            temp_range = np.random.uniform(35, 42) - 28.0
            temp = 28.0 + temp_range * solar_factor + np.random.normal(0, 0.5)
            
            # Rainfall: Zero
            rain = 0.0
            
        # 2. Southwest Monsoon (June - September: months 6, 7, 8, 9)
        elif m in [6, 7, 8, 9]:
            # Moderate irradiance: 200 - 400 W/m² peak (partially cloudy)
            peak_irr = np.random.uniform(200, 400)
            irr = solar_factor * peak_irr if is_day else 0.0
            
            # Temperature: 28 - 34 °C
            temp_range = np.random.uniform(32, 34) - 27.0
            temp = 27.0 + temp_range * solar_factor + np.random.normal(0, 0.5)
            
            # Rainfall: Moderate showers. Assume 30% chance of rain on any day,
            # and if it rains, it occurs for a few hours.
            # We'll use a Markov-like process or random thresholding.
            # We determine rain based on an hourly probability
            if np.random.rand() < 0.15:  # 15% chance of rain in any given hour
                rain = np.random.uniform(5, 30)
                # Cloud cover reduces irradiance by 75%
                irr *= 0.25
                
        # 3. Northeast Monsoon (October - December: months 10, 11, 12)
        elif m in [10, 11, 12]:
            # Low solar irradiance: 100 - 250 W/m² peak (thick cloud cover)
            peak_irr = np.random.uniform(100, 250)
            irr = solar_factor * peak_irr if is_day else 0.0
            
            # Temperature: 24 - 30 °C
            temp_range = np.random.uniform(28, 30) - 23.0
            temp = 23.0 + temp_range * solar_factor + np.random.normal(0, 0.5)
            
            # Rainfall: Heavy cyclonic rainfall.
            # 35% chance of hourly rainfall.
            if np.random.rand() < 0.35:
                # Heavy rainfall: 10 to 80 mm/h
                rain = np.random.uniform(10, 80)
                # Irradiance drops by 90% (virtually zero)
                irr *= 0.10
                
            # Severe Grid Outages:
            # During heavy rainfall (rain > 20 mm/h), the grid fails (trips).
            # Otherwise, there is a 10% chance of random grid failure during these months.
            if rain > 20.0:
                grid = 0.0
            elif np.random.rand() < 0.10:
                grid = 0.0
                
        # 4. Winter (January - February: months 1, 2)
        else:
            # High solar irradiance: 700 - 900 W/m² peak
            peak_irr = np.random.uniform(700, 900)
            irr = solar_factor * peak_irr if is_day else 0.0
            
            # Temperature: 22 - 28 °C
            temp_range = np.random.uniform(26, 28) - 21.0
            temp = 21.0 + temp_range * solar_factor + np.random.normal(0, 0.5)
            
            # Rainfall: Minimal/none. 1% chance of drizzle
            if np.random.rand() < 0.01:
                rain = np.random.uniform(0.5, 3.0)
                irr *= 0.8 # minor drop
                
        irradiance.append(irr)
        temperature.append(temp)
        rainfall.append(rain)
        grid_available.append(grid)
        
    df['solar_irradiance'] = irradiance
    df['ambient_temp'] = temperature
    df['rainfall_rate'] = rainfall
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
