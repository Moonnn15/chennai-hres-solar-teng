import numpy as np
import pandas as pd
from models import calculate_solar_power, calculate_teng_power, BatteryStorage, WindTurbine

class HRESSimulator:
    """
    Simulates the Hybrid Renewable Energy System (HRES) operation
    hour-by-hour over the full year.
    """
    def __init__(self, battery_capacity=250.0, pv_area=500.0, teng_area=500.0, num_wind_turbines=10):
        self.battery_capacity = battery_capacity
        self.pv_area = pv_area
        self.teng_area = teng_area
        self.num_wind_turbines = num_wind_turbines
        self.wind_turbine = WindTurbine()
        
    def run_simulation(self, df_data):
        """
        Runs the simulation using weather, load, and forecasted load.
        
        Controls battery dispatch based on current generation, school load,
        grid availability, and AI load forecast.
        """
        # Initialize battery
        battery = BatteryStorage(capacity_kwh=self.battery_capacity, initial_soc=0.5)
        
        # Lists to store hourly results
        solar_gen = []
        wind_gen = []
        teng_gen = []
        bat_soc = []
        bat_power = []  # positive: charging, negative: discharging
        bat_loss = []
        grid_import = []
        unmet_load = []
        surplus_curtailed = []
        energy_dispatch_source = []  # logs dominant energy source
        
        for idx, row in df_data.iterrows():
            irr = row['solar_irradiance']
            temp = row['ambient_temp']
            rain = row['rainfall_rate']
            grid_avail = row['grid_available']
            load = row['school_load']
            pred_load = row['predicted_load']
            month = row['month']
            
            # 1. Calculate generation
            p_pv = calculate_solar_power(irradiance=irr, ambient_temp=temp, area=self.pv_area)
            p_wind = self.wind_turbine.calculate_power(row.get('wind_speed_mps', 0.0)) * self.num_wind_turbines
            p_teng = calculate_teng_power(rainfall_rate=rain, area=self.teng_area)
            
            p_gen = p_pv + p_wind + p_teng
            net_power = p_gen - load
            
            p_bat_exchanged = 0.0
            p_bat_loss = 0.0
            p_grid_imp = 0.0
            p_unmet = 0.0
            p_surplus = 0.0
            source = "Grid"
            
            # --- AI CONTROLLER LOGIC ---
            # The AI controller predicts upcoming load using LSTM and acts defensively during
            # the Northeast Monsoon (Oct-Dec) to prevent blackouts when grid failures are likely.
            
            is_monsoon = month in [10, 11, 12]
            
            if grid_avail == 0:
                # --- GRID IS DOWN (Outage) ---
                # We must rely on local generation and battery
                if net_power >= 0:
                    # Generation covers load; charge battery with surplus
                    p_bat_exchanged, p_bat_loss = battery.step(net_power)
                    # Remaining excess power is curtailed/wasted
                    p_surplus = max(0.0, net_power - (p_bat_exchanged / battery.eta_charge))
                    source = "Solar/Wind/TENG (Surplus)"
                else:
                    # Generation is insufficient; discharge battery to meet deficit
                    deficit = abs(net_power)
                    p_bat_exchanged, p_bat_loss = battery.step(-deficit)
                    
                    # Check if load is fully met
                    met_by_battery = abs(p_bat_exchanged)
                    remaining_deficit = deficit - met_by_battery
                    
                    if remaining_deficit > 0.01:
                        p_unmet = remaining_deficit
                        source = "Unmet (Blackout)"
                    else:
                        source = "Battery Backup"
            else:
                # --- GRID IS AVAILABLE ---
                if net_power >= 0:
                    # Generation covers load; charge battery, no grid import
                    p_bat_exchanged, p_bat_loss = battery.step(net_power)
                    p_surplus = max(0.0, net_power - (p_bat_exchanged / battery.eta_charge))
                    source = "Solar/Wind/TENG"
                else:
                    # Generation is insufficient. We have a choice: discharge battery or import grid power.
                    # AI Rule: If in Monsoon and the battery is getting low (SOC < 40%), and upcoming forecasted
                    # school load is high (average > 30 kW), we IMPORT from grid to preserve battery charge for
                    # imminent grid outages. Otherwise, we discharge the battery to save money.
                    
                    deficit = abs(net_power)
                    
                    # Check next 4 hours load forecast
                    # We locate the next indices in df_data (up to 4 steps ahead)
                    current_pos = df_data.index.get_loc(idx)
                    future_indices = df_data.index[current_pos + 1 : min(current_pos + 5, len(df_data))]
                    
                    if len(future_indices) > 0:
                        avg_future_load = df_data.loc[future_indices, 'predicted_load'].mean()
                    else:
                        avg_future_load = pred_load
                        
                    preserve_battery = is_monsoon and (battery.soc < 0.40) and (avg_future_load > 30.0)
                    
                    if preserve_battery:
                        # Preserve battery: support load using grid import
                        p_grid_imp = deficit
                        source = "Grid (Preserving Battery)"
                    else:
                        # Standard economic dispatch: discharge battery first
                        p_bat_exchanged, p_bat_loss = battery.step(-deficit)
                        met_by_battery = abs(p_bat_exchanged)
                        remaining_deficit = deficit - met_by_battery
                        
                        if remaining_deficit > 0.01:
                            p_grid_imp = remaining_deficit
                            source = "Grid + Battery"
                        else:
                            source = "Battery"
                            
            # Record state
            solar_gen.append(p_pv)
            wind_gen.append(p_wind)
            teng_gen.append(p_teng)
            bat_soc.append(battery.soc)
            bat_power.append(p_bat_exchanged)
            bat_loss.append(p_bat_loss)
            grid_import.append(p_grid_imp)
            unmet_load.append(p_unmet)
            surplus_curtailed.append(p_surplus)
            energy_dispatch_source.append(source)
            
        # Compile results back into the DataFrame
        df_sim = df_data.copy()
        df_sim['solar_gen_kw'] = solar_gen
        df_sim['wind_gen_kw'] = wind_gen
        df_sim['teng_gen_kw'] = teng_gen
        df_sim['total_gen_kw'] = df_sim['solar_gen_kw'] + df_sim['wind_gen_kw'] + df_sim['teng_gen_kw']
        df_sim['battery_soc'] = bat_soc
        df_sim['battery_power_kw'] = bat_power
        df_sim['battery_loss_kw'] = bat_loss
        df_sim['grid_import_kw'] = grid_import
        df_sim['unmet_load_kw'] = unmet_load
        df_sim['surplus_curtailed_kw'] = surplus_curtailed
        df_sim['dispatch_source'] = energy_dispatch_source
        
        return df_sim
        
    def calculate_kpis(self, df_sim, diesel_co2_factor=0.8, electricity_tariff_inr=8.0):
        """
        Calculates annual and monthly key performance indicators (KPIs)
        from simulation results.
        """
        # Sums representing total kWh since time step is 1 hour
        total_load = df_sim['school_load'].sum()
        total_solar = df_sim['solar_gen_kw'].sum()
        total_wind = df_sim['wind_gen_kw'].sum()
        total_teng = df_sim['teng_gen_kw'].sum()
        total_gen = total_solar + total_wind + total_teng
        
        total_grid_import = df_sim['grid_import_kw'].sum()
        total_unmet = df_sim['unmet_load_kw'].sum()
        total_surplus = df_sim['surplus_curtailed_kw'].sum()
        total_bat_loss = df_sim['battery_loss_kw'].sum()
        
        # Renewable energy consumed directly or from battery
        # This is Total Load - Grid Import - Unmet Load
        re_consumed = total_load - total_grid_import - total_unmet
        
        # 1. Renewable Energy Fraction (%)
        renewable_fraction = (re_consumed / total_load) * 100.0 if total_load > 0 else 0.0
        
        # 2. CO2 Emissions Saved vs Diesel (kg CO2)
        co2_saved_kg = re_consumed * diesel_co2_factor
        
        # 3. Cost Savings in INR (based on tariff)
        cost_savings_inr = re_consumed * electricity_tariff_inr
        
        # 4. System Efficiency (%)
        useful_gen = total_gen - total_surplus - total_bat_loss
        system_efficiency = (useful_gen / total_gen) * 100.0 if total_gen > 0 else 0.0
        
        kpis = {
            'total_load_kwh': total_load,
            'total_solar_kwh': total_solar,
            'total_wind_kwh': total_wind,
            'total_teng_kwh': total_teng,
            'total_gen_kwh': total_gen,
            'grid_import_kwh': total_grid_import,
            'unmet_load_kwh': total_unmet,
            'surplus_curtailed_kwh': total_surplus,
            're_consumed_kwh': re_consumed,
            'renewable_fraction_pct': renewable_fraction,
            'co2_saved_kg': co2_saved_kg,
            'system_efficiency_pct': system_efficiency,
            'cost_savings_inr': cost_savings_inr
        }
        
        return kpis

if __name__ == "__main__":
    # Test simulation execution
    from data_generator import generate_chennai_weather_and_load
    print("Testing Simulation...")
    data = generate_chennai_weather_and_load()
    data['predicted_load'] = data['school_load'] # Dummy forecaster
    
    simulator = HRESSimulator()
    results = simulator.run_simulation(data)
    kpis = simulator.calculate_kpis(results)
    
    print("\nSimulation completed successfully!")
    print(f"Total Load: {kpis['total_load_kwh']:.2f} kWh")
    print(f"Solar Generation: {kpis['total_solar_kwh']:.2f} kWh")
    print(f"Wind Generation: {kpis['total_wind_kwh']:.2f} kWh")
    print(f"TENG Generation: {kpis['total_teng_kwh']:.2f} kWh")
    print(f"Renewable Fraction: {kpis['renewable_fraction_pct']:.2f}%")
    print(f"CO2 Saved: {kpis['co2_saved_kg']:.2f} kg")
    print(f"System Efficiency: {kpis['system_efficiency_pct']:.2f}%")
    print(f"Cost Savings: INR {kpis['cost_savings_inr']:,.2f}")
