import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from data_generator import generate_chennai_weather_and_load
from lstm_forecaster import LSTMForecaster
from simulation import HRESSimulator
from models import calculate_teng_power

# Set matplotlib style for premium dark-themed aesthetics
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['figure.facecolor'] = '#121212'
plt.rcParams['axes.facecolor'] = '#1e1e1e'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['grid.color'] = '#444444'
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['text.color'] = '#ffffff'
plt.rcParams['xtick.color'] = '#bbbbbb'
plt.rcParams['ytick.color'] = '#bbbbbb'

# Output paths
PROJECT_PLOT_DIR = r"C:\Users\monalisa\.gemini\antigravity\scratch\hres_simulation\plots"
ARTIFACT_DIR = r"C:\Users\monalisa\.gemini\antigravity\brain\465e6851-e228-48cb-a7fd-2d3bf709c787"

# Create output directories if they don't exist
os.makedirs(PROJECT_PLOT_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def save_plot(filename):
    """Saves the active plot to both the project folder and the artifact directory."""
    p1 = os.path.join(PROJECT_PLOT_DIR, filename)
    p2 = os.path.join(ARTIFACT_DIR, filename)
    plt.savefig(p1, dpi=300, bbox_inches='tight')
    plt.savefig(p2, dpi=300, bbox_inches='tight')
    print(f"Exported: {filename} (300 DPI)")
    plt.close()

def main():
    print("==================================================================")
    print("       CHENNAI SOLAR-TENG HYBRID RENEWABLE ENERGY SYSTEM SIM      ")
    print("==================================================================")
    
    # 1. Generate Climate and Load Data
    print("\n[Step 1/4] Generating Chennai climate and school load data...")
    df_raw = generate_chennai_weather_and_load(year=2026)
    print(f"Data generation complete. Total records: {len(df_raw)} hours.")
    
    # 2. Train LSTM Load Forecaster
    print("\n[Step 2/4] Initializing and training AI LSTM load forecaster...")
    forecaster = LSTMForecaster(lookback=24)
    # Using 6 epochs to speed up training while achieving reasonable convergence
    df_with_forecast, lstm_metrics = forecaster.train_and_forecast_all(df_raw, train_split=0.8, epochs=6)
    print("LSTM Forecaster results:")
    print(f"  - Train MAE: {lstm_metrics['train_mae']:.2f} kW")
    print(f"  - Test MAE:  {lstm_metrics['test_mae']:.2f} kW")
    print(f"  - Test RMSE: {lstm_metrics['test_rmse']:.2f} kW")
    
    # 3. Run HRES Simulation
    print("\n[Step 3/4] Running HRES simulation & battery dispatch logic...")
    simulator = HRESSimulator(battery_capacity=250.0, pv_area=500.0, teng_area=500.0)
    df_sim = simulator.run_simulation(df_with_forecast)
    kpis = simulator.calculate_kpis(df_sim)
    
    # Print KPIs
    print("\n==================================================================")
    print("                   SYSTEM PERFORMANCE KPIs                        ")
    print("==================================================================")
    print(f"Total Annual Load Demanded:       {kpis['total_load_kwh']:,.2f} kWh")
    print(f"Total Solar Energy Generated:      {kpis['total_solar_kwh']:,.2f} kWh")
    print(f"Total Wind Energy Generated:       {kpis['total_wind_kwh']:,.2f} kWh")
    print(f"Total TENG Energy Generated:       {kpis['total_teng_kwh']:,.2f} kWh")
    print(f"Total Renewable Energy Generated:  {kpis['total_gen_kwh']:,.2f} kWh")
    print(f"Total Grid Import:                 {kpis['grid_import_kwh']:,.2f} kWh")
    print(f"Total Unmet Load (Outages):        {kpis['unmet_load_kwh']:,.2f} kWh")
    print(f"Renewable Energy Fraction (RF):    {kpis['renewable_fraction_pct']:.2f} %")
    print(f"Annual CO2 Emissions Saved:        {kpis['co2_saved_kg']/1000.0:.2f} tonnes (vs diesel)")
    print(f"System Operational Efficiency:     {kpis['system_efficiency_pct']:.2f} %")
    print(f"Annual Cost Savings for School:    INR {kpis['cost_savings_inr']:,.2f}")
    print("==================================================================")
    
    # 4. Generate & Export Graphs
    print("\n[Step 4/4] Generating high-resolution graphs for presentation...")
    
    # Identify a typical heavy rainfall day during Northeast Monsoon (November)
    # We search for a day in November (month 11) with the highest rainfall accumulation
    november_days = df_sim[df_sim['month'] == 11]
    daily_rain = november_days.groupby('day')['rainfall_rate'].sum()
    peak_rain_day = daily_rain.idxmax()
    print(f"Selected representative heavy rain day: November {peak_rain_day}, 2026")
    
    # Extract data for that day
    day_df = df_sim[(df_sim['month'] == 11) & (df_sim['day'] == peak_rain_day)].copy()
    day_df.index = [f"{h:02d}:00" for h in day_df['hour']]
    
    # --- GRAPH 1: Solar vs TENG Output across 24 hours ---
    plt.figure(figsize=(10, 6))
    plt.plot(day_df.index, day_df['solar_gen_kw'], label='Solar PV Output (kW)', color='#FF8C00', linewidth=2.5)
    plt.plot(day_df.index, day_df['teng_gen_kw'], label='TENG Output (kW)', color='#00E5FF', linewidth=2.5)
    plt.plot(day_df.index, day_df['school_load'], label='School Load (kW)', color='#FF4081', linestyle='--', linewidth=2)
    plt.title(f'Solar vs TENG Power Output (Rainy Day - Nov {peak_rain_day})', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Power (kW)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend(frameon=True, facecolor='#1e1e1e', edgecolor='#333333')
    plt.tight_layout()
    save_plot('1_solar_vs_teng_24h.png')
    
    # --- GRAPH 2: Monthly energy generation comparison chart ---
    monthly_data = df_sim.groupby('month')[['solar_gen_kw', 'wind_gen_kw', 'teng_gen_kw', 'grid_import_kw', 'unmet_load_kw', 'school_load']].sum()
    months_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    plt.figure(figsize=(11, 7))
    # Stacked bar charts
    b1 = plt.bar(months_labels, monthly_data['solar_gen_kw'], label='Solar PV', color='#FFA726', width=0.6)
    b2 = plt.bar(months_labels, monthly_data['wind_gen_kw'], bottom=monthly_data['solar_gen_kw'], label='Wind Energy', color='#4CAF50', width=0.6)
    b3 = plt.bar(months_labels, monthly_data['teng_gen_kw'], bottom=monthly_data['solar_gen_kw'] + monthly_data['wind_gen_kw'], label='TENG Film', color='#26C6DA', width=0.6)
    b4 = plt.bar(months_labels, monthly_data['grid_import_kw'], bottom=monthly_data['solar_gen_kw'] + monthly_data['wind_gen_kw'] + monthly_data['teng_gen_kw'], label='Grid Import', color='#78909C', width=0.6)
    b5 = plt.bar(months_labels, monthly_data['unmet_load_kw'], bottom=monthly_data['solar_gen_kw'] + monthly_data['wind_gen_kw'] + monthly_data['teng_gen_kw'] + monthly_data['grid_import_kw'], label='Unmet Load (Grid Cut)', color='#E53935', width=0.6)
    
    # School load line comparison
    plt.plot(months_labels, monthly_data['school_load'], label='Total School Load', color='#FF4081', marker='o', linewidth=2.5, markersize=6)
    
    plt.title('Monthly Energy Generation & Supply Mix (Chennai School)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Energy (kWh)', fontsize=12)
    plt.grid(True, axis='y')
    plt.legend(frameon=True, facecolor='#1e1e1e', edgecolor='#333333', loc='upper right')
    plt.tight_layout()
    save_plot('2_monthly_energy_comparison.png')
    
    # --- GRAPH 3: Battery charge level throughout the day ---
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    # Plot SOC in percentage
    color_soc = '#66BB6A'
    ax1.plot(day_df.index, day_df['battery_soc'] * 100.0, label='Battery SOC (%)', color=color_soc, linewidth=3)
    ax1.set_xlabel('Hour of Day', fontsize=12)
    ax1.set_ylabel('State of Charge (SOC %)', color=color_soc, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color_soc)
    ax1.set_ylim(0, 105)
    ax1.grid(True)
    
    # Secondary axis to show battery charge/discharge power
    ax2 = ax1.twinx()
    color_pow = '#AB47BC'
    ax2.bar(day_df.index, day_df['battery_power_kw'], alpha=0.4, label='Battery Power Exchange (kW)', color=color_pow, width=0.5)
    ax2.set_ylabel('Power Exchanged (kW) [>0 Charging, <0 Discharging]', color=color_pow, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color_pow)
    ax2.axhline(0, color='white', linewidth=0.8, alpha=0.7)
    
    plt.title(f'Battery Storage Cycle Analysis (Nov {peak_rain_day})', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_plot('3_battery_charge_level.png')
    
    # --- GRAPH 4: Energy surplus and deficit bar chart ---
    # Net energy = Generated - School Load
    net_energy_hourly = day_df['total_gen_kw'] - day_df['school_load']
    
    plt.figure(figsize=(10, 6))
    surplus_bars = np.clip(net_energy_hourly, 0, None)
    deficit_bars = np.clip(net_energy_hourly, None, 0)
    
    plt.bar(day_df.index, surplus_bars, color='#26A69A', alpha=0.85, label='Net Energy Surplus (kW)')
    plt.bar(day_df.index, deficit_bars, color='#EF5350', alpha=0.85, label='Net Energy Deficit (kW)')
    
    plt.title(f'Hourly Net Energy Surplus & Deficit Profile (Nov {peak_rain_day})', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Net Power (kW)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.axhline(0, color='white', linewidth=1.0)
    plt.legend(frameon=True, facecolor='#1e1e1e', edgecolor='#333333')
    plt.tight_layout()
    save_plot('4_energy_surplus_deficit.png')
    
    # --- GRAPH 5: Rainfall vs TENG power output curve ---
    rainfall_rates = np.linspace(0, 80, 200)
    teng_powers = [calculate_teng_power(r, area=500.0) for r in rainfall_rates]
    
    plt.figure(figsize=(10, 6))
    plt.plot(rainfall_rates, teng_powers, color='#00E5FF', linewidth=3)
    plt.title('Rainfall Intensity vs TENG Electrical Power Output (500 m² Film)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Rainfall Intensity (mm/hour)', fontsize=12)
    plt.ylabel('TENG Generated Power (kW)', fontsize=12)
    plt.fill_between(rainfall_rates, teng_powers, color='#00E5FF', alpha=0.15)
    plt.grid(True)
    plt.tight_layout()
    save_plot('5_rainfall_vs_teng_curve.png')
    
    print("\nAll tasks completed successfully!")

if __name__ == "__main__":
    main()
