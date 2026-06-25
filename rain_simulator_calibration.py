import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# Import models
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import calculate_teng_power

# Configuration
R_load = 1e7  # 10 MOhm resistor
A_proto = 0.09  # 30x30 cm prototype area
eta_model_base = 0.025  # 2.5% base efficiency
calibration_rain_rate = 50.0  # mm/h
PROJECT_PLOT_DIR = r"C:\Users\monalisa\.gemini\antigravity\scratch\hres_simulation\plots"
ARTIFACT_DIR = r"C:\Users\monalisa\.gemini\antigravity\brain\465e6851-e228-48cb-a7fd-2d3bf709c787"

os.makedirs(PROJECT_PLOT_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def calculate_mech_power_density(rainfall_rate):
    """Calculates mechanical kinetic energy power density (W/m2) for a given rain rate."""
    if rainfall_rate <= 0:
        return 0.0, 0.0
    rho_water = 1000.0
    d = 2.23e-3 * (rainfall_rate ** 0.102)
    v = 9.58 * (1.0 - np.exp(-(d * 1000.0) / 1.77))
    v_drop = (np.pi * (d ** 3)) / 6.0
    m_drop = rho_water * v_drop
    e_k = 0.5 * m_drop * (v ** 2)
    vol_flux = (rainfall_rate / 1000.0) / 3600.0
    n_impact_density = vol_flux / v_drop
    p_mech_density = n_impact_density * e_k
    return p_mech_density, n_impact_density

def prototype_vs_model_plot(eta_proto, k_amp):
    """Generates the dual y-axis prototype vs model validation plot."""
    rainfall_rates = np.linspace(0.1, 80.0, 100)
    proto_outputs_uW_scaled = []
    model_outputs_W = []
    
    for r in rainfall_rates:
        p_mech_density, _ = calculate_mech_power_density(r)
        
        # Prototype power (raw W) = area * mech_density * efficiency
        p_proto_raw_W = A_proto * p_mech_density * eta_proto
        # Scaled up to 500 m2 (W)
        p_proto_scaled_W = p_proto_raw_W * (500.0 / A_proto)
        # Scaled prototype power in uW
        proto_outputs_uW_scaled.append(p_proto_scaled_W * 1e6)
        
        # Full model predicted power (kW -> W)
        p_model_W = calculate_teng_power(r, area=500.0) * 1000.0
        model_outputs_W.append(p_model_W)
        
    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Left Y axis: Scaled Prototype Output
    color_proto = '#00D9FF'
    ax1.plot(rainfall_rates, proto_outputs_uW_scaled, label='Prototype Output (Scaled to 500 m²)', color=color_proto, linewidth=2.5)
    ax1.set_xlabel('Rainfall Intensity (mm/hour)', fontsize=12)
    ax1.set_ylabel('Prototype Output (µW) [Left Axis]', color=color_proto, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color_proto)
    ax1.grid(True, alpha=0.3)
    
    # Right Y axis: Full Model Output
    ax2 = ax1.twinx()
    color_model = '#4ADE80'
    ax2.plot(rainfall_rates, model_outputs_W, label='Full HRES Model Output', color=color_model, linestyle='--', linewidth=2.5)
    ax2.set_ylabel('Full Model Output (W) [Right Axis]', color=color_model, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color_model)
    
    plt.title('HRES TENG Simulation Model Validation & Calibration', fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    
    p1 = os.path.join(PROJECT_PLOT_DIR, '6_prototype_vs_model_validation.png')
    p2 = os.path.join(ARTIFACT_DIR, '6_prototype_vs_model_validation.png')
    plt.savefig(p1, dpi=300, bbox_inches='tight')
    plt.savefig(p2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Validation Plot] Exported plot to {p1}")

def main():
    measured_voltage_mv = 2500.0
    if len(sys.argv) > 1:
        try:
            measured_voltage_mv = float(sys.argv[1])
        except ValueError:
            pass
    else:
        # Check if stdin is a tty (interactive)
        if sys.stdin.isatty():
            try:
                val = input("Enter real measured TENG prototype voltage (mV) [default 2500]: ").strip()
                if val:
                    measured_voltage_mv = float(val)
            except Exception:
                pass
        else:
            print("Non-interactive run. Using default voltage: 2500.0 mV")
            
    print(f"\n--- TENG Calibration System ---")
    print(f"Calibration Rain Rate:   {calibration_rain_rate:.1f} mm/h")
    print(f"Measured Voltage:        {measured_voltage_mv:.1f} mV")
    print(f"Load Resistance:         {R_load / 1e6:.1f} MOhm")
    
    # Calculate measured power in Watts: P = V^2 / R
    measured_voltage_volts = measured_voltage_mv / 1000.0
    p_meas_W = (measured_voltage_volts ** 2) / R_load
    
    # Calculate mechanical power density
    p_mech_density, _ = calculate_mech_power_density(calibration_rain_rate)
    p_mech_proto_W = A_proto * p_mech_density
    
    # Calculate prototype efficiency
    eta_proto = p_meas_W / p_mech_proto_W if p_mech_proto_W > 0 else 0.0
    
    # Calculate correction factor k_amp = eta_proto / eta_model_base
    k_amp = eta_proto / eta_model_base if eta_model_base > 0 else 1.0
    
    # Save k_amp to JSON config
    cal_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'teng_calibration.json')
    cal_data = {
        'measured_voltage_mv': measured_voltage_mv,
        'prototype_efficiency': eta_proto,
        'k_amp': k_amp
    }
    
    with open(cal_file, 'w') as f:
        json.dump(cal_data, f, indent=4)
        
    print(f"Prototype efficiency = {eta_proto * 100.0:.4f}%, model correction factor = {k_amp:.4f}")
    print(f"Calibration details saved to {cal_file}")
    
    # Generate and save the validation plot
    prototype_vs_model_plot(eta_proto, k_amp)

if __name__ == "__main__":
    main()
