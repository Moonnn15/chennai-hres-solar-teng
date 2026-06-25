import sys
import numpy as np

def simulate_prototype(rainfall_rate=50.0, duration_seconds=60.0):
    """
    Simulates a 30x30 cm TENG prototype under constant rainfall.
    
    Parameters:
    - rainfall_rate: intensity in mm/h (default: 50.0)
    - duration_seconds: duration of test in seconds (default: 60.0)
    """
    area_proto = 0.09  # 30x30 cm = 0.09 m2
    eta_teng = 0.025   # 2.5% efficiency
    rho_water = 1000.0 # kg/m3
    
    if rainfall_rate <= 0:
        return {
            "pulses_per_second": 0.0,
            "power_microwatts": 0.0,
            "energy_millijoules": 0.0
        }
        
    # 1. Median droplet diameter (meters)
    d = 2.23e-3 * (rainfall_rate ** 0.102)
    d_mm = d * 1000.0
    
    # 2. Terminal velocity (m/s)
    v = 9.58 * (1.0 - np.exp(-d_mm / 1.77))
    
    # 3. Mass of a single droplet (kg)
    v_drop = (np.pi * (d ** 3)) / 6.0
    m_drop = rho_water * v_drop
    
    # 4. Kinetic energy of a single droplet (Joules)
    e_k = 0.5 * m_drop * (v ** 2)
    
    # 5. Raindrop impact rate (drops per m2 per second)
    vol_flux = (rainfall_rate / 1000.0) / 3600.0
    n_impact_density = vol_flux / v_drop
    
    # Pulses per second on the prototype
    pulses_per_second = n_impact_density * area_proto
    
    # 6. Raw mechanical power density (W/m2)
    p_mech_density = n_impact_density * e_k
    
    # 7. Raw electrical power (W) for the prototype area
    p_elec_watts = area_proto * p_mech_density * eta_teng
    
    # Convert to microwatts
    p_elec_uw = p_elec_watts * 1e6
    
    # 8. Cumulative energy in millijoules (mJ) over duration
    energy_mj = p_elec_watts * duration_seconds * 1000.0
    
    return {
        "pulses_per_second": pulses_per_second,
        "power_microwatts": p_elec_uw,
        "energy_millijoules": energy_mj
    }

if __name__ == "__main__":
    # Get rainfall rate from command line if available, otherwise default to 50 mm/h
    rain_rate = 50.0
    if len(sys.argv) > 1:
        try:
            rain_rate = float(sys.argv[1])
        except ValueError:
            print("Invalid rainfall rate argument. Using default 50.0 mm/h.")
            
    results = simulate_prototype(rain_rate)
    
    print("==========================================================")
    print("      30x30 cm PROTOTYPE TENG SIMULATION RESULTS          ")
    print("==========================================================")
    print(f"Rainfall Intensity:        {rain_rate:.2f} mm/h")
    print(f"Prototype Area:            0.09 m² (30x30 cm)")
    print(f"Simulation Duration:       60.0 seconds")
    print("----------------------------------------------------------")
    print(f"Raindrop Impact Rate:      {results['pulses_per_second']:.2f} pulses/second")
    print(f"Predicted Power Output:    {results['power_microwatts']:.2f} µW")
    print(f"Cumulative Energy:         {results['energy_millijoules']:.2f} mJ")
    print("==========================================================")
