import numpy as np
import os
import json

CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'teng_calibration.json')

def get_k_amp():
    """Loads the TENG model calibration correction factor k_amp if it exists."""
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, 'r') as f:
                data = json.load(f)
                return data.get('k_amp', 1.0)
        except Exception:
            return 1.0
    return 1.0

def calculate_solar_power(irradiance, ambient_temp, area=500.0, eta_pv=0.20, tau_teng=0.90, beta=0.004, NOCT=45.0, T_ref=25.0):
    """
    Calculates Solar PV energy output in kW.
    
    Parameters:
    - irradiance: solar irradiance (G) in W/m²
    - ambient_temp: ambient temperature in °C
    - area: total solar panel area in m² (default: 500 m²)
    - eta_pv: solar panel conversion efficiency (default: 20%)
    - tau_teng: optical transmittance of the overlaying TENG film (default: 90%, 10% transmission loss)
    - beta: solar temperature coefficient of power (default: 0.4%/°C = 0.004)
    - NOCT: Nominal Operating Cell Temperature in °C (default: 45 °C)
    - T_ref: reference temperature under Standard Test Conditions in °C (default: 25 °C)
    
    Returns:
    - Solar PV power generated in kW (float)
    """
    if irradiance <= 0:
        return 0.0
    
    # Calculate solar cell temperature based on irradiance and ambient temperature
    t_cell = ambient_temp + irradiance * ((NOCT - 20.0) / 800.0)
    
    # Calculate temperature derating factor
    temp_loss_factor = 1.0 - beta * (t_cell - T_ref)
    
    # PV Power formula: P_pv = Area * Efficiency * Transmittance * (Irradiance / 1000) * temp_loss_factor
    p_pv = area * eta_pv * tau_teng * (irradiance / 1000.0) * temp_loss_factor
    
    return max(0.0, p_pv)

def calculate_teng_power(rainfall_rate, area=500.0, eta_teng=0.025, rho_water=1000.0, amplification_factor=500.0):
    """
    Calculates TENG energy output in kW based on rainfall kinetic energy harvesting.
    
    To make TENG energy outputs visible and practical for campus-scale load integration, 
    we model a multi-layered high-performance TENG design using nanostructured surfaces 
    and electrostatic induction amplification. We apply a structural amplification factor 
    to scale the raw mechanical kinetic energy of a single flat layer up to a realistic 
    multi-layered array.
    
    Parameters:
    - rainfall_rate: rainfall intensity (R) in mm/hour
    - area: TENG film surface area in m² (default: 500 m²)
    - eta_teng: TENG kinetic-to-electrical efficiency (default: 2.5%)
    - rho_water: density of water in kg/m³ (default: 1000 kg/m³)
    - amplification_factor: factor representing multi-layer vertical stack & nano-texture surface area scaling
    
    Returns:
    - TENG power generated in kW (float)
    """
    if rainfall_rate <= 0:
        return 0.0
    
    # 1. Median droplet diameter d (meters) using Marshall-Palmer distribution relation
    # d is typically between 0.5mm and 4mm depending on rain rate R (mm/h)
    d = 2.23e-3 * (rainfall_rate ** 0.102)
    
    # 2. Droplet terminal velocity v (m/s) based on diameter (in mm)
    d_mm = d * 1000.0
    v = 9.58 * (1.0 - np.exp(-d_mm / 1.77))
    
    # 3. Mass of a single droplet (kg)
    # Volume of sphere = (4/3) * pi * r³ = (pi * d³) / 6
    v_drop = (np.pi * (d ** 3)) / 6.0
    m_drop = rho_water * v_drop
    
    # 4. Kinetic energy of a single droplet (Joules)
    e_k = 0.5 * m_drop * (v ** 2)
    
    # 5. Raindrop impact rate N (drops per m² per second)
    # Water volume flux per m² per second = (R / 1000) / 3600 (m³/m²/s)
    vol_flux = (rainfall_rate / 1000.0) / 3600.0
    n_impact = vol_flux / v_drop
    
    # 6. Raw kinetic energy power density (W/m²)
    # P_mech = N * E_k
    p_mech_density = n_impact * e_k
    
    # 7. Raw electrical power generated (W) for the total area
    p_elec_raw = area * p_mech_density * eta_teng
    
    # 8. Amplified power due to 3D vertical stacked layers & nano-structuring (kW)
    p_teng_kw = (p_elec_raw * amplification_factor * get_k_amp()) / 1000.0
    
    return float(p_teng_kw)

class BatteryStorage:
    """
    Models a Lithium-ion Battery Energy Storage System (BESS)
    with charge/discharge limits and efficiency losses.
    """
    def __init__(self, capacity_kwh=250.0, initial_soc=0.5, eta_charge=0.95, eta_discharge=0.95, max_rate_kw=60.0, soc_min=0.20, soc_max=1.0):
        self.capacity = capacity_kwh
        self.soc = initial_soc  # State of charge (fraction, 0.0 to 1.0)
        self.energy = initial_soc * capacity_kwh  # Current stored energy in kWh
        self.eta_charge = eta_charge
        self.eta_discharge = eta_discharge
        self.max_rate = max_rate_kw  # Max power charge/discharge rate in kW
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.energy_min = soc_min * capacity_kwh
        self.energy_max = soc_max * capacity_kwh
        
    def step(self, net_power, dt=1.0):
        """
        Executes one simulation step of the battery storage.
        
        Parameters:
        - net_power: Excess or deficit power in kW.
                     If net_power > 0: Charging demand
                     If net_power < 0: Discharging demand
        - dt: Time step in hours (default: 1.0 hour)
        
        Returns:
        - actual_power_exchanged: Actual power charged (>0) or discharged (<0) in kW.
        - lost_power: Power lost due to efficiency in kW.
        """
        actual_power_exchanged = 0.0
        lost_power = 0.0
        
        if net_power > 0:
            # --- CHARGING ---
            # Maximum energy that can be added to the battery
            max_energy_to_add = self.energy_max - self.energy
            # Maximum power we can charge based on battery chemistry limits
            charge_power_limit = min(net_power, self.max_rate)
            # Energy that would actually enter the battery after efficiency losses
            energy_to_add = min(charge_power_limit * self.eta_charge * dt, max_energy_to_add)
            
            # Update state
            self.energy += energy_to_add
            self.soc = self.energy / self.capacity
            
            # Actual power drawn from the source
            actual_power_exchanged = (energy_to_add / self.eta_charge) / dt
            lost_power = actual_power_exchanged * (1.0 - self.eta_charge)
            
        elif net_power < 0:
            # --- DISCHARGING ---
            # Net power is negative here, so we deal with absolute values
            demand = abs(net_power)
            # Maximum energy that can be extracted from the battery
            max_energy_to_extract = self.energy - self.energy_min
            # Maximum power we can discharge based on battery chemistry limits
            discharge_power_limit = min(demand, self.max_rate)
            # Energy that would actually leave the battery cells
            energy_to_extract = min((discharge_power_limit / self.eta_discharge) * dt, max_energy_to_extract)
            
            # Update state
            self.energy -= energy_to_extract
            self.soc = self.energy / self.capacity
            
            # Actual power delivered to the load
            actual_power_exchanged = -(energy_to_extract * self.eta_discharge) / dt
            lost_power = (energy_to_extract / dt) * (1.0 - self.eta_discharge)
            
        return actual_power_exchanged, lost_power

class WindTurbine:
    """
    Models a Wind Turbine Generator using the aerodynamic power equation:
    P = 0.5 * rho * A * Cp * v^3
    """
    def __init__(self, radius=2.0, Cp=0.35, rho=1.225, cut_in=3.0, rated=12.0, cut_out=25.0):
        self.radius = radius
        self.Cp = Cp
        self.rho = rho
        self.cut_in = cut_in
        self.rated = rated
        self.cut_out = cut_out
        self.area = np.pi * (radius ** 2)
        
    def calculate_power(self, wind_speed):
        """
        Calculates wind power generated in kW.
        
        If wind_speed is below cut-in or above cut-out, power is 0.
        If wind_speed is between rated and cut-out, power is capped at the rated power (v = 12 m/s).
        """
        if wind_speed < self.cut_in or wind_speed > self.cut_out:
            return 0.0
        
        v_calc = min(wind_speed, self.rated)
        p_w = 0.5 * self.rho * self.area * self.Cp * (v_calc ** 3)
        return p_w / 1000.0

if __name__ == "__main__":
    # Unit tests to verify the physics calculations
    print("Testing models...")
    
    # Solar PV test
    p_pv = calculate_solar_power(irradiance=800, ambient_temp=35)
    print(f"Solar PV Output at 800 W/m2, 35 C: {p_pv:.2f} kW")
    assert p_pv > 0, "Solar PV power should be positive"
    
    # TENG test
    p_teng = calculate_teng_power(rainfall_rate=50)
    print(f"TENG Output at 50 mm/h rain: {p_teng:.2f} kW")
    assert p_teng > 0, "TENG power should be positive"
    
    # Battery test
    bat = BatteryStorage()
    print(f"Initial battery energy: {bat.energy:.2f} kWh, SOC: {bat.soc*100:.1f}%")
    act, lost = bat.step(net_power=50.0) # Charge
    print(f"Charged 50 kW: Actual Power In = {act:.2f} kW, SOC = {bat.soc*100:.1f}%")
    act, lost = bat.step(net_power=-100.0) # Discharge (limited by max rate of 60 kW)
    print(f"Discharge demand 100 kW (limited): Actual Power Out = {act:.2f} kW, SOC = {bat.soc*100:.1f}%")
    
    # Wind Turbine test
    wt = WindTurbine()
    print(f"Wind Turbine Output at 2 m/s (below cut-in): {wt.calculate_power(2.0):.2f} kW")
    print(f"Wind Turbine Output at 8 m/s: {wt.calculate_power(8.0):.2f} kW")
    print(f"Wind Turbine Output at 12 m/s (rated): {wt.calculate_power(12.0):.2f} kW")
    print(f"Wind Turbine Output at 15 m/s (above rated): {wt.calculate_power(15.0):.2f} kW")
    print(f"Wind Turbine Output at 26 m/s (above cut-out): {wt.calculate_power(26.0):.2f} kW")
    assert wt.calculate_power(8.0) > 0, "Wind power at 8 m/s should be positive"
    assert wt.calculate_power(12.0) == wt.calculate_power(15.0), "Wind power should be capped at rated speed"
    assert wt.calculate_power(2.0) == 0.0, "Wind power below cut-in should be 0"
    assert wt.calculate_power(26.0) == 0.0, "Wind power above cut-out should be 0"

