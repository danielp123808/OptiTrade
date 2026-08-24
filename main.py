import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================================
# 🛠️ GLOBAL SYSTEM INPUTS (EDIT DESIGN SPECS HERE)
# =====================================================================
DRY_FRAME_MASS_KG   = 1.6    # Structural bare weight without battery
LIFT_TO_DRAG_RATIO  = 1.2    # Multirotor aerodynamic cruising efficiency 
CRUISE_SPEED_MPS    = 12.0   # Forward flight velocity speed (meters per second)
PROPULSION_EFF      = 0.45   # Combined electrical matching efficiency (0.0 - 1.0)

# Target Cargo Envelope Specs
MIN_PAYLOAD_G       = 500.0  # Minimum target payload (grams)
MAX_PAYLOAD_G       = 3000.0 # Maximum target payload (grams)

# Optimization Search Space Parameters
MIN_BATTERY_WH      = 100.0  # Smallest battery option explored by GA
MAX_BATTERY_WH      = 600.0  # Largest battery option explored by GA
# =====================================================================


def evaluate_drone_performance(battery_wh, payload_g):
    """Calculates drone takeoff mass and realistic flight range values."""
    GRAVITY = 9.81
    
    # Calculate battery mass footprint: ~5g per Wh for high-density LiPo packs
    battery_mass_kg = (battery_wh * 5.0) / 1000.0
    payload_mass_kg = payload_g / 1000.0
    
    # Total takeoff weight updates dynamically
    total_mass_kg = DRY_FRAME_MASS_KG + battery_mass_kg + payload_mass_kg
    
    # Operational structural ceiling constraint (raised to allow heavy weights)
    if total_mass_kg > 10.0:
        return -1.0, float(payload_g)
        
    # Drag calculation based strictly on user L/D override parameter
    weight_n = total_mass_kg * GRAVITY
    aerodynamic_drag_n = weight_n / LIFT_TO_DRAG_RATIO
    
    # Realistic multirotor propulsion power draw matrix
    hover_power_w = total_mass_kg * 140.0  # Power required to stay aloft
    parasitic_power_w = aerodynamic_drag_n * CRUISE_SPEED_MPS # Power to cut wind
    
    total_electrical_power_w = (hover_power_w + parasitic_power_w) / PROPULSION_EFF
    
    # Calculate flight range limits
    endurance_hours = battery_wh / total_electrical_power_w
    range_km = (CRUISE_SPEED_MPS * (endurance_hours * 3600.0)) / 1000.0
    
    return float(range_km), float(payload_g)


def check_domination(p1_range, p1_pay, p2_range, p2_pay):
    """Returns True if individual 1 dominates individual 2."""
    # Domination condition: Better or equal in all objectives, strictly better in one
    if (p1_range >= p2_range and p1_pay >= p2_pay) and (p1_range > p2_range or p1_pay > p2_pay):
        return True
    return False


def run_nsga_optimization(pop_size=350, generations=40):
    """Evolves populations using true multi-objective non-domination criteria."""
    
    # Generate initial random combinations of battery capacity and payload weight
    pop_battery_wh = np.random.uniform(MIN_BATTERY_WH, MAX_BATTERY_WH, size=pop_size)
    pop_payload_g = np.random.uniform(MIN_PAYLOAD_G, MAX_PAYLOAD_G, size=pop_size)
    
    for gen in range(generations):
        scores = np.array([evaluate_drone_performance(b, p) for b, p in zip(pop_battery_wh, pop_payload_g)])
        
        # 🎯 THE FIX: Tournament selection based on Pareto Domination, not just Range alone
        idx1 = np.random.randint(0, pop_size, pop_size)
        idx2 = np.random.randint(0, pop_size, pop_size)
        
        parent_battery = np.zeros(pop_size)
        parent_payload = np.zeros(pop_size)
        
        for i in range(pop_size):
            i1, i2 = idx1[i], idx2[i]
            # If candidate 1 dominates candidate 2, choose candidate 1
            if check_domination(scores[i1, 0], scores[i1, 1], scores[i2, 0], scores[i2, 1]):
                parent_battery[i] = pop_battery_wh[i1]
                parent_payload[i] = pop_payload_g[i1]
            # If candidate 2 dominates candidate 1, choose candidate 2
            elif check_domination(scores[i2, 0], scores[i2, 1], scores[i1, 0], scores[i1, 1]):
                parent_battery[i] = pop_battery_wh[i2]
                parent_payload[i] = pop_payload_g[i2]
            # If they don't dominate each other, pick a random survivor to keep diversity alive
            else:
                chosen_idx = np.random.choice([i1, i2])
                parent_battery[i] = pop_battery_wh[chosen_idx]
                parent_payload[i] = pop_payload_g[chosen_idx]
        
        # Crossover genetic breeding
        cross_mask = np.random.rand(pop_size) < 0.5
        pop_battery_wh = np.where(cross_mask, parent_battery, np.roll(parent_battery, 1))
        pop_payload_g = np.where(cross_mask, parent_payload, np.roll(parent_payload, 1))
        
        # Random mutation adjustments
        mut_mask = np.random.rand(pop_size) < 0.2
        pop_battery_wh[mut_mask] += np.random.normal(0, 15, size=np.sum(mut_mask))
        pop_payload_g[mut_mask] += np.random.normal(0, 100, size=np.sum(mut_mask))
        
        # Boundary constraints clipping
        pop_battery_wh = np.clip(pop_battery_wh, MIN_BATTERY_WH, MAX_BATTERY_WH)
        pop_payload_g = np.clip(pop_payload_g, MIN_PAYLOAD_G, MAX_PAYLOAD_G)

    # Evaluate final optimized designs
    final_scores = np.array([evaluate_drone_performance(b, p) for b, p in zip(pop_battery_wh, pop_payload_g)])
    df = pd.DataFrame({'Battery_Wh': pop_battery_wh, 'Payload_g': pop_payload_g, 'Range_KM': final_scores[:, 0]})
    
    # Strip out failed or overweight configurations
    df = df[df['Range_KM'] > 0].reset_index(drop=True)
    
    # Isolate the final true Pareto Frontier line
    pareto_flags = []
    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if check_domination(other['Range_KM'], other['Payload_g'], row['Range_KM'], row['Payload_g']):
                dominated = True
                break
        pareto_flags.append(not dominated)
        
    df['Pareto_Optimal'] = pareto_flags
    return df


if __name__ == "__main__":
    print("Running Corrected Multi-Objective Optimization Engine...")
    df_results = run_nsga_optimization()
    
    suboptimal = df_results[df_results['Pareto_Optimal'] == False]
    optimized  = df_results[df_results['Pareto_Optimal'] == True].sort_values(by='Payload_g')
    
    # Plotting layout setup
    plt.figure(figsize=(10, 6))
    
    # Gray Cloud: Suboptimal configurations spread nicely across the screen
    plt.scatter(suboptimal['Payload_g'], suboptimal['Range_KM'], c='#B0BEC5', alpha=0.4, s=25, label='Explored Variations')
    
    # Crimson Line: Beautiful trade curve sweeping perfectly all the way from 500g to 3,000g
    plt.scatter(optimized['Payload_g'], optimized['Range_KM'], c='#D32F2F', edgecolors='black', s=80, zorder=3, label='Optimized Pareto Line')
    plt.plot(optimized['Payload_g'], optimized['Range_KM'], '#00838F', linestyle='--', linewidth=2, alpha=0.9, zorder=2)
    
    # Visual Framing Limits
    plt.xlim(MIN_PAYLOAD_G - 50, MAX_PAYLOAD_G + 50)
    plt.ylim(df_results['Range_KM'].min() - 0.5, df_results['Range_KM'].max() + 1.0)
    
    plt.title("NSGA Optimization: Cargo Payload Weight vs. Realistic Flight Range\n(Fixed Genetic Extinction Bug | Full Target Envelope: 500g - 3000g)", fontsize=11, fontweight='bold', pad=12)
    plt.xlabel("Available Cargo Payload Capacity (grams)", fontsize=10)
    plt.ylabel("Calculated Flight Cruise Range (kilometers)", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
