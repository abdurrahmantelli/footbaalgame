import numpy as np
import pandas as pd
import os

# Physical parameters of a standard football
GRAVITY = 9.81         # m/s^2
AIR_DENSITY = 1.2      # kg/m^3
BALL_MASS = 0.43       # kg
BALL_RADIUS = 0.11     # meters
BALL_AREA = np.pi * (BALL_RADIUS ** 2)
DRAG_COEFF = 0.25      # Cd
MAGNUS_COEFF = 0.15    # Cl parameter

GOAL_WIDTH = 7.32      # meters (-3.66 to 3.66)
GOAL_HEIGHT = 2.44     # meters (0 to 2.44)
PENALTY_DISTANCE = 11.0 # meters (from z=0 to z=11)

def simulate_shot(vx0, vy0, vz0, spin, dt=0.005):
    """
    Simulates a 3D shot trajectory from the penalty spot (0, 0, 0) to the goal line (z = 11).
    spin: angular velocity in rad/s around the y-axis (causes horizontal curve - Magnus effect)
    """
    x, y, z = 0.0, BALL_RADIUS, 0.0
    vx, vy, vz = vx0, vy0, vz0
    
    # We want to record the "early" state at 20ms to simulate what the AI reads
    # This represents the early flick gesture features.
    early_vx, early_vy = 0.0, 0.0
    recorded_early = False
    
    t = 0.0
    while z < PENALTY_DISTANCE:
        if t >= 0.020 and not recorded_early:
            early_vx = vx
            early_vy = vy
            recorded_early = True
            
        v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
        if v_mag == 0:
            break
            
        # Drag Force
        f_drag_x = -0.5 * AIR_DENSITY * DRAG_COEFF * BALL_AREA * v_mag * vx
        f_drag_y = -0.5 * AIR_DENSITY * DRAG_COEFF * BALL_AREA * v_mag * vy
        f_drag_z = -0.5 * AIR_DENSITY * DRAG_COEFF * BALL_AREA * v_mag * vz
        
        # Magnus Force (simplified horizontal Magnus effect: cross product of spin (vertical) x velocity)
        # F = Cl * rho * A * r * (omega x v)
        # spin is around y-axis (vertical spin)
        f_magnus_x = MAGNUS_COEFF * AIR_DENSITY * BALL_AREA * BALL_RADIUS * spin * vz
        f_magnus_z = -MAGNUS_COEFF * AIR_DENSITY * BALL_AREA * BALL_RADIUS * spin * vx
        
        # Total Accelerations
        ax = (f_drag_x + f_magnus_x) / BALL_MASS
        ay = -GRAVITY + (f_drag_y) / BALL_MASS
        az = (f_drag_z + f_magnus_z) / BALL_MASS
        
        # Euler integration
        vx += ax * dt
        vy += ay * dt
        vz += az * dt
        
        x += vx * dt
        y += vy * dt
        z += vz * dt
        
        t += dt
        
        # Ground collision
        if y < BALL_RADIUS and vy < 0:
            y = BALL_RADIUS
            vy = -vy * 0.5 # bounce
            
        # Stop if velocity gets too low or ball goes backwards
        if vz <= 0:
            break

    if not recorded_early:
        early_vx = vx
        early_vy = vy
        
    return x, y, early_vx, early_vy

def get_goal_zone(x, y):
    """
    Maps landing coordinates (x, y) at the goal plane (z=11) to one of the 9 zones (0 to 8).
    If it misses the goal, returns -1.
    """
    if x < -GOAL_WIDTH / 2 or x > GOAL_WIDTH / 2:
        return -1
    if y < 0 or y > GOAL_HEIGHT:
        return -1
        
    # Horizontal split
    if x < -GOAL_WIDTH / 6:      # Left third: x < -1.22
        col = 0
    elif x > GOAL_WIDTH / 6:     # Right third: x > 1.22
        col = 2
    else:                        # Center third: -1.22 <= x <= 1.22
        col = 1
        
    # Vertical split
    if y > (2 * GOAL_HEIGHT / 3):   # Top third: y > 1.63
        row = 0
    elif y < (GOAL_HEIGHT / 3):     # Bottom third: y < 0.81
        row = 2
    else:                           # Middle third: 0.81 <= y <= 1.63
        row = 1
        
    # Index = row * 3 + col
    return row * 3 + col

def generate_dataset(num_samples=20000):
    print("Generating synthetic penalty kick dataset...")
    data = []
    
    # Random distributions for shot properties
    # Let's generate a diverse set of initial velocities
    np.random.seed(42)
    
    count = 0
    attempts = 0
    
    while count < num_samples and attempts < num_samples * 10:
        attempts += 1
        # initial velocities
        vx0 = np.random.uniform(-10.0, 10.0)
        vy0 = np.random.uniform(2.0, 12.0)
        vz0 = np.random.uniform(22.0, 32.0) # speed towards the goal
        
        # spin (Magnus effect) in rad/s (e.g. -40 to 40 rad/s)
        spin = np.random.uniform(-45.0, 45.0)
        
        # Simulate
        x_final, y_final, early_vx, early_vy = simulate_shot(vx0, vy0, vz0, spin)
        zone = get_goal_zone(x_final, y_final)
        
        if zone != -1:
            # We save the features: 
            # 1. Early velocity X (first 20ms)
            # 2. Early velocity Y (first 20ms)
            # 3. Spin/Curve amount
            # 4. Total flick speed (initial speed magnitude)
            flick_speed = np.sqrt(vx0**2 + vy0**2 + vz0**2)
            data.append({
                'early_vx': early_vx,
                'early_vy': early_vy,
                'spin': spin,
                'flick_speed': flick_speed,
                'target_zone': zone,
                'x_final': x_final,
                'y_final': y_final
            })
            count += 1

    df = pd.DataFrame(data)
    os.makedirs('core', exist_ok=True)
    df.to_csv('core/penalty_dataset.csv', index=False)
    print(f"Generated {len(df)} valid on-target shots stored in core/penalty_dataset.csv")
    print(df['target_zone'].value_counts().sort_index())

if __name__ == "__main__":
    generate_dataset(15000)
