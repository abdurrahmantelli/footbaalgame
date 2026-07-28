import pygame
import numpy as np
import tensorflow as tf
import json
import time
import os

# Physical parameters (aligned with generate_data.py)
GRAVITY = 9.81
AIR_DENSITY = 1.2
BALL_MASS = 0.43
BALL_RADIUS = 0.11
BALL_AREA = np.pi * (BALL_RADIUS ** 2)
DRAG_COEFF = 0.25
MAGNUS_COEFF = 0.15

GOAL_WIDTH = 7.32
GOAL_HEIGHT = 2.44
PENALTY_DISTANCE = 11.0

# Pygame Window Constants
WIDTH, HEIGHT = 900, 650
FPS = 60

# Colors (Sleek dark theme with glowing neon details)
BG_COLOR = (15, 23, 42)       # Slate 900
PITCH_COLOR = (30, 41, 59)    # Slate 800
WHITE = (255, 255, 255)
NEON_GREEN = (34, 197, 94)    # Emerald 500 (glowing saves)
NEON_RED = (239, 68, 68)      # Red 500 (goals)
NEON_BLUE = (59, 130, 246)    # Blue 500 (telemetry / goalkeeper)
TEXT_COLOR = (241, 245, 249)  # Slate 100
GRID_COLOR = (51, 65, 85)     # Slate 700

# Mapping from 3D coords to 2D Pygame screen coords (Goal view: looking from penalty spot to goal)
# Real goal width: -3.66m to +3.66m, height: 0m to 2.44m
# We will draw the goal in the center of the screen
GOAL_SCALE = 85 # Pixels per meter
GOAL_CENTER_X = WIDTH // 2
GOAL_BOTTOM_Y = HEIGHT - 180

def to_screen_coords(x, y):
    """Converts 3D coordinates (relative to goal center) to 2D screen pixels."""
    screen_x = int(GOAL_CENTER_X + x * GOAL_SCALE)
    screen_y = int(GOAL_BOTTOM_Y - y * GOAL_SCALE)
    return screen_x, screen_y

class TFLiteGoalkeeper:
    def __init__(self, model_path, scaler_path):
        # Load Scaler parameters
        with open(scaler_path, 'r') as f:
            params = json.load(f)
        self.means = np.array(params['mean'])
        self.scales = np.array(params['scale'])
        
        # Load TFLite Model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
    def predict(self, early_vx, early_vy, spin, flick_speed):
        # Scale inputs
        raw_features = np.array([early_vx, early_vy, spin, flick_speed])
        scaled_features = (raw_features - self.means) / self.scales
        scaled_features = np.expand_dims(scaled_features.astype(np.float32), axis=0)
        
        # Set input tensor
        start_time = time.perf_counter()
        self.interpreter.set_tensor(self.input_details[0]['index'], scaled_features)
        
        # Invoke TFLite (runs the quantized model)
        self.interpreter.invoke()
        
        # Get output probabilities
        probs = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000.0
        predicted_zone = np.argmax(probs)
        
        return predicted_zone, probs, latency_ms

class Ball:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.x, self.y, self.z = 0.0, BALL_RADIUS, 0.0
        self.vx, self.vy, self.vz = 0.0, 0.0, 0.0
        self.spin = 0.0
        self.active = False
        self.trajectory = []
        
    def launch(self, vx0, vy0, vz0, spin):
        self.vx, self.vy, self.vz = vx0, vy0, vz0
        self.spin = spin
        self.active = True
        self.trajectory = []
        
    def update(self, dt=0.005):
        if not self.active:
            return
            
        self.trajectory.append((self.x, self.y, self.z))
        
        v_mag = np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)
        if v_mag == 0:
            self.active = False
            return
            
        # Drag Force
        f_drag_x = -0.5 * AIR_DENSITY * DRAG_COEFF * BALL_AREA * v_mag * self.vx
        f_drag_y = -0.5 * AIR_DENSITY * DRAG_COEFF * BALL_AREA * v_mag * self.vy
        f_drag_z = -0.5 * AIR_DENSITY * DRAG_COEFF * BALL_AREA * v_mag * self.vz
        
        # Magnus Force (simplified horizontal Magnus effect: spin is around y-axis)
        f_magnus_x = MAGNUS_COEFF * AIR_DENSITY * BALL_AREA * BALL_RADIUS * self.spin * self.vz
        f_magnus_z = -MAGNUS_COEFF * AIR_DENSITY * BALL_AREA * BALL_RADIUS * self.spin * self.vx
        
        # Accelerations
        ax = (f_drag_x + f_magnus_x) / BALL_MASS
        ay = -GRAVITY + (f_drag_y) / BALL_MASS
        az = (f_drag_z + f_magnus_z) / BALL_MASS
        
        # Integrate
        self.vx += ax * dt
        self.vy += ay * dt
        self.vz += az * dt
        
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt
        
        # Ground bounce
        if self.y < BALL_RADIUS and self.vy < 0:
            self.y = BALL_RADIUS
            self.vy = -self.vy * 0.4
            
        # Check goal plane collision
        if self.z >= PENALTY_DISTANCE:
            self.active = False

def get_zone_rect(zone_idx):
    """Returns Pygame Rect of a specific zone."""
    col = zone_idx % 3
    row = zone_idx // 3
    
    # Each zone is 1/3 goal width and 1/3 goal height
    w = (GOAL_WIDTH / 3) * GOAL_SCALE
    h = (GOAL_HEIGHT / 3) * GOAL_SCALE
    
    left_x = GOAL_CENTER_X - (GOAL_WIDTH / 2) * GOAL_SCALE + col * w
    top_y = GOAL_BOTTOM_Y - GOAL_HEIGHT * GOAL_SCALE + row * h
    
    return pygame.Rect(left_x, top_y, w, h)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pocket-Keeper AI: Predictive Goalkeeper Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Outfit", 18)
    font_bold = pygame.font.SysFont("Outfit", 20, bold=True)
    font_title = pygame.font.SysFont("Outfit", 26, bold=True)
    
    # Initialize Goalkeeper TFLite model
    model_path = 'core/goalkeeper_ai_9zone.tflite'
    scaler_path = 'core/scaler_params.json'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Model or scaler not found. Make sure you run train_goalkeeper.py first.")
        return
        
    ai_keeper = TFLiteGoalkeeper(model_path, scaler_path)
    ball = Ball()
    
    # Goalkeeper movement parameters
    keeper_x, keeper_y = 0.0, BALL_RADIUS # Centered, on the ground initially
    target_keeper_x, target_keeper_y = 0.0, BALL_RADIUS
    keeper_speed = 7.0 # Horizontal speed (will be used for return to center, not during jump)
    
    # Goalkeeper jump parameters
    keeper_is_jumping = False
    jump_start_time = 0.0
    jump_duration = 0.5 # seconds
    jump_height = 2.0 # Max jump height in meters
    keeper_jump_vx = 0.0 # Initial horizontal velocity for jump
    keeper_jump_vy = 0.0 # Initial vertical velocity for jump
    initial_keeper_x = 0.0 # Keeper's x position when jump starts
    initial_keeper_y = BALL_RADIUS # Keeper's y position when jump starts
    
    # Touch event variables
    touch_points = []
    touch_start_time = 0
    early_features = None
    
    # Stats
    goals = 0
    saves = 0
    total_shots = 0
    ai_prediction = None
    ai_probs = np.zeros(9)
    inference_time = 0.0
    shot_outcome = ""
    outcome_color = TEXT_COLOR
    
    # Simulation state flags
    flick_active = False
    
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        
        # Event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if not ball.active:
                    ball.reset()
                    touch_points = [event.pos]
                    touch_start_time = time.perf_counter()
                    flick_active = True
                    shot_outcome = ""
                    ai_prediction = None
                    early_features = None
                    target_keeper_x, target_keeper_y = 0.0, BALL_RADIUS # Reset keeper to ground
                    keeper_is_jumping = False # Reset jump state
                    
            elif event.type == pygame.MOUSEMOTION:
                if flick_active:
                    touch_points.append(event.pos)
                    
                    # Capture early gesture characteristics at ~25ms
                    elapsed = (time.perf_counter() - touch_start_time) * 1000.0
                    if 15 <= elapsed <= 30 and early_features is None and len(touch_points) >= 3:
                        # Estimate early physics vectors
                        p1 = touch_points[0]
                        p2 = touch_points[-1]
                        
                        # vx maps to x movement (positive right)
                        early_vx = (p2[0] - p1[0]) * 0.07
                        # vy maps to y movement (positive up)
                        early_vy = (p1[1] - p2[1]) * 0.05
                        
                        # Simple curve estimation from curvature of early touch points
                        # If the swipe curls, we estimate spin
                        mid_idx = len(touch_points) // 2
                        pm = touch_points[mid_idx]
                        curve_val = (p2[0] - p1[0]) * (pm[1] - p1[1]) - (p2[1] - p1[1]) * (pm[0] - p1[0])
                        spin = np.clip(curve_val * 0.1, -45.0, 45.0)
                        
                        flick_speed = np.sqrt(early_vx**2 + early_vy**2 + 25.0**2)
                        
                        early_features = (early_vx, early_vy, spin, flick_speed)
                        
                        # Run INT8 TFLite Prediction
                        ai_prediction, ai_probs, inference_time = ai_keeper.predict(
                            early_vx, early_vy, spin, flick_speed
                        )
                        
                        # Move goalkeeper to predict target zone
                        # Map target zone (0-8) to real-world target coordinates
                        col = ai_prediction % 3
                        row = ai_prediction // 3
                        target_keeper_x = (col - 1) * (GOAL_WIDTH / 3.0)
                        target_keeper_y = GOAL_HEIGHT - (row + 0.5) * (GOAL_HEIGHT / 3.0) # Target y is the desired save height
                        
                        # Trigger jump when AI predicts
                        keeper_is_jumping = True
                        jump_start_time = time.perf_counter()
                        initial_keeper_x = keeper_x # Store current x
                        initial_keeper_y = keeper_y # Store current y
                        
                        # Calculate jump velocities for a parabolic trajectory to target_keeper_x, target_keeper_y
                        # Assuming a fixed jump duration, calculate initial vertical velocity needed to reach target_keeper_y
                        time_to_peak = jump_duration / 2.0
                        # Calculate vy to reach target_keeper_y at half duration, accounting for gravity
                        keeper_jump_vy = (target_keeper_y - initial_keeper_y + 0.5 * GRAVITY * time_to_peak**2) / time_to_peak
                        
                        # Calculate vx to reach target_keeper_x in full duration
                        keeper_jump_vx = (target_keeper_x - initial_keeper_x) / jump_duration
                        
            elif event.type == pygame.MOUSEBUTTONUP:
                if flick_active and len(touch_points) >= 2:
                    flick_active = False
                    p1 = touch_points[0]
                    p_end = touch_points[-1]
                    
                    # Compute final release physical properties
                    vx0 = (p_end[0] - p1[0]) * 0.07
                    vy0 = (p1[1] - p_end[1]) * 0.05
                    vz0 = np.random.uniform(23.0, 30.0) # speed based on swipe distance
                    
                    # Estimate Magnus spin based on curvature
                    pm = touch_points[len(touch_points)//2]
                    curve_val = (p_end[0] - p1[0]) * (pm[1] - p1[1]) - (p_end[1] - p1[1]) * (pm[0] - p1[0])
                    spin = np.clip(curve_val * 0.12, -45.0, 45.0)
                    
                    # If we released before early prediction triggered, predict now
                    if early_features is None:
                        early_vx, early_vy = vx0, vy0
                        flick_speed = np.sqrt(early_vx**2 + early_vy**2 + vz0**2)
                        ai_prediction, ai_probs, inference_time = ai_keeper.predict(
                            early_vx, early_vy, spin, flick_speed
                        )
                        col = ai_prediction % 3
                        row = ai_prediction // 3
                        target_keeper_x = (col - 1) * (GOAL_WIDTH / 3.0)
                        target_keeper_y = GOAL_HEIGHT - (row + 0.5) * (GOAL_HEIGHT / 3.0) # Target y is the desired save height
                        
                        # Trigger jump when AI predicts
                        keeper_is_jumping = True
                        jump_start_time = time.perf_counter()
                        initial_keeper_x = keeper_x # Store current x
                        initial_keeper_y = keeper_y # Store current y
                        
                        # Calculate jump velocities for a parabolic trajectory to target_keeper_x, target_keeper_y
                        time_to_peak = jump_duration / 2.0
                        keeper_jump_vy = (target_keeper_y - initial_keeper_y + 0.5 * GRAVITY * time_to_peak**2) / time_to_peak
                        keeper_jump_vx = (target_keeper_x - initial_keeper_x) / jump_duration
                    
                    # Launch physical simulation
                    ball.launch(vx0, vy0, vz0, spin)
                    total_shots += 1
        
        # 1. Update Physics
        if ball.active:
            # Run physics multiple times per frame for high integration precision
            for _ in range(5):
                ball.update(dt / 5.0)
                
            # Goalkeeper jumping logic
            if keeper_is_jumping:
                time_in_jump = time.perf_counter() - jump_start_time
                if time_in_jump < jump_duration:
                    # Update keeper position based on jump physics
                    keeper_x = initial_keeper_x + keeper_jump_vx * time_in_jump
                    keeper_y = initial_keeper_y + keeper_jump_vy * time_in_jump - 0.5 * GRAVITY * time_in_jump**2
                    
                    # Ensure keeper doesn't go below ground
                    if keeper_y < BALL_RADIUS:
                        keeper_y = BALL_RADIUS
                else:
                    keeper_is_jumping = False
                    keeper_x = target_keeper_x # Land at target x
                    keeper_y = BALL_RADIUS # Land on the ground
                    
            # If ball reached goal plane, evaluate outcome
            if not ball.active:
                # Target coordinate at goal plane
                final_x = ball.x
                final_y = ball.y
                
                # Check collision with goalkeeper (radius based overlap)
                keeper_radius = 1.0 # Goalkeeper arm span
                dist_to_keeper = np.sqrt((final_x - keeper_x)**2 + (final_y - keeper_y)**2)
                
                # Check if it was in the goal
                if -GOAL_WIDTH/2 <= final_x <= GOAL_WIDTH/2 and 0 <= final_y <= GOAL_HEIGHT:
                    if dist_to_keeper <= keeper_radius:
                        saves += 1
                        shot_outcome = "KURTARIŞ! (SAVE)"
                        outcome_color = NEON_GREEN
                    else:
                        goals += 1
                        shot_outcome = "GOOOL!"
                        outcome_color = NEON_RED
                else:
                    shot_outcome = "OUT! (MISS)"
                    outcome_color = GRID_COLOR
        else:
            # Goalkeeper slowly returns to center
            if not flick_active and not keeper_is_jumping: # Only return to center if not jumping
                target_keeper_x, target_keeper_y = 0.0, BALL_RADIUS
                keeper_x += (target_keeper_x - keeper_x) * 2.0 * dt
                keeper_y += (target_keeper_y - keeper_y) * 2.0 * dt

        # 2. Rendering
        screen.fill(BG_COLOR)
        
        # Draw grass pitch base line
        pygame.draw.rect(screen, PITCH_COLOR, (0, GOAL_BOTTOM_Y, WIDTH, HEIGHT - GOAL_BOTTOM_Y))
        pygame.draw.line(screen, WHITE, (0, GOAL_BOTTOM_Y), (WIDTH, GOAL_BOTTOM_Y), 3)
        
        # Draw 9-zone goal posts & divisions
        goal_rect = pygame.Rect(
            GOAL_CENTER_X - int((GOAL_WIDTH / 2) * GOAL_SCALE),
            GOAL_BOTTOM_Y - int(GOAL_HEIGHT * GOAL_SCALE),
            int(GOAL_WIDTH * GOAL_SCALE),
            int(GOAL_HEIGHT * GOAL_SCALE)
        )
        pygame.draw.rect(screen, WHITE, goal_rect, 4)
        
        # Draw zone boundaries & ID labels
        for idx in range(9):
            z_rect = get_zone_rect(idx)
            pygame.draw.rect(screen, GRID_COLOR, z_rect, 1)
            # Label
            lbl = font.render(f"Zone {idx + 1}", True, GRID_COLOR)
            screen.blit(lbl, (z_rect.x + 8, z_rect.y + 8))
            
            # Highlight predicted zone
            if ai_prediction is not None and ai_prediction == idx:
                glow = pygame.Surface((z_rect.width, z_rect.height), pygame.SRCALPHA)
                glow.fill((59, 130, 246, 50)) # semi-transparent neon blue
                screen.blit(glow, z_rect)
                pygame.draw.rect(screen, NEON_BLUE, z_rect, 2)
                
        # Draw Goalkeeper Figure (Vector art representation)
        gk_px, gk_py = to_screen_coords(keeper_x, keeper_y)
        
        # Colors for the keeper figure
        jersey_color = (234, 179, 8)    # Yellow 500
        jersey_stripe = (15, 23, 42)    # Dark Slate
        skin_color = (253, 186, 116)    # Skin peach peach-300
        glove_color = (249, 115, 22)    # Orange 500
        pants_color = (30, 41, 59)      # Slate 800
        
        # 1. Draw Legs (stretching down)
        baseline_y = min(GOAL_BOTTOM_Y, gk_py + 50)
        pygame.draw.line(screen, pants_color, (gk_px - 8, gk_py + 10), (gk_px - 12, baseline_y), 6)
        pygame.draw.line(screen, pants_color, (gk_px + 8, gk_py + 10), (gk_px + 12, baseline_y), 6)
        
        # 2. Draw Torso (Jersey)
        torso_rect = pygame.Rect(gk_px - 15, gk_py - 15, 30, 30)
        pygame.draw.rect(screen, jersey_color, torso_rect, border_radius=4)
        # Jersey stripes
        pygame.draw.line(screen, jersey_stripe, (gk_px - 15, gk_py - 5), (gk_px + 15, gk_py - 5), 3)
        pygame.draw.line(screen, jersey_stripe, (gk_px - 15, gk_py + 5), (gk_px + 15, gk_py + 5), 3)
        
        # 3. Draw Head
        pygame.draw.circle(screen, skin_color, (gk_px, gk_py - 25), 9)
        # Draw hair / cap
        pygame.draw.rect(screen, (30, 41, 59), (gk_px - 9, gk_py - 31, 18, 7), border_radius=3)
        
        # 4. Outstretched Arms & Gloves (Diving posture)
        # Left Arm
        pygame.draw.line(screen, jersey_color, (gk_px - 15, gk_py - 10), (gk_px - 45, gk_py - 5), 6)
        # Right Arm
        pygame.draw.line(screen, jersey_color, (gk_px + 15, gk_py - 10), (gk_px + 45, gk_py - 5), 6)
        
        # Goalkeeper Gloves (Orange)
        pygame.draw.circle(screen, glove_color, (gk_px - 47, gk_py - 4), 8)
        pygame.draw.circle(screen, glove_color, (gk_px + 47, gk_py - 4), 8)
        
        # Draw Ball
        if ball.active or len(ball.trajectory) > 0:
            # Draw trajectory path
            for pt in ball.trajectory:
                px, py = to_screen_coords(pt[0], pt[1])
                # Scale dot based on depth (z coord)
                z_depth = pt[2]
                traj_radius = max(2, int(6 * (1.0 - z_depth / PENALTY_DISTANCE)))
                pygame.draw.circle(screen, (100, 116, 139), (px, py), traj_radius)
                
            # Draw current ball
            b_px, b_py = to_screen_coords(ball.x, ball.y)
            # Scale ball size as it approaches the goal
            ball_scale = max(6, int(15 * (ball.z / PENALTY_DISTANCE)))
            pygame.draw.circle(screen, WHITE, (b_px, b_py), ball_scale)
            # Ball inner core
            pygame.draw.circle(screen, (200, 200, 200), (b_px, b_py), max(2, ball_scale - 3))
            
        # Draw Touch / Flick gesture vector while swiping
        if flick_active and len(touch_points) >= 2:
            pygame.draw.lines(screen, WHITE, False, touch_points, 3)
            pygame.draw.circle(screen, NEON_BLUE, touch_points[-1], 8)
            
        # Draw Telemetry & Debug Interface
        # Header / Title
        title_text = font_title.render("POCKET-KEEPER AI: DEMO TEST HARNESS", True, TEXT_COLOR)
        screen.blit(title_text, (20, 20))
        
        # Subtitle
        sub_text = font.render("Cihaz İçi / On-Device Predictive Goalkeeper (Arm Quantized INT8)", True, (148, 163, 184))
        screen.blit(sub_text, (20, 50))
        
        # Statistics Panel
        stats_x = 20
        stats_y = 110
        pygame.draw.rect(screen, (30, 41, 59), (stats_x, stats_y, 250, 120), border_radius=6)
        pygame.draw.rect(screen, NEON_BLUE, (stats_x, stats_y, 250, 120), 1, border_radius=6)
        
        s1 = font_bold.render("İSTATİSTİKLER (STATS)", True, TEXT_COLOR)
        s2 = font.render(f"Toplam Şut (Shots): {total_shots}", True, TEXT_COLOR)
        s3 = font.render(f"Goller (Goals): {goals}", True, NEON_RED)
        s4 = font.render(f"Kurtarışlar (Saves): {saves}", True, NEON_GREEN)
        screen.blit(s1, (stats_x + 15, stats_y + 10))
        screen.blit(s2, (stats_x + 15, stats_y + 35))
        screen.blit(s3, (stats_x + 15, stats_y + 60))
        screen.blit(s4, (stats_x + 15, stats_y + 85))
        
        # Telemetry Panel
        tel_x = 285
        tel_y = 110
        pygame.draw.rect(screen, (30, 41, 59), (tel_x, tel_y, 350, 120), border_radius=6)
        pygame.draw.rect(screen, NEON_BLUE, (tel_x, tel_y, 350, 120), 1, border_radius=6)
        
        t1 = font_bold.render("ARM EDGE-AI TELEMETRİ (TELEMETRY)", True, TEXT_COLOR)
        t2 = font.render(f"Model Boyutu (Model Size): 4.16 KB (INT8)", True, TEXT_COLOR)
        t3 = font.render(f"Tahmin Gecikmesi (Inference Latency): {inference_time:.3f} ms", True, NEON_GREEN if inference_time < 2.0 else WHITE)
        t4 = font.render(f"Tahmin Edilen Bölge (Predicted): {f'Zone {ai_prediction + 1}' if ai_prediction is not None else 'BEKLEMEDE'}", True, NEON_BLUE)
        screen.blit(t1, (tel_x + 15, tel_y + 10))
        screen.blit(t2, (tel_x + 15, tel_y + 35))
        screen.blit(t3, (tel_x + 15, tel_y + 60))
        screen.blit(t4, (tel_x + 15, tel_y + 85))
        
        # AI Output Probabilities Chart
        prob_x = 650
        prob_y = 110
        pygame.draw.rect(screen, (30, 41, 59), (prob_x, prob_y, 230, 120), border_radius=6)
        pygame.draw.rect(screen, NEON_BLUE, (prob_x, prob_y, 230, 120), 1, border_radius=6)
        
        pr1 = font_bold.render("AI TAHMİN OLASILIKLARI", True, TEXT_COLOR)
        screen.blit(pr1, (prob_x + 10, prob_y + 10))
        
        # Draw minibar chart for 9 zones
        for z_i in range(9):
            bar_w = int(ai_probs[z_i] * 120)
            bar_rect = pygame.Rect(prob_x + 60, prob_y + 35 + z_i * 9, bar_w, 6)
            pygame.draw.rect(screen, NEON_BLUE, bar_rect)
            lbl = font.render(f"Z{z_i+1}: {int(ai_probs[z_i]*100)}%", True, (148, 163, 184))
            # Smaller scale text
            tiny_font = pygame.font.SysFont("Outfit", 9)
            lbl_surf = tiny_font.render(f"Z{z_i+1}: {int(ai_probs[z_i]*100)}%", True, TEXT_COLOR)
            screen.blit(lbl_surf, (prob_x + 10, prob_y + 35 + z_i * 9 - 2))
            
        # Outcome / Result announcement text
        if shot_outcome:
            outcome_surf = font_title.render(shot_outcome, True, outcome_color)
            screen.blit(outcome_surf, (WIDTH // 2 - outcome_surf.get_width() // 2, HEIGHT - 90))
            
            hint_surf = font.render("Yeni şut çekmek için farenin sol tıkına basıp sürükleyin.", True, (148, 163, 184))
            screen.blit(hint_surf, (WIDTH // 2 - hint_surf.get_width() // 2, HEIGHT - 50))
        else:
            if flick_active:
                status_surf = font.render("SÜRÜKLE (İlk 15-30ms algılanıyor...)", True, NEON_BLUE)
            else:
                status_surf = font_bold.render("ŞUT ÇEKMEK İÇİN FAREYİ SOL TIK İLE SÜRÜKLEYİN (FLICK TO SHOOT)", True, TEXT_COLOR)
            screen.blit(status_surf, (WIDTH // 2 - status_surf.get_width() // 2, HEIGHT - 80))
            
        pygame.display.flip()
        
    pygame.quit()

if __name__ == "__main__":
    main()
