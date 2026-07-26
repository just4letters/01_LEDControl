import math
import random

def spawn_particles(particles, cursor_x, cursor_y, canvas_w, canvas_h, state):
    """Handles the math for spawning new particles based on the current environment state."""
    for _ in range(state['rain_rate']):
        if state['gravity'] == 0.0 or state['wind_swirl'] > 0.0:
            base_vx = random.uniform(-1.0, 1.0)
            base_vy = random.uniform(-1.0, 1.0)
            spawn_offscreen_y = random.randint(0, canvas_h) 
        else:
            base_vx = random.uniform(-0.1, 0.1) + state['wind']
            base_vy = random.uniform(0.1, 0.4) * (1 if state['gravity'] >= 0 else -1)
            spawn_offscreen_y = -20 if state['gravity'] >= 0 else canvas_h + 20 
            
        if state['emitter_source'] == "finger" and cursor_x > 0 and cursor_y > 0:
            spawn_x = cursor_x + random.uniform(-5, 5)
            spawn_y = cursor_y + random.uniform(-5, 5)
        else:
            spawn_x = random.randint(0, canvas_w)
            spawn_y = spawn_offscreen_y
            
        spawn_color = state['rain_color']
        if state['rain_color_2']:
            spawn_color = random.choice([state['rain_color'], state['rain_color_2']])
            
        particles.append({
            "x": spawn_x, "y": spawn_y,
            "vx": base_vx,
            "vy": base_vy,
            "color": spawn_color
        })
        
    # Cull oldest particles if we exceed the AI's requested density ceiling
    while len(particles) > state['max_particles']:
        particles.pop(0)

def update_physics(particles, obstacles, current_time, cursor_x, cursor_y, canvas_w, canvas_h, state):
    """Applies forces, velocity, and collision detection to all active particles."""
    for p in particles[:]:
        p['vy'] += state['gravity'] 
        p['vx'] += (state['wind'] * 0.1) # Scaled down for intuitive AI prompting
        
        # Swirl Physics
        if state['wind_swirl'] > 0:
            pulse_x = math.sin(current_time * 3.0 + p['y'] * 0.05) 
            pulse_y = math.cos(current_time * 2.5 + p['x'] * 0.05)
            p['vx'] += pulse_x * state['wind_swirl']
            p['vy'] += pulse_y * state['wind_swirl']
        
        # Finger Repulsion Physics
        if cursor_x > 0 and cursor_y > 0 and state['emitter_source'] != "finger":
            dx = p['x'] - cursor_x; dy = p['y'] - cursor_y
            dist = math.hypot(dx, dy)
            if 0 < dist < 40:
                force = (40 - dist) / 40.0
                p['vx'] += (dx / dist) * force * 1.0 
                p['vy'] += (dy / dist) * force * 1.0
                
        # Friction & Velocity Application
        p['vx'] *= 0.99 
        p['vy'] *= 0.99
        p['x'] += p['vx']
        p['y'] += p['vy']
        
        # Obstacle Collision Detection
        ps = state['particle_size']
        px, py = p['x'], p['y']
        
        for obs in obstacles:
            cx, cy, osz = obs['x'], obs['y'], obs['size']
            half_s = osz / 2
            
            if (cx - half_s < px + ps and cx + half_s > px and cy - half_s < py + ps and cy + half_s > py):
                pcx, pcy = px + (ps / 2), py + (ps / 2)
                
                d_left = abs(pcx - (cx - half_s))
                d_right = abs(pcx - (cx + half_s))
                d_top = abs(pcy - (cy - half_s))
                d_bottom = abs(pcy - (cy + half_s))
                
                min_dist = min(d_left, d_right, d_top, d_bottom)
                
                if min_dist == d_left:
                    p['vx'] = -abs(p['vx']) * 0.8 - 0.5
                    p['x'] = (cx - half_s) - ps
                elif min_dist == d_right:
                    p['vx'] = abs(p['vx']) * 0.8 + 0.5
                    p['x'] = cx + half_s
                elif min_dist == d_top:
                    p['vy'] = -abs(p['vy']) * 0.8 - 0.5
                    p['y'] = (cy - half_s) - ps
                elif min_dist == d_bottom:
                    p['vy'] = abs(p['vy']) * 0.8 + 0.5
                    p['y'] = cy + half_s
        
        # Boundary Logic (Wrap around for zero-gravity, kill for normal gravity)
        if state['gravity'] == 0.0 or state['wind_swirl'] > 0.0:
            if p['x'] < 0: p['x'] = canvas_w
            elif p['x'] > canvas_w: p['x'] = 0
            if p['y'] < 0: p['y'] = canvas_h
            elif p['y'] > canvas_h: p['y'] = 0
        else:
            if p['y'] > canvas_h + 100 or p['y'] < -100 or p['x'] < -100 or p['x'] > canvas_w + 100:
                if p in particles:
                    particles.remove(p)