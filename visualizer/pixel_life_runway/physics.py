import pygame
import math
import random

# --- 1. STICKY SELECTION LOGIC ---
def get_selected_node(cursor_x, cursor_y, nodes, current_selected_node):
    if cursor_x < 0 or cursor_y < 0:
        return None
        
    GRAB_RADIUS = 20     
    RELEASE_RADIUS = 35  
    
    if current_selected_node in nodes:
        dx = cursor_x - current_selected_node['x']
        dy = cursor_y - current_selected_node['y']
        if math.hypot(dx, dy) <= RELEASE_RADIUS:
            return current_selected_node 
            
    closest_node = None
    min_dist = GRAB_RADIUS
    
    for node in nodes:
        dx = cursor_x - node['x']
        dy = cursor_y - node['y']
        distance = math.hypot(dx, dy)
        if distance < min_dist:
            min_dist = distance
            closest_node = node
            
    return closest_node

# --- 2. PARTICLE EMISSION LOGIC (Always Emits from Center, Default to Size 1) ---
def spawn_particles(particles, nodes, finger_state, current_time):
    # Node Emission
    for node in nodes:
        if node['emit_rate'] > 0 and len(node.get('emit_colors', [])) > 0:
            time_between_emissions = 1.0 / node['emit_rate']
            if current_time - node.get('last_emit_time', 0) >= time_between_emissions:
                angle = random.uniform(0, 2 * math.pi)
                speed = node['emit_speed']
                color = random.choice(node['emit_colors'])
                p_size = node.get('emit_size', 1) # Default to 1
                
                # Emit strictly from center coordinate
                particles.append({
                    'x': node['x'],
                    'y': node['y'],
                    'vx': math.cos(angle) * speed,
                    'vy': math.sin(angle) * speed,
                    'color': color,
                    'size': p_size,
                    'life': 255 
                })
                node['last_emit_time'] = current_time

    # Finger Emission
    if finger_state['mode'] == 'emit' and finger_state['x'] > 0:
        rate = finger_state.get('emit_rate', 10)
        time_between = 1.0 / rate if rate > 0 else 0
        if current_time - finger_state.get('last_emit_time', 0) >= time_between:
            angle = random.uniform(0, 2 * math.pi)
            speed = finger_state.get('emit_speed', 3.0)
            colors = finger_state.get('emit_colors', [(255, 255, 255)])
            
            particles.append({
                'x': finger_state['x'],
                'y': finger_state['y'],
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'color': random.choice(colors),
                'size': 1,
                'life': 255
            })
            finger_state['last_emit_time'] = current_time

# --- 3. PHYSICS ENGINE (Geometric Bouncing & Attraction) ---
def update_physics(particles, nodes, finger_state, global_state, canvas_w, canvas_h):
    grav_x = global_state.get('gravity_x', 0)
    grav_y = global_state.get('gravity_y', 0)
    
    for i in range(len(particles) - 1, -1, -1):
        p = particles[i]
        
        # 1. Apply Global Gravity
        p['vx'] += grav_x
        p['vy'] += grav_y
        
        # 2. Apply Node Physics (Attract vs Elastic Bouncing on Circumference)
        for node in nodes:
            effect = node.get('physics_effect', 'none')
            if effect == 'none':
                continue
                
            dx = p['x'] - node['x']
            dy = p['y'] - node['y']
            dist = math.hypot(dx, dy)
            
            # The radius of our interaction circumference
            radius = node['size'] if node['size'] > 1 else 1.0
            
            if effect == 'bounce':
                # Check if the particle has penetrated the object's boundary
                if dist < radius + 1:
                    if dist > 0.1:
                        # Vector Normal pointing from center outward (circumference perpendicular)
                        nx = dx / dist
                        ny = dy / dist
                        
                        # Dot product of velocity and normal
                        dot_prod = p['vx'] * nx + p['vy'] * ny
                        
                        # Only bounce if particle is actually moving towards center
                        if dot_prod < 0:
                            # Reflect: V_new = V - 2 * (V.N) * N
                            p['vx'] = p['vx'] - 2 * dot_prod * nx
                            p['vy'] = p['vy'] - 2 * dot_prod * ny
                            
                            # Push particle slightly outside boundary to prevent collision stickiness
                            p['x'] = node['x'] + nx * (radius + 1.5)
                            p['y'] = node['y'] + ny * (radius + 1.5)
                            
                            # Apply a small friction dampening to the bounce
                            p['vx'] *= 0.95
                            p['vy'] *= 0.95
                            
            elif effect == 'attract':
                # Attract from edge of circumference
                eff_dist = max(1.0, dist - radius)
                if 1 < eff_dist < 200: 
                    force = node['physics_strength'] / eff_dist
                    # Pull towards center
                    p['vx'] -= (dx / dist) * force
                    p['vy'] -= (dy / dist) * force

        # 3. Apply Finger Physics
        if finger_state['x'] > 0 and finger_state['mode'] in ['attract', 'repel']:
            dx = finger_state['x'] - p['x']
            dy = finger_state['y'] - p['y']
            dist_sq = dx**2 + dy**2
            
            if 10 < dist_sq < 60000:
                dist = math.sqrt(dist_sq)
                force = finger_state.get('physics_strength', 2.0) / dist
                if finger_state['mode'] == 'repel':
                    force = -force
                    
                p['vx'] += (dx / dist) * force
                p['vy'] += (dy / dist) * force

        # 4. Move particle
        p['x'] += p['vx']
        p['y'] += p['vy']
        p['life'] -= 1.5 
        
        if (p['life'] <= 0 or 
            p['x'] < 0 or p['x'] > canvas_w or 
            p['y'] < 0 or p['y'] > canvas_h):
            particles.pop(i)

# --- 4. RENDER ENGINE ---
def draw_scene(screen, particles, nodes, selected_node, current_time):
    # Draw Particles (Strictly respects dynamic size 1-6)
    for p in particles:
        alpha_ratio = max(0, p['life'] / 255.0)
        c = p['color']
        faded_color = (int(c[0]*alpha_ratio), int(c[1]*alpha_ratio), int(c[2]*alpha_ratio))
        p_size = p.get('size', 1)
        pygame.draw.rect(screen, faded_color, (int(p['x']), int(p['y']), p_size, p_size))

    # Draw Nodes
    for node in nodes:
        x, y = int(node['x']), int(node['y'])
        size = node['size']
        color = node['color']
        anim = node['animation']
        
        draw_size = size
        draw_color = color
        
        if anim == "pulse":
            pulse_val = (math.sin(current_time * 6.0) + 1.0) / 2.0
            brightness = 0.5 + (0.5 * pulse_val)
            draw_color = (int(color[0] * brightness), int(color[1] * brightness), int(color[2] * brightness))
            draw_size = size + int(2.0 * pulse_val)
        
        elif anim == "blink":
            if int(current_time * 2) % 2 == 0:
                draw_color = (0, 0, 0)
                
        elif anim == "glisten":
            bright_mod = random.uniform(0.3, 1.0)
            draw_color = (int(color[0]*bright_mod), int(color[1]*bright_mod), int(color[2]*bright_mod))

        shape = node.get('shape', 'circle')
        if draw_color != (0, 0, 0):
            if draw_size == 1:
                # Absolute Single-Pixel Override
                pygame.draw.rect(screen, draw_color, (x, y, 1, 1))
            elif shape == "square":
                pygame.draw.rect(screen, draw_color, (x - draw_size, y - draw_size, draw_size*2, draw_size*2))
            elif shape == "rectangle":
                pygame.draw.rect(screen, draw_color, (x - draw_size*2, y - draw_size, draw_size*4, draw_size*2))
            elif shape == "line":
                pygame.draw.line(screen, draw_color, (x - draw_size*2, y), (x + draw_size*2, y), 1)
            elif shape == "vertical_line":
                pygame.draw.line(screen, draw_color, (x, y - draw_size*2), (x, y + draw_size*2), 1)
            elif shape == "triangle":
                p1 = (x, y - draw_size)
                p2 = (x - draw_size, y + draw_size)
                p3 = (x + draw_size, y + draw_size)
                pygame.draw.polygon(screen, draw_color, [p1, p2, p3])
            else: 
                pygame.draw.circle(screen, draw_color, (x, y), draw_size)

        if node == selected_node and draw_size > 1:
            pygame.draw.circle(screen, (255, 255, 255), (x, y), draw_size + 4, 1)