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

# --- 2. PARTICLE EMISSION LOGIC (Always Emits from Center) ---
def spawn_particles(particles, nodes, finger_state, current_time):
    # Node Emission
    for node in nodes:
        if node['emit_rate'] > 0 and len(node.get('emit_colors', [])) > 0:
            time_between_emissions = 1.0 / node['emit_rate']
            if current_time - node.get('last_emit_time', 0) >= time_between_emissions:
                angle = random.uniform(0, 2 * math.pi)
                speed = node['emit_speed']
                color = random.choice(node['emit_colors'])
                p_size = node.get('emit_size', 1) 
                
                particles.append({
                    'x': node['x'], 'y': node['y'],
                    'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                    'color': color, 'size': p_size, 'life': 255 
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
                'x': finger_state['x'], 'y': finger_state['y'],
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'color': random.choice(colors), 'size': 1, 'life': 255
            })
            finger_state['last_emit_time'] = current_time


# --- 3. PERIMETER GEOMETRY HELPERS ---
def closest_point_on_segment(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0: return x1, y1
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))
    return x1 + t * dx, y1 + t * dy

def get_shape_perimeter_info(px, py, node):
    """Calculates the closest point on the exact perimeter of the shape."""
    shape = node.get('shape', 'circle')
    cx, cy = node['x'], node['y']
    s = max(1.0, float(node.get('size', 1)))
    
    bx, by = cx, cy
    is_inside = False
    
    if shape == 'circle' or s == 1.0:
        dx, dy = px - cx, py - cy
        d = math.hypot(dx, dy)
        if d <= s:
            is_inside = True
            bx, by = (cx, cy - s) if d == 0 else (cx + (dx/d)*s, cy + (dy/d)*s)
        else:
            bx, by = cx + (dx/d)*s, cy + (dy/d)*s
            
    elif shape in ['square', 'rectangle']:
        hw = s if shape == 'square' else s * 2
        hh = s
        dx, dy = px - cx, py - cy
        if abs(dx) <= hw and abs(dy) <= hh:
            is_inside = True
            dl, dr, dt, db = dx + hw, hw - dx, dy + hh, hh - dy
            min_d = min(dl, dr, dt, db)
            if min_d == dr: bx, by = cx + hw, py
            elif min_d == dl: bx, by = cx - hw, py
            elif min_d == db: bx, by = px, cy + hh
            else: bx, by = px, cy - hh
        else:
            bx = cx + max(-hw, min(hw, dx))
            by = cy + max(-hh, min(hh, dy))
            
    elif shape == 'line':
        bx, by = closest_point_on_segment(px, py, cx - s*2, cy, cx + s*2, cy)
        
    elif shape == 'vertical_line':
        bx, by = closest_point_on_segment(px, py, cx, cy - s*2, cx, cy + s*2)
        
    elif shape == 'triangle':
        p1, p2, p3 = (cx, cy - s), (cx - s, cy + s), (cx + s, cy + s)
        def sign(pt, v1, v2): return (pt[0]-v2[0])*(v1[1]-v2[1]) - (v1[0]-v2[0])*(pt[1]-v2[1])
        b1, b2, b3 = sign((px,py), p1, p2) < 0, sign((px,py), p2, p3) < 0, sign((px,py), p3, p1) < 0
        is_inside = ((b1 == b2) and (b2 == b3))
        
        edges = [(p1, p2), (p2, p3), (p3, p1)]
        min_d = float('inf')
        for e in edges:
            cpx, cpy = closest_point_on_segment(px, py, e[0][0], e[0][1], e[1][0], e[1][1])
            d = math.hypot(px - cpx, py - cpy)
            if d < min_d: min_d, bx, by = d, cpx, cpy

    # Calculate Normal and Distance to Perimeter
    nx, ny = (bx - px, by - py) if is_inside else (px - bx, py - by)
    dist = -math.hypot(nx, ny) if is_inside else math.hypot(nx, ny)
    
    if math.hypot(nx, ny) == 0: nx, ny = 0, -1
    else:
        mag = math.hypot(nx, ny)
        nx /= mag; ny /= mag
        
    return bx, by, nx, ny, dist

# --- 4. PHYSICS ENGINE (Perimeter Based) ---
def update_physics(particles, nodes, finger_state, global_state, canvas_w, canvas_h):
    grav_x = global_state.get('gravity_x', 0)
    grav_y = global_state.get('gravity_y', 0)
    
    for i in range(len(particles) - 1, -1, -1):
        p = particles[i]
        p['vx'] += grav_x
        p['vy'] += grav_y
        
        # Node Perimeter Physics
        for node in nodes:
            effect = node.get('physics_effect', 'none')
            if effect == 'none': continue
                
            bx, by, nx, ny, dist_to_boundary = get_shape_perimeter_info(p['x'], p['y'], node)
            
            if effect == 'bounce':
                if dist_to_boundary <= 1.5:
                    dot_prod = p['vx'] * nx + p['vy'] * ny
                    if dot_prod < 0: # Only bounce if heading INWARDS
                        p['vx'] -= 2 * dot_prod * nx
                        p['vy'] -= 2 * dot_prod * ny
                        p['x'] = bx + nx * 2.0
                        p['y'] = by + ny * 2.0
                        p['vx'] *= 0.95
                        p['vy'] *= 0.95
                        
            elif effect == 'attract':
                if 0 < dist_to_boundary < 200:
                    force = node['physics_strength'] / max(1.0, dist_to_boundary)
                    # nx/ny point outward from the shape, so subtract to pull inwards
                    p['vx'] -= nx * force
                    p['vy'] -= ny * force

        # Finger Point Physics
        if finger_state['x'] > 0 and finger_state['mode'] in ['attract', 'repel']:
            dx = finger_state['x'] - p['x']
            dy = finger_state['y'] - p['y']
            dist_sq = dx**2 + dy**2
            
            if 10 < dist_sq < 60000:
                dist = math.sqrt(dist_sq)
                force = finger_state.get('physics_strength', 2.0) / dist
                if finger_state['mode'] == 'repel': force = -force
                p['vx'] += (dx / dist) * force
                p['vy'] += (dy / dist) * force

        p['x'] += p['vx']
        p['y'] += p['vy']
        p['life'] -= 1.5 
        
        if (p['life'] <= 0 or p['x'] < 0 or p['x'] > canvas_w or p['y'] < 0 or p['y'] > canvas_h):
            particles.pop(i)

# --- 5. RENDER ENGINE ---
def draw_scene(screen, particles, nodes, selected_node, current_time):
    for p in particles:
        alpha_ratio = max(0, p['life'] / 255.0)
        c = p['color']
        faded_color = (int(c[0]*alpha_ratio), int(c[1]*alpha_ratio), int(c[2]*alpha_ratio))
        p_size = p.get('size', 1)
        pygame.draw.rect(screen, faded_color, (int(p['x']), int(p['y']), p_size, p_size))

    for node in nodes:
        x, y = int(node['x']), int(node['y'])
        size, color, anim = node['size'], node['color'], node['animation']
        draw_size, draw_color = size, color
        
        if anim == "pulse":
            pulse_val = (math.sin(current_time * 6.0) + 1.0) / 2.0
            brightness = 0.5 + (0.5 * pulse_val)
            draw_color = (int(color[0] * brightness), int(color[1] * brightness), int(color[2] * brightness))
            draw_size = size + int(2.0 * pulse_val)
        elif anim == "blink" and int(current_time * 2) % 2 == 0:
            draw_color = (0, 0, 0)
        elif anim == "glisten":
            bright_mod = random.uniform(0.3, 1.0)
            draw_color = (int(color[0]*bright_mod), int(color[1]*bright_mod), int(color[2]*bright_mod))

        shape = node.get('shape', 'circle')
        if draw_color != (0, 0, 0):
            if draw_size == 1:
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
                pygame.draw.polygon(screen, draw_color, [(x, y - draw_size), (x - draw_size, y + draw_size), (x + draw_size, y + draw_size)])
            else: 
                pygame.draw.circle(screen, draw_color, (x, y), draw_size)

        # Draw the selection ring for ALL nodes (including size 1 pixels)
        if node == selected_node:
            ring_radius = draw_size + 4 if draw_size > 1 else 6
            pygame.draw.circle(screen, (255, 255, 255), (x, y), ring_radius, 1)