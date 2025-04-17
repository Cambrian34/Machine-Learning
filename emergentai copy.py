# --- Imports ---
# REMOVED: from anyio import current_time (Unused)
import pygame
import numpy as np
import random
import math
import json
import pickle
import base64
import os
# REMOVED: import time (pygame.time is used instead)

# Initialize Pygame
pygame.init()

# --- Constants ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
APPLE_SIZE = 10
CREATURE_SIZE = 8
SCORE_ZONE_SIZE = 50
WORLD_PADDING = 20
BOX_SIZE = 20
NUM_BOXES = 10

# Simulation Parameters
POPULATION_SIZE = 100
ENEMY_POPULATION_SIZE = 0 # CHANGED: Moved initialization value here
MAX_APPLES = 30
APPLE_RESPAWN_RATE = 0.01 # CHANGED: Adjusted rate interpretation (see respawn logic)
STARTING_ENERGY = 130.0
ENERGY_DECAY_RATE = 0.1
MOVE_ENERGY_COST = 0.1 # Note: Not explicitly used, decay covers general cost
GATHER_ENERGY_BONUS = 30.0
MAX_ENERGY = 200.0
REST_ENERGY_REGAIN = 0.01 # CHANGED: Added constant for resting gain

# Genetic Algorithm Parameters
GENERATION_TIME = 3000
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.5
ELITISM_COUNT = 2

# NN Architecture Constants
HIDDEN_NODES_CREATURE = 10 # CHANGED: Added constant
HIDDEN_NODES_ENEMY = 6     # CHANGED: Adjusted to match expected weight size

# --- Utility Functions ---
# REMOVED: Unused activation functions (sigmoid, relu, leaky_relu)
# def sigmoid(x): ...
# def relu(x): ...
# def leaky_relu(x, alpha=0.01): ...

# --- Save/Load Functions ---
# NOTE: These functions save/load only the single BEST creature's state,
# NOTE: not the entire population. Resuming a simulation via load_simulation_state
# NOTE: will start the generation with only the previously best creature's NN
# NOTE: (potentially replicated) and the rest randomly initialized.
# NOTE: Consider pickling the entire population list for true state saving.

def save_best_creature(creature, filename="best_creature.save"):
    """Save the best creature's NN and stats to a file"""
    if creature is None:
        print("No best creature to save.")
        return
    # Ensure creature has necessary attributes before saving
    if not hasattr(creature, 'nn') or not hasattr(creature, 'apples_deposited_total') or not hasattr(creature, 'fitness'):
         print("Creature object is missing required attributes for saving.")
         return

    save_data = {
        'weights': base64.b64encode(pickle.dumps(creature.nn.get_weights())).decode('utf-8'),
        'apples_deposited_total': getattr(creature, 'apples_deposited_total', 0), # Use getattr for safety
        'fitness': getattr(creature, 'fitness', 0.0), # Use getattr for safety
        'params': {
            'NUM_INPUTS': Creature.NUM_INPUTS,
            'NUM_OUTPUTS': Creature.NUM_OUTPUTS,
            'HIDDEN_NODES': HIDDEN_NODES_CREATURE # CHANGED: Save hidden node count
        }
    }
    try:
        with open(filename, 'w') as f:
            json.dump(save_data, f)
        # CHANGED: Use f-string
        print(f"Saved best creature to {filename}")
    except Exception as e:
        print(f"Error saving best creature: {e}")


def save_simulation_state(generation, current_all_time_best, current_all_time_best_fitness, filename="simulation_state.save"):
    """Save the current simulation generation and best creature state"""
     # CHANGED: Pass state explicitly instead of relying on globals
    if current_all_time_best is None:
        print("No all-time best creature defined, cannot save simulation state.")
        return

    try:
        # Ensure the creature object is suitable for pickling (or just save its relevant data)
        # Here we still use the previous method, but acknowledge limitations.
        best_creature_data = pickle.dumps(current_all_time_best) # This might be large or fail if creature is complex
        state = {
            'generation': generation,
            'best_creature_pickle': base64.b64encode(best_creature_data).decode('utf-8'),
            'best_fitness': current_all_time_best_fitness
        }
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        # CHANGED: Use f-string
        print(f"Saved simulation state (generation {generation}) to {filename}")
    except Exception as e:
        print(f"Error saving simulation state: {e}")

def load_best_creature(filename="best_creature.save"):
    """Load a saved creature's NN from file"""
    try:
        with open(filename, 'r') as f:
            save_data = json.load(f)

        weights_data = base64.b64decode(save_data['weights'].encode('utf-8'))
        weights = pickle.loads(weights_data)

        # CHANGED: Use constants and saved params for NN creation
        num_inputs = save_data['params']['NUM_INPUTS']
        num_outputs = save_data['params']['NUM_OUTPUTS']
        hidden_nodes = save_data['params'].get('HIDDEN_NODES', HIDDEN_NODES_CREATURE) # Default if not saved

        nn = NeuralNetwork(num_inputs, hidden_nodes, num_outputs)
        nn.set_weights(weights)

        creature = Creature(
            random.randint(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING),
            random.randint(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING),
            nn
        )
        # Restore stats
        creature.apples_deposited_total = save_data.get('apples_deposited_total', 0)
        # Fitness is typically reset, but we can load it if needed for seeding
        # creature.fitness = save_data.get('fitness', 0)
        creature.fitness = 0.0 # Reset fitness for the new generation

        # CHANGED: Use f-string
        print(f"Loaded creature NN from {filename}")
        return creature
    except FileNotFoundError:
        print(f"Save file not found: {filename}")
        return None
    except Exception as e:
        # CHANGED: Use f-string
        print(f"Failed to load creature from {filename}: {e}")
        return None

def load_simulation_state(filename="simulation_state.save"):
    """Load the simulation generation number and best creature state"""
    # Returns (generation, loaded_best_creature, loaded_best_fitness)
    try:
        with open(filename, 'rb') as f:
            state = pickle.load(f)

        loaded_generation = state['generation']
        loaded_best_fitness = state['best_fitness']

        # Attempt to load the pickled creature
        try:
            creature_pickle_data = base64.b64decode(state['best_creature_pickle'])
            loaded_best_creature = pickle.loads(creature_pickle_data)
             # Basic validation
            if not isinstance(loaded_best_creature, Creature):
                print("Warning: Loaded object is not a Creature instance.")
                loaded_best_creature = None # Discard if type mismatch
        except Exception as e:
            print(f"Warning: Could not load pickled best creature: {e}")
            loaded_best_creature = None # Cannot use if loading failed

        # CHANGED: Use f-string
        print(f"Loaded simulation state from {filename}. Resuming at generation {loaded_generation + 1}.")
        return loaded_generation, loaded_best_creature, loaded_best_fitness
    except FileNotFoundError:
         print(f"Simulation state file not found: {filename}. Starting from generation 1.")
         return 0, None, -float('inf') # Return generation 0 (will become 1), no creature, default fitness
    except Exception as e:
        # CHANGED: Use f-string
        print(f"Failed to load simulation state from {filename}: {e}")
        return 0, None, -float('inf') # Return generation 0 (will become 1), no creature, default fitness

# --- Neural Network ---
class NeuralNetwork:
    # CHANGED: Use constants for hidden size
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size # Store sizes for saving/loading/reshaping
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights with small random values
        # Improved weight initialization using He initialization

        self.weights_ih = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2.0 / self.input_size)
        self.weights_ho = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2.0 / self.hidden_size)
        # Simple biases (can be evolved too, but starting simple)
        # NOTE: Consider adding biases to get_weights/set_weights/mutate for potentially better learning
        self.bias_h = np.zeros((1, self.hidden_size)) * 0.1
        self.bias_o = np.zeros((1, self.output_size)) * 0.1

    def predict(self, inputs):
        inputs = np.array(inputs).reshape(1, -1)
        # Use ReLU for hidden layer, tanh for output
        hidden_raw = np.dot(inputs, self.weights_ih) + self.bias_h
        hidden_activated = np.maximum(0, hidden_raw)  # ReLU
        output_raw = np.dot(hidden_activated, self.weights_ho) + self.bias_o
        output_activated = np.tanh(output_raw)
        return output_activated[0]

    def get_weights(self):
        # NOTE: Only includes weights, not biases currently
        return np.concatenate([self.weights_ih.flatten(), self.weights_ho.flatten()])

    def set_weights(self, flat_weights):
        ih_size = self.input_size * self.hidden_size
        ho_size = self.hidden_size * self.output_size
        bias_h_size = self.hidden_size
        bias_o_size = self.output_size
        
        expected_size = ih_size + ho_size + bias_h_size + bias_o_size
        if len(flat_weights) != expected_size:
            raise ValueError(f"Invalid weight size. Expected {expected_size}, got {len(flat_weights)}")
        
        ptr = 0
        self.weights_ih = flat_weights[ptr:ptr+ih_size].reshape((self.input_size, self.hidden_size))
        ptr += ih_size
        self.weights_ho = flat_weights[ptr:ptr+ho_size].reshape((self.hidden_size, self.output_size))
        ptr += ho_size
        self.bias_h = flat_weights[ptr:ptr+bias_h_size].reshape((1, self.hidden_size))
        ptr += bias_h_size
        self.bias_o = flat_weights[ptr:ptr+bias_o_size].reshape((1, self.output_size))

    def mutate(self, rate, strength):
        # Vectorized mutation using NumPy operations
        weights = self.get_weights()
        mask = np.random.rand(*weights.shape) < rate
        mutations = np.random.randn(*weights.shape) * strength
        new_weights = weights + mask * mutations
        self.set_weights(new_weights)


# --- Creature Class ---
class Creature:
    # CHANGED: Updated NUM_OUTPUTS to match indices
    NUM_INPUTS = 8
    NUM_OUTPUTS = 6 # turn, accelerate, gather, deposit, rest, flee

    # INPUTS indices
    IDX_DIST_APPLE = 0
    IDX_ANGLE_APPLE = 1
    IDX_DIST_ZONE = 2
    IDX_ANGLE_ZONE = 3
    IDX_ENERGY = 4
    IDX_DIST_ENEMY = 5
    IDX_ANGLE_ENEMY = 6
    IDX_APPLES_HELD = 7

    # OUTPUTS indices
    IDX_TURN = 0        # -1 (left) to +1 (right)
    IDX_ACCELERATE = 1  # -1 (brake/reverse) to +1 (accelerate)
    IDX_GATHER = 2      # > 0.5 to attempt gather
    IDX_DEPOSIT = 3     # > 0.5 to attempt deposit
    #IDX_REST = 5        # > 0.5 to attempt rest (if slow)
    IDX_FLEE = 4        # > 0.5 to attempt flee (if enemy nearby)

    GATHER_RADIUS = CREATURE_SIZE + APPLE_SIZE + 5


    def __init__(self, x, y, nn):
        self.pos = np.array([x, y], dtype=np.float64)
        self.velocity = np.array([0.0, 0.0], dtype=np.float64)
        self.angle = random.uniform(0, 2 * math.pi)
        self.speed = 0.0
        self.max_speed = 3.0
        self.max_turn_rate = 0.2

        self.energy = STARTING_ENERGY
        self.apples_held = 0 # CHANGED: Start with 0 apples
        self.max_apples = 5
        self.nn = nn
        self.fitness = 0.0
        self.apples_deposited_total = 0
        self.lifetime = 0

        self.sight_radius = 190

        # REMOVED: Unused memory attributes
        # self.short_term_memory = {...}
        # self.racial_memory = {...}

        # CHANGED: Initialize font once
        self.font = pygame.font.SysFont(None, 18)

    def get_nearest_object(self, objects):
        if not objects:
            return None, self.sight_radius, 0.0

        # Vectorized distance calculation
        positions = np.array([obj.pos for obj in objects])
        deltas = positions - self.pos
        dists_sq = np.einsum('ij,ij->i', deltas, deltas)
        
        min_idx = np.argmin(dists_sq)
        min_dist_sq = dists_sq[min_idx]
        
        if min_dist_sq > self.sight_radius**2:
            return None, self.sight_radius, 0.0

        dist = math.sqrt(min_dist_sq)
        direction = deltas[min_idx]
        target_angle = math.atan2(direction[1], direction[0])
        relative_angle = (target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
        return objects[min_idx], dist, relative_angle / math.pi
    

    def get_inputs(self, apples, score_zone, enemies):
        inputs = np.zeros(Creature.NUM_INPUTS) # Use class constant

        # Nearest Apple Info
        nearest_apple, dist_apple, angle_apple = self.get_nearest_object(apples)
        # Normalize distance: 1.0 = very close, 0.0 = at sight radius edge or further
        inputs[Creature.IDX_DIST_APPLE] = max(0.0, (self.sight_radius - dist_apple) / self.sight_radius)
        inputs[Creature.IDX_ANGLE_APPLE] = angle_apple # Already normalized to [-1, 1]

        # Nearest Enemy Info
        nearest_enemy, dist_enemy, angle_enemy = self.get_nearest_object(enemies)
        # Normalize distance
        inputs[Creature.IDX_DIST_ENEMY] = max(0.0, (self.sight_radius - dist_enemy) / self.sight_radius)
        inputs[Creature.IDX_ANGLE_ENEMY] = angle_enemy # Already normalized to [-1, 1]


        # Score Zone Info
        zone_dir = score_zone.pos - self.pos
        dist_zone = np.linalg.norm(zone_dir)
        if dist_zone > 0: # Avoid division by zero if right on top
            zone_target_angle = math.atan2(zone_dir[1], zone_dir[0])
            zone_relative_angle = zone_target_angle - self.angle
            zone_relative_angle = (zone_relative_angle + math.pi) % (2 * math.pi) - math.pi
            # Normalize angle and distance (closer is higher value)
            # Using SCREEN_WIDTH as max distance measure here, might need adjustment
            inputs[Creature.IDX_DIST_ZONE] = max(0.0, (SCREEN_WIDTH - dist_zone) / SCREEN_WIDTH)
            inputs[Creature.IDX_ANGLE_ZONE] = zone_relative_angle / math.pi
        else:
            inputs[Creature.IDX_DIST_ZONE] = 1.0 # Max closeness
            inputs[Creature.IDX_ANGLE_ZONE] = 0.0 # No angle difference


        # Energy Level (normalized)
        #inputs[Creature.IDX_ENERGY] = self.energy / MAX_ENERGY

        # Apples Held (normalized)
        inputs[Creature.IDX_APPLES_HELD] = self.apples_held / self.max_apples

        return inputs

    def update(self, apples, score_zone, enemies, boxes, dt=1.0):
        inputs = self.get_inputs(apples, score_zone, enemies)
        actions = self.nn.predict(inputs)

        # --- Extract actions from NN ---
        turn_request = actions[Creature.IDX_TURN]       # -1 to 1
        accel_request = actions[Creature.IDX_ACCELERATE] # -1 to 1
        gather_attempt = actions[Creature.IDX_GATHER] > 0.5
        deposit_attempt = actions[Creature.IDX_DEPOSIT] > 0.5
        #rest_attempt = actions[Creature.IDX_REST] > 0.5
        flee_attempt = actions[Creature.IDX_FLEE] > 0.5

        # --- Handle Actions ---

        # Fleeing overrides other movement if active and enemy is near
        if flee_attempt:
            nearest_enemy, dist_enemy, angle_enemy = self.get_nearest_object(enemies)
            # Flee if enemy is within half sight radius (tunable threshold)
            if nearest_enemy and dist_enemy < self.sight_radius * 0.5:
                # Turn directly away from the enemy
                # angle_enemy is normalized [-1, 1], corresponds to [-pi, pi]
                # To turn away, we want the opposite direction, which means adding pi to the relative angle
                # In normalized terms, add/subtract 1 and wrap around [-1, 1]
                flee_turn_target = angle_enemy + 1.0 if angle_enemy < 0 else angle_enemy - 1.0
                # Simple proportional control for turning away
                turn_request = np.clip(flee_turn_target * 2.0, -1.0, 1.0) # Stronger turn away
                accel_request = 1.0 # Accelerate maximally
                #add fitness bonus for fleeing
                self.fitness += 0.5 # Optional: Small bonus for fleeing

        # Resting overrides acceleration if active and speed is low
        #if rest_attempt and self.speed < 0.1:
        #    self.energy += REST_ENERGY_REGAIN * dt
        #    self.energy = min(self.energy, MAX_ENERGY)
        #    accel_request = -1.0 # Ensure braking

        # Gather Apples
        # Optimized gathering using nearest apple check
        if gather_attempt and self.apples_held < self.max_apples:
            nearest_apple, dist_apple, _ = self.get_nearest_object(apples)
            if nearest_apple and dist_apple < self.GATHER_RADIUS:
                apples.remove(nearest_apple)
                self.apples_held += 1
                self.energy = min(self.energy + GATHER_ENERGY_BONUS, MAX_ENERGY)
                self.fitness += 2.0 + (0.5 if self.apples_held < self.max_apples else 0)

        # Deposit Apples
        if deposit_attempt and self.apples_held > 0:
            dist_sq_zone = np.dot(self.pos - score_zone.pos, self.pos - score_zone.pos)
            if dist_sq_zone < (SCORE_ZONE_SIZE + CREATURE_SIZE)**2:
                deposit_bonus = self.apples_held * 10 # Base bonus
                # Bonus for depositing more apples at once?
                deposit_bonus += (self.apples_held**2) * 0.5
                self.fitness += deposit_bonus
                self.apples_deposited_total += self.apples_held
                self.apples_held = 0
                # Optional: Energy bonus for successful deposit?
                self.energy += 20

        # --- Energy Decay ---
        self.energy -= ENERGY_DECAY_RATE * dt
        # Add movement cost based on acceleration/speed?
        move_cost = abs(accel_request) * 0.02 + self.speed * 0.01
        self.energy -= move_cost * dt

        # --- Movement Physics ---
        if self.energy > 0:
            # Apply turn
            self.angle += turn_request * self.max_turn_rate * dt
            self.angle %= (2 * math.pi) # Keep angle within [0, 2pi)

            # Apply acceleration/braking
            acceleration = accel_request * 0.2 # Tuning factor for acceleration strength
            self.speed += acceleration * dt
            # Apply drag/friction
            self.speed *= (1.0 - 0.05 * dt) # Friction factor
            self.speed = np.clip(self.speed, 0, self.max_speed) # Clamp speed

            # Update velocity vector based on new angle and speed
            self.velocity[0] = math.cos(self.angle) * self.speed
            self.velocity[1] = math.sin(self.angle) * self.speed

            # Update position
            new_pos = self.pos + self.velocity * dt

            # --- Collision Handling ---
            # Boundary Collision
            bounced = False
            if new_pos[0] < WORLD_PADDING + CREATURE_SIZE:
                new_pos[0] = WORLD_PADDING + CREATURE_SIZE
                self.angle = math.pi - self.angle
                bounced = True
            elif new_pos[0] > SCREEN_WIDTH - WORLD_PADDING - CREATURE_SIZE:
                new_pos[0] = SCREEN_WIDTH - WORLD_PADDING - CREATURE_SIZE
                self.angle = math.pi - self.angle
                bounced = True

            if new_pos[1] < WORLD_PADDING + CREATURE_SIZE:
                new_pos[1] = WORLD_PADDING + CREATURE_SIZE
                self.angle = -self.angle
                bounced = True
            elif new_pos[1] > SCREEN_HEIGHT - WORLD_PADDING - CREATURE_SIZE:
                new_pos[1] = SCREEN_HEIGHT - WORLD_PADDING - CREATURE_SIZE
                self.angle = -self.angle
                bounced = True

            if bounced:
                self.speed *= 0.8 # Lose some speed on bounce
                # Recalculate velocity after bounce angle change
                self.velocity[0] = math.cos(self.angle) * self.speed
                self.velocity[1] = math.sin(self.angle) * self.speed

            # Box Collision (simple push)
            nearby_boxes = spatial_grid.get_nearby(self.pos, CREATURE_SIZE + BOX_SIZE)

            for box in nearby_boxes:
                delta_pos = self.pos - box.pos # Vector from box center to creature center
                dist_sq = np.dot(delta_pos, delta_pos)
                min_dist = CREATURE_SIZE + box.size / 2
                if dist_sq < min_dist**2 and dist_sq > 1e-6: # Avoid division by zero if perfectly overlapping
                    dist = math.sqrt(dist_sq)
                    overlap = min_dist - dist
                    # Push direction is from box center to creature center
                    push_vec = delta_pos / dist
                    # Move creature out of overlap
                    new_pos += push_vec * overlap * 0.5 # Move creature slightly
                    # Apply force to box (simple impulse)
                    box.velocity += push_vec * 0.3 # Push box away

            self.pos = new_pos

        # --- Update Lifetime & Check Survival ---
        self.lifetime += dt
        # Return True if alive, False otherwise
        return self.energy > 0

    def draw(self, screen):
        # Draw direction line
        end_line = self.pos + np.array([math.cos(self.angle), math.sin(self.angle)]) * (CREATURE_SIZE + 5)
        pygame.draw.line(screen, (0, 255, 0), self.pos.astype(int), end_line.astype(int), 2)

        # Determine color based on apples deposited (example)
        # Maximize green component based on total deposited, capped at some value
        green_intensity = min(255, 100 + self.apples_deposited_total * 5)
        color = (0, green_intensity, 0)

        # Draw main body
        pygame.draw.circle(screen, color, self.pos.astype(int), CREATURE_SIZE)

        # Draw Energy Bar
        energy_ratio = max(0, self.energy / MAX_ENERGY)
        energy_bar_width = int(CREATURE_SIZE * 2 * energy_ratio)
        energy_bar_height = 3
        energy_bar_pos = self.pos - np.array([CREATURE_SIZE, CREATURE_SIZE + energy_bar_height + 1])
        # Draw red background bar
        pygame.draw.rect(screen, (255, 0, 0), (*energy_bar_pos.astype(int), CREATURE_SIZE * 2, energy_bar_height))
        # Draw green foreground bar
        if energy_bar_width > 0:
             pygame.draw.rect(screen, (0, 255, 0), (*energy_bar_pos.astype(int), energy_bar_width, energy_bar_height))

        # Draw apple count if holding apples
        if self.apples_held > 0:
            # Font initialized in __init__ now
            text = self.font.render(str(self.apples_held), True, (255, 255, 255))
            # Position text inside the circle
            text_rect = text.get_rect(center=self.pos.astype(int))
            screen.blit(text, text_rect)

# --- Box Class ---
# REMOVED: Duplicate Box class definition

# --- Enemy Class ---
class Enemy:
    # CHANGED: Use constants
    NUM_INPUTS = 6
    NUM_OUTPUTS = 3 # turn, accelerate, attack

    def __init__(self, x, y, nn=None):
        self.pos = np.array([x, y], dtype=np.float64)
        self.velocity = np.array([0.0, 0.0], dtype=np.float64)
        self.angle = random.uniform(0, 2 * math.pi)
        self.speed = 0.0
        self.max_speed = 2.5 # Slightly slower than creatures?
        self.max_turn_rate = 0.15
        self.health = 100 # Reduced health?
        self.attack_damage = 15
        self.attack_cooldown = 30 # Frames between attacks
        self.attack_timer = 0
        self.fitness = 0.0 # Reset fitness naming convention
        # CHANGED: Use constant for hidden nodes
        self.nn = nn if nn else NeuralNetwork(self.NUM_INPUTS, HIDDEN_NODES_ENEMY, self.NUM_OUTPUTS)
        self.sight_radius = 180 # Slightly less sight?

    # ... (get_inputs remains similar, but use Creature class constants)
    def get_inputs(self, creatures):
        inputs = np.zeros(Enemy.NUM_INPUTS)

        # Find nearest creature
        nearest_creature, dist_creature, angle_creature = self.get_nearest_object(creatures) # Use own method
        if nearest_creature:
            inputs[0] = max(0.0, (self.sight_radius - dist_creature) / self.sight_radius)
            inputs[1] = angle_creature # Normalized angle [-1, 1]
            # Use Creature constants for clarity if accessing its properties
            inputs[2] = nearest_creature.energy / MAX_ENERGY
        # No target: inputs[0], inputs[1], inputs[2] remain 0

        inputs[3] = self.health / 100.0 # Normalized health
        # Normalized position (relative to world size)
        inputs[4] = self.pos[0] / SCREEN_WIDTH
        inputs[5] = self.pos[1] / SCREEN_HEIGHT

        return inputs

    # CHANGED: Use get_nearest_object similar to Creature for angle calculation
    def get_nearest_object(self, objects):
        nearest_obj = None
        min_dist_sq = self.sight_radius**2

        for obj in objects:
            delta_pos = obj.pos - self.pos
            dist_sq = np.dot(delta_pos, delta_pos)
            if 0 < dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest_obj = obj

        if nearest_obj:
            dist = math.sqrt(min_dist_sq)
            direction_vec = nearest_obj.pos - self.pos
            target_angle = math.atan2(direction_vec[1], direction_vec[0])
            relative_angle = target_angle - self.angle
            relative_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi
            return nearest_obj, dist, relative_angle / math.pi # Normalized angle
        else:
            return None, self.sight_radius, 0.0

    def update(self, creatures, boxes, dt=1.0):
        inputs = self.get_inputs(creatures)
        actions = self.nn.predict(inputs)

        turn_request = actions[0]
        accel_request = actions[1]
        attack_attempt = actions[2] > 0.7 # Threshold for attack

        # --- Movement (similar to creature, maybe simpler?) ---
        self.angle += turn_request * self.max_turn_rate * dt
        self.angle %= (2 * math.pi)

        acceleration = accel_request * 0.15 # Enemy acceleration factor
        self.speed += acceleration * dt
        self.speed *= (1.0 - 0.06 * dt) # Enemy friction
        self.speed = np.clip(self.speed, 0, self.max_speed)

        self.velocity[0] = math.cos(self.angle) * self.speed
        self.velocity[1] = math.sin(self.angle) * self.speed
        new_pos = self.pos + self.velocity * dt

        # Boundary Collision (same as creature)
        bounced = False
        # ... (Boundary check code identical to Creature.update) ...
        if new_pos[0] < WORLD_PADDING + CREATURE_SIZE:
            new_pos[0] = WORLD_PADDING + CREATURE_SIZE
            self.angle = math.pi - self.angle; bounced = True
        elif new_pos[0] > SCREEN_WIDTH - WORLD_PADDING - CREATURE_SIZE:
            new_pos[0] = SCREEN_WIDTH - WORLD_PADDING - CREATURE_SIZE
            self.angle = math.pi - self.angle; bounced = True
        if new_pos[1] < WORLD_PADDING + CREATURE_SIZE:
            new_pos[1] = WORLD_PADDING + CREATURE_SIZE
            self.angle = -self.angle; bounced = True
        elif new_pos[1] > SCREEN_HEIGHT - WORLD_PADDING - CREATURE_SIZE:
            new_pos[1] = SCREEN_HEIGHT - WORLD_PADDING - CREATURE_SIZE
            self.angle = -self.angle; bounced = True

        if bounced:
            self.speed *= 0.8
            self.velocity[0] = math.cos(self.angle) * self.speed
            self.velocity[1] = math.sin(self.angle) * self.speed

        # Box Collision (same as creature)
        for box in boxes:
            delta_pos = self.pos - box.pos
            dist_sq = np.dot(delta_pos, delta_pos)
            min_dist = CREATURE_SIZE + box.size / 2 # Enemy uses CREATURE_SIZE too
            if dist_sq < min_dist**2 and dist_sq > 1e-6:
                dist = math.sqrt(dist_sq)
                overlap = min_dist - dist
                push_vec = delta_pos / dist
                new_pos += push_vec * overlap * 0.5
                box.velocity += push_vec * 0.3 # Push box

        self.pos = new_pos

        # --- Attack Logic ---
        if self.attack_timer > 0:
            self.attack_timer -= dt # Cooldown ticks down

        if attack_attempt and self.attack_timer <= 0:
            nearest_creature, dist, _ = self.get_nearest_object(creatures)
            # Attack if close enough
            attack_range_sq = (CREATURE_SIZE * 2.5)**2 # Slightly larger than just touching
            if nearest_creature and dist*dist < attack_range_sq:
                nearest_creature.energy -= self.attack_damage
                self.fitness += 5 # Reward for successful attack
                self.attack_timer = self.attack_cooldown # Reset cooldown
                # Optional: Small health gain on successful attack?
                self.health = min(100, self.health + 2)


        # --- Health & Fitness ---
        self.fitness += 0.01 * dt # Small reward for surviving
        self.health -= 0.05 * dt # Slower health decay?
        return self.health > 0

    # REMOVED: check_collision_with_box method (logic wasn't robust)

    def draw(self, screen):
        # Draw direction line (optional)
        end_line = self.pos + np.array([math.cos(self.angle), math.sin(self.angle)]) * (CREATURE_SIZE + 4)
        pygame.draw.line(screen, (255, 100, 100), self.pos.astype(int), end_line.astype(int), 1)
        # Draw body
        pygame.draw.circle(screen, (128, 0, 128), self.pos.astype(int), CREATURE_SIZE) # Purple
        # Draw health bar (optional)
        health_ratio = max(0, self.health / 100.0)
        health_bar_width = int(CREATURE_SIZE * 1.5 * health_ratio)
        health_bar_pos = self.pos - np.array([CREATURE_SIZE*0.75, CREATURE_SIZE + 4])
        pygame.draw.rect(screen, (255, 100, 100), (*health_bar_pos.astype(int), int(CREATURE_SIZE*1.5), 2))
        if health_bar_width > 0:
             pygame.draw.rect(screen, (0, 255, 100), (*health_bar_pos.astype(int), health_bar_width, 2))

# --- ScoreZone Class ---
class ScoreZone:
    def __init__(self, x, y):
        self.pos = np.array([x, y], dtype=np.float64)

    def draw(self, screen):
        pygame.draw.circle(screen, (0, 0, 255, 100), self.pos.astype(int), SCORE_ZONE_SIZE) # Make semi-transparent

# --- Apple Class ---
class Apple:
    def __init__(self, x, y):
        self.pos = np.array([x, y], dtype=np.float64)

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 0, 0), self.pos.astype(int), APPLE_SIZE)

# --- Box Class ---
class Box:
    def __init__(self, x, y):
        self.pos = np.array([x, y], dtype=np.float64)
        self.velocity = np.array([0.0, 0.0], dtype=np.float64)
        self.size = BOX_SIZE
        self.color = (139, 69, 19)  # Brown

    def update(self):
        # Apply friction
        self.velocity *= 0.85
        self.pos += self.velocity
        # Keep within bounds
        self.pos[0] = np.clip(self.pos[0], WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING - self.size)
        self.pos[1] = np.clip(self.pos[1], WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING - self.size)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.pos[0], self.pos[1], self.size, self.size))

# --draw world boundaries--
def draw_world_boundaries(screen):
    boundary_color = (100, 100, 100, 150)  # Semi-transparent gray
    boundary_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    # Draw four rectangles around the edges
    pygame.draw.rect(boundary_surface, boundary_color, (0, 0, SCREEN_WIDTH, WORLD_PADDING))
    pygame.draw.rect(boundary_surface, boundary_color, (0, SCREEN_HEIGHT - WORLD_PADDING, SCREEN_WIDTH, WORLD_PADDING))
    pygame.draw.rect(boundary_surface, boundary_color, (0, 0, WORLD_PADDING, SCREEN_HEIGHT))
    pygame.draw.rect(boundary_surface, boundary_color, (SCREEN_WIDTH - WORLD_PADDING, 0, WORLD_PADDING, SCREEN_HEIGHT))
    screen.blit(boundary_surface, (0, 0))


# --- Genetic Algorithm Functions ---
# REMOVED: First (unused) set of selection/reproduction functions

# --- SECOND SET (KEEP THESE) ---
def selection(population):
    """Selects parents based on fitness (Tournament Selection variation)"""
    parents = []
    population.sort(key=lambda creature: creature.fitness, reverse=True) # Sort needed for elitism later

    # Keep elites for direct copying later in reproduction
    elites = population[:ELITISM_COUNT]

    # Select remaining parents using tournament selection
    num_parents_needed = (POPULATION_SIZE // 2) # Aim for roughly half the pop size as parents pool
    tournament_size = 5 # Pick 5 individuals randomly

    selected_parents = []
    if len(population) > tournament_size: # Ensure enough individuals for tournament
        for _ in range(num_parents_needed):
            tournament = random.sample(population, tournament_size)
            winner = max(tournament, key=lambda c: c.fitness)
            selected_parents.append(winner)
    else: # Fallback if population is very small
        selected_parents = population[:num_parents_needed]

    # Combine elites and tournament winners for the parent pool
    # Ensure elites are included if they didn't win tournaments
    parent_pool = elites + [p for p in selected_parents if p not in elites]

    if not parent_pool: # Handle edge case of empty population
        return []

    return parent_pool


def reproduction(parents, generation):
    """Creates the next generation population using elitism, crossover, and mutation."""
    next_population = []

    if not parents: # Handle empty parent list
        # Re-initialize if no parents (e.g., after full extinction)
        print("Warning: No parents selected for reproduction. Re-initializing.")
        for _ in range(POPULATION_SIZE):
            nn = NeuralNetwork(Creature.NUM_INPUTS, HIDDEN_NODES_CREATURE, Creature.NUM_OUTPUTS)
            next_population.append(Creature(random.randint(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING),
                                            random.randint(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING), nn))
        return next_population

    # Elitism: Copy the best N creatures directly
    # Assumes parents list already contains the elites from selection
    elites = parents[:ELITISM_COUNT]
    for elite_parent in elites:
        # Create a new Creature instance but copy the NN weights
        child_nn = NeuralNetwork(Creature.NUM_INPUTS, HIDDEN_NODES_CREATURE, Creature.NUM_OUTPUTS)
        child_nn.set_weights(elite_parent.nn.get_weights()) # Copy weights
        child = Creature(random.randint(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING),
                       random.randint(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING),
                       child_nn)
        child.fitness = 0 # Reset fitness for the new generation
        child.apples_deposited_total = 0 # Reset deposited count
        next_population.append(child)

    # Crossover and Mutation to fill the rest of the population
    while len(next_population) < POPULATION_SIZE:
        # Select two parents (can be the same if parent pool is small)
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)

        # Perform crossover
        p1_weights = parent1.nn.get_weights()
        p2_weights = parent2.nn.get_weights()
        child_weights = crossover(p1_weights, p2_weights) # Use the crossover function

        # Create child NN and set weights
        child_nn = NeuralNetwork(Creature.NUM_INPUTS, HIDDEN_NODES_CREATURE, Creature.NUM_OUTPUTS)
        child_nn.set_weights(child_weights)

        # Mutate the child's NN
        child_nn.mutate(MUTATION_RATE, MUTATION_STRENGTH)

        # Create the new creature
        child = Creature(random.randint(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING),
                         random.randint(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING),
                         child_nn)
        next_population.append(child)

    return next_population


def crossover(weights1, weights2):
    """Performs single-point crossover between two sets of weights."""
    if len(weights1) != len(weights2):
        raise ValueError("Weight vectors must have the same length for crossover.")
    if len(weights1) < 2:
        return weights1 # Cannot perform crossover on vectors shorter than 2

    # Choose a crossover point (excluding the very beginning and end)
    crossover_point = random.randint(1, len(weights1) - 1)
    # Combine weights from parents
    child_weights = np.concatenate([weights1[:crossover_point], weights2[crossover_point:]])
    return child_weights

def evolve_enemies(enemy_population):
    """Evolves the enemy population using similar GA principles."""
    if not enemy_population:
        return [] # Return empty if no enemies to evolve

    enemy_population.sort(key=lambda e: e.fitness, reverse=True)

    # Similar selection (can reuse or have a separate one for enemies)
    # Using simple top half selection for enemies for variety
    num_parents = max(2, ENEMY_POPULATION_SIZE // 2) # Ensure at least 2 parents if possible
    parents = enemy_population[:num_parents]

    if not parents: return [] # Cannot reproduce without parents

    next_gen_enemies = []

    # Enemy Elitism (keep best 1 or 2)
    enemy_elitism_count = 1 # Keep only the single best enemy maybe
    for i in range(min(enemy_elitism_count, len(parents))):
        elite_nn = NeuralNetwork(Enemy.NUM_INPUTS, HIDDEN_NODES_ENEMY, Enemy.NUM_OUTPUTS)
        elite_nn.set_weights(parents[i].nn.get_weights())
        new_enemy = Enemy(random.randint(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING),
                          random.randint(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING),
                          elite_nn)
        next_gen_enemies.append(new_enemy)

    # Breed new enemies using crossover and mutation
    while len(next_gen_enemies) < ENEMY_POPULATION_SIZE:
        p1 = random.choice(parents)
        p2 = random.choice(parents)
        child_nn = NeuralNetwork(Enemy.NUM_INPUTS, HIDDEN_NODES_ENEMY, Enemy.NUM_OUTPUTS)

        # CHANGED: Use the same single-point crossover for consistency
        child_weights = crossover(p1.nn.get_weights(), p2.nn.get_weights())
        child_nn.set_weights(child_weights)
        child_nn.mutate(MUTATION_RATE, MUTATION_STRENGTH) # Use same mutation params? Maybe tune separately?

        new_enemy = Enemy(random.randint(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING),
                          random.randint(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING),
                          child_nn)
        next_gen_enemies.append(new_enemy)

    return next_gen_enemies


# --- Game Setup ---
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Emergent AI - Apple Collectors')
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24) # General font for UI text

# Initialize objects
score_zone = ScoreZone(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)

# Initialize spatial grid for collision detection
class SpatialGrid:
    def __init__(self, cell_size, width, height):
        self.cell_size = cell_size
        self.width = width
        self.height = height
        self.grid = {}

    def add(self, obj):
        cell = self._get_cell(obj.pos)
        if cell not in self.grid:
            self.grid[cell] = []
        self.grid[cell].append(obj)

    def get_nearby(self, pos, radius):
        nearby = []
        cell = self._get_cell(pos)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                neighbor_cell = (cell[0] + dx, cell[1] + dy)
                if neighbor_cell in self.grid:
                    nearby.extend(self.grid[neighbor_cell])
        return [obj for obj in nearby if np.linalg.norm(obj.pos - pos) <= radius]

    def clear(self):
        self.grid.clear()

    def _get_cell(self, pos):
        return (int(pos[0] // self.cell_size), int(pos[1] // self.cell_size))

spatial_grid = SpatialGrid(cell_size=50, width=SCREEN_WIDTH, height=SCREEN_HEIGHT)
apples = [Apple(random.randint(WORLD_PADDING + APPLE_SIZE, SCREEN_WIDTH - WORLD_PADDING - APPLE_SIZE),
                random.randint(WORLD_PADDING + APPLE_SIZE, SCREEN_HEIGHT - WORLD_PADDING - APPLE_SIZE))
          for _ in range(MAX_APPLES)]

boxes = [Box(random.randint(WORLD_PADDING + BOX_SIZE, SCREEN_WIDTH - WORLD_PADDING - BOX_SIZE),
             random.randint(WORLD_PADDING + BOX_SIZE, SCREEN_HEIGHT - WORLD_PADDING - BOX_SIZE))
         for _ in range(NUM_BOXES)]

# Population Initialization
population = []
loaded_creature_nn = None # Placeholder if we load a specific NN

# --- State Loading ---
start_generation = 1
all_time_best_creature_obj = None # Stores the actual best creature object
all_time_best_fitness = -float('inf')

if os.path.exists("simulation_state.save"):
    loaded_gen, loaded_best_obj, loaded_best_fitness = load_simulation_state()
    start_generation = loaded_gen + 1
    all_time_best_creature_obj = loaded_best_obj # Might be None if load failed
    all_time_best_fitness = loaded_best_fitness
    print(f"Attempting to resume from Generation {start_generation}")
    if all_time_best_creature_obj and hasattr(all_time_best_creature_obj, 'nn'):
         loaded_creature_nn = all_time_best_creature_obj.nn # Get the NN to seed population
         print("Using loaded best creature's NN to seed initial population.")
    else:
         print("Could not load best creature object from state, using random NNs.")

# Initialize Creature Population
for i in range(POPULATION_SIZE):
    # Use loaded NN for the first creature if available, otherwise random
    init_nn = None
    if i == 0 and loaded_creature_nn:
        # Create a new NN instance and copy weights to avoid sharing object
        init_nn = NeuralNetwork(Creature.NUM_INPUTS, HIDDEN_NODES_CREATURE, Creature.NUM_OUTPUTS)
        try:
            init_nn.set_weights(loaded_creature_nn.get_weights())
        except Exception as e:
            print(f"Warning: Failed to set weights from loaded NN: {e}. Using random NN.")
            init_nn = None # Fallback to random if weights are incompatible

    if init_nn is None: # If no loaded NN or setting weights failed
        init_nn = NeuralNetwork(Creature.NUM_INPUTS, HIDDEN_NODES_CREATURE, Creature.NUM_OUTPUTS)

    creature = Creature(
        random.randint(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING),
        random.randint(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING),
        init_nn
    )
    population.append(creature)


# Initialize Enemy Population
enemy_population = [Enemy(random.randint(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING),
                          random.randint(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING))
                    for _ in range(ENEMY_POPULATION_SIZE)]


# --- Main Loop Setup ---
running = True
frame_count = 0
generation = start_generation # Start from loaded or 1
enemy_generation = 1 # Enemy gen counter (could also be saved/loaded)

saving_message_timer = 0
loading_message_timer = 0
MESSAGE_DURATION = 2000 # Milliseconds (2 seconds)


# --- Game Loop ---
while running:
    current_time_ms = pygame.time.get_ticks()

    # --- Event Handling ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # Auto-save on quit (optional, but good practice)
            if all_time_best_creature_obj:
                print("Auto-saving state on quit...")
                save_best_creature(all_time_best_creature_obj) # Save the best NN separately
                save_simulation_state(generation, all_time_best_creature_obj, all_time_best_fitness)
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s: # Save State
                if all_time_best_creature_obj:
                    save_best_creature(all_time_best_creature_obj)
                    save_simulation_state(generation, all_time_best_creature_obj, all_time_best_fitness)
                    saving_message_timer = current_time_ms # Start timer for message
                else:
                    print("No 'all time best' creature recorded yet to save.")
            elif event.key == pygame.K_l: # Load Best Creature NN into population
                loaded_creature = load_best_creature()
                if loaded_creature:
                    # Strategy: Replace a random creature (or the worst one)
                    if population:
                        replace_index = random.randrange(len(population))
                        population[replace_index] = loaded_creature
                        print(f"Loaded best creature NN into population at index {replace_index}.")
                    else:
                        population.append(loaded_creature) # Add if population empty
                    loading_message_timer = current_time_ms # Start timer for message
                else:
                    print("Failed to load creature NN.")

    # --- Update Logic ---
    frame_count += 1
    dt = 1.0 # Assume constant time step for simplicity, could use clock.tick for variable dt

    # Respawn apples
    # CHANGED: More robust respawn logic, higher chance when fewer apples
    respawn_attempts = MAX_APPLES - len(apples)
    if respawn_attempts > 0:
         # Chance increases based on how many apples are missing
         respawn_chance = APPLE_RESPAWN_RATE * respawn_attempts * dt
         if random.random() < respawn_chance:
             # Ensure apples don't spawn inside padding or other objects (basic check)
             valid_spawn = False
             for _ in range(10): # Try a few times to find a good spot
                 rx = random.randint(WORLD_PADDING + APPLE_SIZE, SCREEN_WIDTH - WORLD_PADDING - APPLE_SIZE)
                 ry = random.randint(WORLD_PADDING + APPLE_SIZE, SCREEN_HEIGHT - WORLD_PADDING - APPLE_SIZE)
                 # Basic check: avoid spawning exactly on score zone center
                 if np.linalg.norm(np.array([rx, ry]) - score_zone.pos) > SCORE_ZONE_SIZE:
                     apples.append(Apple(rx, ry))
                     valid_spawn = True
                     break
             # if not valid_spawn: print("Could not find valid spot for new apple")


    # Update Creatures
    # Iterate backwards for safe removal if a creature dies
    for i in range(len(population) - 1, -1, -1):
        creature = population[i]
        alive = creature.update(apples, score_zone, enemy_population, boxes, dt)
        if not alive:
            population.pop(i) # Remove dead creature

    # Update Enemies
    # Iterate backwards for safe removal
    for i in range(len(enemy_population) - 1, -1, -1):
        enemy = enemy_population[i]
        alive = enemy.update(population, boxes, dt)
        if not alive:
            enemy_population.pop(i) # Remove dead enemy

    # Update Boxes (simple physics)
    for box in boxes:
        box.update() # Includes friction and boundary checks

    # Check for Generation End
    current_best_fitness_gen = -float('inf')
    best_creature_gen = None
    if population:
        current_best_fitness_gen = max(c.fitness for c in population)
        if current_best_fitness_gen > all_time_best_fitness:
             all_time_best_fitness = current_best_fitness_gen
             # Find the actual creature object with the best fitness this generation
             all_time_best_creature_obj = max(population, key=lambda c: c.fitness)
             # Note: This assigns the *reference*. If you need a snapshot, use copy.deepcopy

    # Generation Turnover Condition
    if len(population) == 0 or frame_count >= GENERATION_TIME:
        # --- End of Generation Processing ---
        print("-" * 30)
        print(f"--- Generation {generation} Complete ---")
        print(f"    Time Elapsed: {frame_count} frames")

        if len(population) > 0:
            # Calculate stats for the completed generation
            total_fitness = sum(c.fitness for c in population)
            avg_fitness = total_fitness / len(population)
            best_fitness_this_gen = max(c.fitness for c in population) # Should match current_best_fitness_gen
            total_deposited = sum(c.apples_deposited_total for c in population)
            avg_lifetime = sum(c.lifetime for c in population) / len(population)

            # Use f-strings for printing stats
            print(f"    Creatures remaining: {len(population)}")
            print(f"    Best fitness this gen: {best_fitness_this_gen:.2f}")
            print(f"    Avg fitness this gen: {avg_fitness:.2f}")
            print(f"    Total apples deposited this gen: {total_deposited}")
            print(f"    Avg lifetime this gen: {avg_lifetime:.2f} frames")
            print(f"    All-time best fitness: {all_time_best_fitness:.2f}")


            # --- Creature Evolution ---
            print("    Running Genetic Algorithm for Creatures...")
            parents = selection(population) # Select parents (includes elites)
            population = reproduction(parents, generation) # Create new generation

        else:
            # Handle extinction
            print("    Population Extinct! Restarting with random creatures.")
            population = []
            for _ in range(POPULATION_SIZE):
                nn = NeuralNetwork(Creature.NUM_INPUTS, HIDDEN_NODES_CREATURE, Creature.NUM_OUTPUTS)
                population.append(Creature(random.randint(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING),
                                            random.randint(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING), nn))
            # Reset all-time best fitness if population died out? Optional.
            # all_time_best_fitness = -float('inf')
            # all_time_best_creature_obj = None

        # --- Enemy Evolution ---
        print("    Running Genetic Algorithm for Enemies...")
        if len(enemy_population) > 0:
            enemy_population = evolve_enemies(enemy_population)
            print(f"    Evolved {len(enemy_population)} enemies.")
        else:
            print("    Enemy population died out. Restarting with random enemies.")
            enemy_population = [Enemy(random.randint(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING),
                                      random.randint(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING))
                                for _ in range(ENEMY_POPULATION_SIZE)]

        # --- Reset for Next Generation ---
        frame_count = 0
        generation += 1
        enemy_generation += 1 # Increment enemy gen counter too
        # Reset apples (optional, maybe keep some?)
        apples = [Apple(random.randint(WORLD_PADDING + APPLE_SIZE, SCREEN_WIDTH - WORLD_PADDING - APPLE_SIZE),
                        random.randint(WORLD_PADDING + APPLE_SIZE, SCREEN_HEIGHT - WORLD_PADDING - APPLE_SIZE))
                  for _ in range(MAX_APPLES)]
        print("-" * 30)
        print(f"--- Starting Generation {generation} ---")


    # --- Drawing ---
    screen.fill((30, 30, 30)) # Dark background

    # Draw world elements (order matters for overlap)
    score_zone.draw(screen)
    for apple in apples:
        apple.draw(screen)
    for box in boxes:
        box.draw(screen) # Draw boxes before creatures/enemies

    # Draw Creatures
    for creature in population:
        creature.draw(screen)

    # Draw Enemies
    for enemy in enemy_population:
        enemy.draw(screen)

    # Draw World Boundaries (semi-transparent overlay)
    draw_world_boundaries(screen)


    # --- UI Text & Info ---
    # CHANGED: Use f-strings
    gen_text = font.render(f"Generation: {generation}", True, (255, 255, 255))
    pop_text = font.render(f"Creatures: {len(population)}", True, (255, 255, 255))
    time_text = font.render(f"Time: {frame_count} / {GENERATION_TIME}", True, (255, 255, 255))
    apple_text = font.render(f"Apples: {len(apples)} / {MAX_APPLES}", True, (255, 255, 255))
    enemy_text = font.render(f"Enemies: {len(enemy_population)} (Gen {enemy_generation})", True, (255, 100, 100)) # Enemy color

    best_fitness_text = font.render(f"Best Fitness (All Time): {all_time_best_fitness:.2f}", True, (200, 200, 0))


    screen.blit(gen_text, (10, 10))
    screen.blit(pop_text, (10, 30))
    screen.blit(time_text, (10, 50))
    screen.blit(apple_text, (10, 70))
    screen.blit(enemy_text, (10, 90))
    screen.blit(best_fitness_text, (10, 110))


    # Display Save/Load messages
    if saving_message_timer > 0 and current_time_ms - saving_message_timer < MESSAGE_DURATION:
        save_text = font.render("State Saved!", True, (0, 255, 0))
        screen.blit(save_text, (SCREEN_WIDTH - save_text.get_width() - 10, SCREEN_HEIGHT - 30))
    else:
        saving_message_timer = 0 # Reset timer

    if loading_message_timer > 0 and current_time_ms - loading_message_timer < MESSAGE_DURATION:
        load_text = font.render("NN Loaded!", True, (0, 255, 0))
        screen.blit(load_text, (SCREEN_WIDTH - load_text.get_width() - 10, SCREEN_HEIGHT - 50))
    else:
        loading_message_timer = 0 # Reset timer


    save_load_prompt = font.render("S: Save State | L: Load Best NN", True, (200, 200, 200))
    screen.blit(save_load_prompt, (10, SCREEN_HEIGHT - 30))


    # --- Final Display Update ---
    pygame.display.flip()

    # --- Frame Rate Control ---
    clock.tick(60) # Target 60 FPS

# --- Cleanup ---
pygame.quit()
print("Simulation finished.")