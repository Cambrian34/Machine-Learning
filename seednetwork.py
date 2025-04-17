from anyio import current_time
import pygame
import numpy as np
import random
import math
import json
import pickle
import base64
import os
import time
# Initialize Pygame
pygame.init()

# --- Constants ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
APPLE_SIZE = 10  # Smaller apples
CREATURE_SIZE = 8 # Smaller creatures
SCORE_ZONE_SIZE = 50
WORLD_PADDING = 20

BOX_SIZE = 20
NUM_BOXES = 10

STATS_HISTORY_LENGTH = 100  # Keep last 100 generations of stats


# Simulation Parameters
POPULATION_SIZE = 50
ENEMY_POPULATION_SIZE = 20
ENEMY_GENERATION_TIME = 3000  # Define the number of frames per enemy generation
MAX_APPLES = 30
APPLE_RESPAWN_RATE = 0.5 # Chance per frame an apple respawns if below max
STARTING_ENERGY = 100.0
ENERGY_DECAY_RATE = 0.1 # Energy lost per frame just existing
MOVE_ENERGY_COST = 0.1  # Extra energy cost for moving
GATHER_ENERGY_BONUS = 30.0 # Energy gained per apple gathered
MAX_ENERGY = 200.0
REST_ENERGY_REGAIN = 0.09

# Genetic Algorithm Parameters
GENERATION_TIME = 3000 # Frames per generation 
MUTATION_RATE = 0.1  # Probability of a weight mutating
MUTATION_STRENGTH = 0.5 # How much a weight can change during mutation
ELITISM_COUNT = 2 # Keep the best N creatures without mutation


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with small random values
        self.weights_ih = np.random.randn(input_size, hidden_size) * 0.1
        self.weights_ho = np.random.randn(hidden_size, output_size) * 0.1
        # Simple biases (can be evolved too, but starting simple)
        self.bias_h = np.zeros((1, hidden_size))
        self.bias_o = np.zeros((1, output_size))

    def predict(self, inputs):
        # Ensure inputs is a 2D array (1 sample, N features)
        inputs = np.array(inputs).reshape(1, -1)

        # Input to Hidden layer
        hidden_raw = np.dot(inputs, self.weights_ih) + self.bias_h
        hidden_activated = np.tanh(hidden_raw) # Use tanh activation function (-1 to 1)

        # Hidden to Output layer
        output_raw = np.dot(hidden_activated, self.weights_ho) + self.bias_o
        output_activated = np.tanh(output_raw) # Use tanh activation function (-1 to 1)

        return output_activated[0] # Return the 1D array of outputs

    def get_weights(self):
        # Return a flat list of all weights for mutation/crossover
        return np.concatenate([self.weights_ih.flatten(), self.weights_ho.flatten()])

    def set_weights(self, flat_weights):
        # Set weights from a flat list
        ih_size = self.weights_ih.size
        ho_size = self.weights_ho.size

        if len(flat_weights) != ih_size + ho_size:
             raise ValueError("Incorrect number of weights provided")

        self.weights_ih = flat_weights[:ih_size].reshape(self.weights_ih.shape)
        self.weights_ho = flat_weights[ih_size:].reshape(self.weights_ho.shape)

    def mutate(self, rate, strength):
         # Mutate weights slightly
         weights = self.get_weights()
         for i in range(len(weights)):
             if random.random() < rate:
                 weights[i] += np.random.randn() * strength
         self.set_weights(weights)


# --- Helper Functions ---
def create_seed_network(input_size=7, hidden_size=8, output_size=4):
    seed_nn = NeuralNetwork(input_size, hidden_size, output_size)

    # Zero everything out first
    seed_nn.weights_ih[:] = 0.0
    seed_nn.weights_ho[:] = 0.0
    seed_nn.bias_h[:] = 0.0
    seed_nn.bias_o[:] = 0.0

    # Map input indexes
    APPLE_IN_SIGHT = 0
    DIR_X = 1
    DIR_Y = 2
    ENERGY = 3
    AGE = 4
    LAST_MOVE_X = 5
    LAST_MOVE_Y = 6

    # Map output indexes
    MOVE_X = 0
    MOVE_Y = 1
    GATHER = 2
    REPRODUCE = 3

    # --- Connect inputs to hidden layer ---
    # Forward apple direction to hidden layer
    seed_nn.weights_ih[DIR_X, 0] = 1.0
    seed_nn.weights_ih[DIR_Y, 1] = 1.0
    seed_nn.weights_ih[APPLE_IN_SIGHT, 2] = 1.0
    seed_nn.weights_ih[ENERGY, 3] = 0.5

    # --- Connect hidden to output layer ---
    # Output move_x and move_y based on direction
    seed_nn.weights_ho[0, MOVE_X] = 1.0  # hidden[0] = DIR_X → MOVE_X
    seed_nn.weights_ho[1, MOVE_Y] = 1.0  # hidden[1] = DIR_Y → MOVE_Y
    seed_nn.weights_ho[2, GATHER] = 1.0  # hidden[2] = APPLE_IN_SIGHT → GATHER
    seed_nn.weights_ho[3, REPRODUCE] = 0.5  # hidden[3] = ENERGY → REPRODUCE

    return seed_nn

class Creature:
    
    NUM_INPUTS = 8
    NUM_OUTPUTS = 4 
    
    # INPUTS indices (for the neural network)
    IDX_DIST_APPLE = 0
    IDX_ANGLE_APPLE = 1
    IDX_DIST_ZONE = 2
    IDX_ANGLE_ZONE = 3
    IDX_ENERGY = 4
    IDX_DIST_ENEMY = 5  # NEW
    IDX_ANGLE_ENEMY = 6  # NEW
    IDX_APPLES_HELD = 7  # NEW


    # OUTPUTS indices (for the neural network)
    IDX_TURN = 0
    IDX_ACCELERATE = 1
    IDX_GATHER = 2
    IDX_DEPOSIT = 3
    IDX_REST = 4
    IDX_FLEE = 5

    def __init__(self, x, y, nn):
        self.pos = np.array([x, y], dtype=np.float64)
        self.velocity = np.array([0.0, 0.0], dtype=np.float64)
        self.angle = random.uniform(0, 2 * math.pi)
        self.speed = 0.0
        self.max_speed = 3.0
        self.max_turn_rate = 0.2

        self.energy = STARTING_ENERGY
        self.apples_held = 0
        self.max_apples = 5
        self.nn = nn
        self.fitness = 0.0
        self.apples_deposited_total = 0
        self.lifetime = 0  # Track how long the creature has lived

        self.sight_radius = 190

    def get_nearest_object(self, objects):
        nearest_obj = None
        min_dist_sq = self.sight_radius**2

        for obj in objects:
            direction = obj.pos - self.pos
            dist_sq = np.dot(direction, direction)

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest_obj = obj

        if nearest_obj:
            dist = math.sqrt(min_dist_sq)
            direction_vec = nearest_obj.pos - self.pos
            target_angle = math.atan2(direction_vec[1], direction_vec[0])
            relative_angle = target_angle - self.angle
            relative_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi
            return nearest_obj, dist, relative_angle / math.pi
        else:
            return None, self.sight_radius, 0.0

    def get_inputs(self, apples, score_zone,enemies):
        inputs = np.zeros(Creature.NUM_INPUTS)

        # Nearest Apple Info
        nearest_apple, dist_apple, angle_apple = self.get_nearest_object(apples)
        inputs[Creature.IDX_DIST_APPLE] = (self.sight_radius - dist_apple) / self.sight_radius
        inputs[Creature.IDX_ANGLE_APPLE] = angle_apple
        inputs[Creature.IDX_APPLES_HELD] = self.apples_held / self.max_apples  # Normalized
        # Nearest Enemy Info
        nearest_enemy, dist_enemy, angle_enemy = self.get_nearest_object(enemies)
        if nearest_enemy:
            inputs[Creature.IDX_DIST_ENEMY] = (self.sight_radius - dist_enemy) / self.sight_radius
            inputs[Creature.IDX_ANGLE_ENEMY] = angle_enemy

        


        # Score Zone Info
        zone_dir = score_zone.pos - self.pos
        dist_zone = np.linalg.norm(zone_dir)
        zone_target_angle = math.atan2(zone_dir[1], zone_dir[0])
        zone_relative_angle = zone_target_angle - self.angle
        zone_relative_angle = (zone_relative_angle + math.pi) % (2 * math.pi) - math.pi

        inputs[Creature.IDX_DIST_ZONE] = max(0, (SCREEN_WIDTH - dist_zone) / SCREEN_WIDTH)
        inputs[Creature.IDX_ANGLE_ZONE] = zone_relative_angle / math.pi

        # Energy Level
        inputs[Creature.IDX_ENERGY] = self.energy / MAX_ENERGY

        #inputs[Creature.IDX_APPLES_HELD] = self.apples_held / self.max_apples  # Normalized


        return inputs

    def update(self, apples, score_zone,enemies,boxes, dt=1.0):
        inputs = self.get_inputs(apples, score_zone,enemies)
        actions = self.nn.predict(inputs)

        turn_request = actions[Creature.IDX_TURN]
        accel_request = actions[Creature.IDX_ACCELERATE]
        gather_request = actions[Creature.IDX_GATHER]
        deposit_request = actions[Creature.IDX_DEPOSIT]
        """
        if gather_request > 0.5:  # Adjust the gather condition to be more realistic
            nearby_apples = [a for a in apples 
                        if np.linalg.norm(self.pos - a.pos) < (CREATURE_SIZE + APPLE_SIZE)*2]
            for apple in nearby_apples:
                dist_sq = np.dot(self.pos - apple.pos, self.pos - apple.pos)
                if dist_sq < (CREATURE_SIZE + APPLE_SIZE)**2 and apple in apples:
                    apples.remove(apple)
                    self.apples_held += 1
                    self.energy += GATHER_ENERGY_BONUS
                    self.energy = min(self.energy, MAX_ENERGY)
                    if self.apples_held >= self.max_apples:
                        break
        """  
        #nearest_enemy, dist_enemy, _ = self.get_nearest_object(enemies)

        # Let the behavior emerge naturally from the neural network's outputs
        # No explicit fleeing behavior is implemented here
        
        # Handle box collisions
        for box in boxes:
            dx = box.pos[0] - self.pos[0]
            dy = box.pos[1] - self.pos[1]
            distance = math.sqrt(dx**2 + dy**2)
            min_dist = CREATURE_SIZE + BOX_SIZE/2
            
            if distance < min_dist:
                # Push box away
                push_force = 0.5
                angle = math.atan2(dy, dx)
                box.velocity[0] += math.cos(angle) * push_force
                box.velocity[1] += math.sin(angle) * push_force
                
                # Adjust creature position
                overlap = min_dist - distance
                self.pos[0] -= math.cos(angle) * overlap/2
                self.pos[1] -= math.sin(angle) * overlap/2

        if gather_request > 0.2:  # Only gather if the neural network requests it
            for apple in apples[:]:  # Iterate over a copy to allow removal
                distance = np.linalg.norm(self.pos - apple.pos)
                if distance < (CREATURE_SIZE + APPLE_SIZE) * 1.2:  # Slightly larger than sum of radii
                    apples.remove(apple)
                    self.apples_held = min(self.apples_held + 1, self.max_apples)
                    self.energy = min(self.energy + GATHER_ENERGY_BONUS, MAX_ENERGY)
                    break

        if deposit_request > 0.5 and self.apples_held > 0:
            dist_sq_zone = np.dot(self.pos - score_zone.pos, self.pos - score_zone.pos)
            if dist_sq_zone < (SCORE_ZONE_SIZE + CREATURE_SIZE)**2:
                self.fitness += self.apples_held * 10
                self.apples_deposited_total += self.apples_held
                self.apples_held = 0

        # Energy management

        # Movement and energy
        self.energy -= ENERGY_DECAY_RATE * dt
        if self.energy > 0:
            self.angle += turn_request * self.max_turn_rate
            self.angle %= (2 * math.pi)
    
            if accel_request > 0:
                self.speed = min(self.speed + accel_request * 0.2, self.max_speed)
            else:
                self.speed *= 0.95
    
            self.speed = max(0, self.speed)
    
            self.velocity[0] = math.cos(self.angle) * self.speed
            self.velocity[1] = math.sin(self.angle) * self.speed
            
            new_pos = self.pos + self.velocity * dt
            
            # Check and handle boundary collisions
            bounced = False
            if new_pos[0] < WORLD_PADDING:
                new_pos[0] = WORLD_PADDING
                self.angle = math.pi - self.angle
                bounced = True
            elif new_pos[0] > SCREEN_WIDTH - WORLD_PADDING:
                new_pos[0] = SCREEN_WIDTH - WORLD_PADDING
                self.angle = math.pi - self.angle
                bounced = True
            
            if new_pos[1] < WORLD_PADDING:
                new_pos[1] = WORLD_PADDING
                self.angle = -self.angle
                bounced = True
            elif new_pos[1] > SCREEN_HEIGHT - WORLD_PADDING:
                new_pos[1] = SCREEN_HEIGHT - WORLD_PADDING
                self.angle = -self.angle
                bounced = True
            
            if bounced:
                self.speed *= 0.8  # Lose some energy when bouncing
                self.velocity[0] = math.cos(self.angle) * self.speed
                self.velocity[1] = math.sin(self.angle) * self.speed
            
            self.pos = new_pos
    
        self.lifetime += dt
        return self.energy > 0
    

    def draw(self, screen):
        #pygame.draw.circle(screen, (0, 150, 0), self.pos.astype(int), CREATURE_SIZE)

        end_line = self.pos + np.array([math.cos(self.angle), math.sin(self.angle)]) * (CREATURE_SIZE + 5)
        pygame.draw.line(screen, (0, 255, 0), self.pos.astype(int), end_line.astype(int), 2)

        color = (0, min(255, 100 + self.apples_deposited_total * 20), 0)

        energy_ratio = max(0, self.energy / MAX_ENERGY)
        energy_bar_width = int(CREATURE_SIZE * 2 * energy_ratio)
        energy_bar_pos = self.pos - np.array([CREATURE_SIZE, CREATURE_SIZE + 4])
        pygame.draw.rect(screen, (255, 0, 0), (*energy_bar_pos.astype(int), CREATURE_SIZE * 2, 3))
        pygame.draw.rect(screen, (0, 255, 0), (*energy_bar_pos.astype(int), energy_bar_width, 3))
        pygame.draw.circle(screen, color, self.pos.astype(int), CREATURE_SIZE)

        save_text = font.render("Press S to save, L to load", True, (200, 200, 200))
        screen.blit(save_text, (SCREEN_WIDTH - 250, SCREEN_HEIGHT - 30))


        if self.apples_held > 0:
            if not hasattr(self, 'font'):
                self.font = pygame.font.SysFont(None, 18)
            text = self.font.render(str(self.apples_held), True, (255, 255, 255))
            screen.blit(text, (self.pos[0] - 4, self.pos[1] - 6))


class Apple:
    def __init__(self, x, y):
        self.pos = np.array([x, y], dtype=np.float64)

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 0, 0), self.pos.astype(int), APPLE_SIZE)


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


# --- ScoreZone Class ---
class ScoreZone:
    def __init__(self, x, y):
        self.pos = np.array([x, y], dtype=np.float64)

    def draw(self, screen):
        pygame.draw.circle(screen, (0, 0, 255, 100), self.pos.astype(int), SCORE_ZONE_SIZE)  # Make semi-transparent


# --- Main Simulation Setup ---
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Emergent AI - Apple Collectors')
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# Initialize objects
score_zone = ScoreZone(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
apples = [Apple(random.randint(APPLE_SIZE, SCREEN_WIDTH - APPLE_SIZE),
                random.randint(APPLE_SIZE, SCREEN_HEIGHT - APPLE_SIZE)) for _ in range(MAX_APPLES)]

# population initialization:
population = []
seed_nn = create_seed_network()  # Create a seed network for the creatures
seed_creature = Creature(
    random.randint(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING),
    random.randint(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING),
    seed_nn
)
population.append(seed_creature)

# Initialize boxes and other game objects
boxes = [Box(random.randint(50, SCREEN_WIDTH-50),
             random.randint(50, SCREEN_HEIGHT-50)) for _ in range(NUM_BOXES)]

# --- Game Loop ---
running = True
frame_count = 0
generation = 1

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- Update Logic ---
    frame_count += 1

    # Respawn apples
    if len(apples) < MAX_APPLES and random.random() < APPLE_RESPAWN_RATE * (MAX_APPLES - len(apples)):
        apples.append(Apple(random.randint(APPLE_SIZE, SCREEN_WIDTH - APPLE_SIZE),
                            random.randint(APPLE_SIZE, SCREEN_HEIGHT - APPLE_SIZE)))

    # Update creatures
    next_population_indices = list(range(len(population)))  # Indices of creatures alive
    for i in range(len(population)):
        creature = population[i]
        creature.update(apples, score_zone, [], boxes)  # Pass empty list for enemies

        # Creature draws itself
        creature.draw(screen)

        if creature.energy <= 0:
            next_population_indices.remove(i)

    # Create new generation based on remaining alive creatures
    if frame_count > GENERATION_TIME:
        population = [population[i] for i in next_population_indices]
        # Reset population if too few survived or generation finished
        if len(population) < POPULATION_SIZE // 2:
            population.append(seed_creature)  # Add back some seed creatures

        # Mutate for next generation
        for i in range(len(population)):
            population[i].nn.mutate(MUTATION_RATE, MUTATION_STRENGTH)
        frame_count = 0  # Reset frame count for new generation

    # --- Drawing ---
    screen.fill((0, 0, 0))  # Black background
    score_zone.draw(screen)

    for apple in apples:
        apple.draw(screen)

    for box in boxes:
        box.update()
        box.draw(screen)

    pygame.display.flip()
    clock.tick(30)

pygame.quit()