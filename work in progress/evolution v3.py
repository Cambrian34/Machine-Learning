import pygame
import random
import math

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
AGENT_RADIUS = 15
FOOD_RADIUS = 5
AGENT_SPEED = 4
MAX_ENERGY = 100
FOOD_VALUE = 30

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Screen setup
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Self-Modifying Agents Simulation")

# Agent class
class Agent:
    def __init__(self, x, y, behavior_code=None):
        self.x = x
        self.y = y
        self.energy = MAX_ENERGY
        self.behavior_code = behavior_code if behavior_code else ["move_toward_food"]
    
    def move_toward_food(self, food):
        """Move towards the nearest food."""
        target_x, target_y = food
        angle = math.atan2(target_y - self.y, target_x - self.x)
        self.x += AGENT_SPEED * math.cos(angle)
        self.y += AGENT_SPEED * math.sin(angle)
    
    def random_walk(self):
        """Move randomly."""
        angle = random.uniform(0, 2 * math.pi)
        self.x += AGENT_SPEED * math.cos(angle)
        self.y += AGENT_SPEED * math.sin(angle)
    
    def act(self, food_position):
        """Agent behavior: move towards food or randomly move."""
        if "move_toward_food" in self.behavior_code:
            self.move_toward_food(food_position)
        elif "random_walk" in self.behavior_code:
            self.random_walk()
        self.energy -= 0.5  # Decay energy with every action
    
    def collect_food(self, food_position):
        """Check if agent is close enough to collect food."""
        distance = math.sqrt((self.x - food_position[0])**2 + (self.y - food_position[1])**2)
        if distance < FOOD_RADIUS + AGENT_RADIUS:
            self.energy = min(MAX_ENERGY, self.energy + FOOD_VALUE)  # Collect food and gain energy

    def mutate_behavior(self):
        """Randomly mutate the agent's behavior."""
        if random.random() < 0.1:  # Mutation chance
            if random.random() < 0.5:
                self.behavior_code.append("random_walk")
            else:
                self.behavior_code.append("move_toward_food")
    
    def is_alive(self):
        """Check if the agent is still alive."""
        return self.energy > 0

# Food class
class Food:
    def __init__(self):
        self.x = random.randint(0, SCREEN_WIDTH)
        self.y = random.randint(0, SCREEN_HEIGHT)

# Initialize simulation
agents = [Agent(random.randint(100, 700), random.randint(100, 500)) for _ in range(5)]
food_items = [Food() for _ in range(5)]

# Simulation loop
running = True
clock = pygame.time.Clock()
generation = 0
while running:
    screen.fill(WHITE)

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update agents and food
    for agent in agents[:]:
        if agent.is_alive():
            closest_food = min(food_items, key=lambda f: math.sqrt((agent.x - f.x)**2 + (agent.y - f.y)**2))
            agent.act((closest_food.x, closest_food.y))  # Agent acts based on its behavior
            if math.sqrt((agent.x - closest_food.x)**2 + (agent.y - closest_food.y)**2) < FOOD_RADIUS + AGENT_RADIUS:
                agent.collect_food((closest_food.x, closest_food.y))  # Check if it collects food
                food_items.remove(closest_food)  # Remove consumed food
                food_items.append(Food())  # Add new food at a random position
            agent.mutate_behavior()  # Allow the agent to mutate its behavior
        else:
            agents.remove(agent)  # Remove dead agent

    # Draw food
    for food in food_items:
        pygame.draw.circle(screen, GREEN, (food.x, food.y), FOOD_RADIUS)


    

    # Draw agents
    for agent in agents:
        pygame.draw.circle(screen, BLUE, (int(agent.x), int(agent.y)), AGENT_RADIUS)

    # Draw text: Generation count and agent count
    font = pygame.font.SysFont("Arial", 24)
    text = font.render(f"Generation: {generation} | Agents: {len(agents)}", True, (0, 0, 0))
    screen.blit(text, (10, 10))

    # Check if agents run out of energy
    if len(agents) == 0:
        generation += 1  # Next generation
        agents = [Agent(random.randint(100, 700), random.randint(100, 500)) for _ in range(5)]  # Rebirth with new agents

    # Update display
    pygame.display.flip()

    # Control frame rate
    clock.tick(30)

pygame.quit()