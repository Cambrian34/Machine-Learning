import pygame
import random
import math
import time

# Pygame initialization
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
FPS = 60
NUM_AGENTS = 30
NUM_FOOD = 30
NUM_BOXES = 5
TOURNAMENT_SIZE = 5  # Number of agents in each tournament

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BROWN = (139, 69, 19)

# Agent Parameters
AGENT_SIZE = 10
AGENT_SPEED = 1
ENERGY_COST = 1
ENERGY_GAIN = 10

# Food Parameters
FOOD_SIZE = 5

# Box Parameters
BOX_SIZE = 20

# Display Setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Self-Modifying Agents with Interactive Boxes")
clock = pygame.time.Clock()

# Agent class
class Agent:
    def __init__(self, x, y, behavior_code):
        self.x = x
        self.y = y
        self.energy = 100
        self.food_collected = 0
        self.behavior_code = behavior_code  # Behavior as simple code

    def mutate_behavior(self):
        """ Mutate agent's behavior randomly. """
        mutation_point = random.choice(self.behavior_code)
        if mutation_point == "move_toward_food":
            self.behavior_code.append("move_randomly")
        elif mutation_point == "move_randomly":
            self.behavior_code.append("move_toward_predator")
        elif mutation_point == "move_toward_box":
            self.behavior_code.append("push_box")
        
    def act(self, food_positions, boxes):
        """ Act based on current behavior. """
        if "move_toward_food" in self.behavior_code:
            closest_food = self.get_closest_food(food_positions)
            if closest_food:
                self.move_towards(closest_food)
                self.food_collected += 1
                self.energy += ENERGY_GAIN
                if closest_food in food_positions:
                    food_positions.remove(closest_food)  # Remove the collected food
            self.energy -= ENERGY_COST

        if "move_randomly" in self.behavior_code:
            self.move_randomly()
            self.energy -= ENERGY_COST

        if "push_box" in self.behavior_code:
            self.push_box(boxes)

        if self.energy <= 0:
            self.die()

    def get_closest_food(self, food_positions):
        """ Find the closest food. """
        closest_food = None
        min_dist = float('inf')
        for food in food_positions:
            dist = math.sqrt((food[0] - self.x) ** 2 + (food[1] - self.y) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_food = food
        return closest_food

    def move_towards(self, target_pos):
        """ Move towards the target position (food). """
        target_x, target_y = target_pos
        dx = target_x - self.x
        dy = target_y - self.y
        distance = math.sqrt(dx ** 2 + dy ** 2)
        if distance == 0:
            return
        dx /= distance
        dy /= distance
        self.x += dx * AGENT_SPEED
        self.y += dy * AGENT_SPEED
    
    def move_randomly(self):
        """ Move randomly. """
        self.x += random.choice([-1, 1]) * AGENT_SPEED
        self.y += random.choice([-1, 1]) * AGENT_SPEED
        
        # Keep the agent within bounds
        self.x = max(0, min(WIDTH, self.x))
        self.y = max(0, min(HEIGHT, self.y))

    def push_box(self, boxes):
        """ Push boxes when colliding with them. """
        for box in boxes:
            dist = math.sqrt((box.x - self.x) ** 2 + (box.y - self.y) ** 2)
            if dist < BOX_SIZE + AGENT_SIZE:  # If close to a box
                # Move the box by a small amount in the direction of the agent
                angle = math.atan2(box.y - self.y, box.x - self.x)
                box.x += math.cos(angle) * AGENT_SPEED
                box.y += math.sin(angle) * AGENT_SPEED

    def die(self):
        """ Agent dies when energy reaches zero. """
        self.energy = 0

    def reproduce(self):
        """ Create a new agent with mutated behavior. """
        new_behavior = self.behavior_code[:]
        child = Agent(self.x, self.y, new_behavior)
        child.mutate_behavior()
        return child

# Box class
class Box:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Tournament Selection
def tournament_selection(agents):
    """ Select an agent based on tournament selection. """
    selected_agents = random.sample(agents, TOURNAMENT_SIZE)
    best_agent = max(selected_agents, key=lambda agent: agent.food_collected)
    return best_agent

# Main Simulation Loop
def main():
    food_positions = [(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(NUM_FOOD)]
    boxes = [Box(random.randint(100, WIDTH-100), random.randint(100, HEIGHT-100)) for _ in range(NUM_BOXES)]
    agents = [Agent(random.randint(0, WIDTH), random.randint(0, HEIGHT), ["move_toward_food", "move_randomly"]) for _ in range(NUM_AGENTS)]
    
    generation = 1
    last_generation_time = time.time()

    while True:
        screen.fill(WHITE)

        total_food_collected = sum(agent.food_collected for agent in agents)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Act agents
        for agent in agents:
            agent.act(food_positions, boxes)

        #spawn 10 food every 10 seconds
        if time.time() - last_generation_time >= 10:
            food_positions.extend([(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(10)])
        # Remove food that is collected
        food_positions = [food for food in food_positions if food not in [(agent.x, agent.y) for agent in agents]]
        
        # Remove dead agents
        
        
        # Render agents
        for agent in agents:
            pygame.draw.circle(screen, BLUE, (int(agent.x), int(agent.y)), AGENT_SIZE)

        # Render food
        for food in food_positions:
            pygame.draw.circle(screen, GREEN, food, FOOD_SIZE)

        # Render boxes
        for box in boxes:
            pygame.draw.rect(screen, BROWN, (box.x - BOX_SIZE//2, box.y - BOX_SIZE//2, BOX_SIZE, BOX_SIZE))

        # Display the total food collected
        font = pygame.font.SysFont("Arial", 20)
        food_text = font.render(f"Total Food Collected: {total_food_collected}", True, (0, 0, 0))
        screen.blit(food_text, (10, 10))

        # Check if all agents are dead or 10 seconds have passed
        if all(agent.energy <= 0 for agent in agents) or time.time() - last_generation_time >= 10:
            # Tournament selection for reproduction
            new_agents = []
            for _ in range(NUM_AGENTS):
                parent = tournament_selection(agents)
                child = parent.reproduce()
                new_agents.append(child)

            agents = new_agents
            last_generation_time = time.time()
            print(f"Generation {generation} ended, starting new generation...")
            generation += 1
        
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()