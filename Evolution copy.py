import pygame
import random
import math
import pickle

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)

# Constants
CREATURE_COUNT = 10
FOOD_COUNT = 50
REPRODUCTION_THRESHOLD = 100
MAX_ENERGY = 200
MUTATION_RATE = 0.01
MUTATION_INCREASE_RATE = 0.05
GENOME_LENGTH = 10
FOOD_RADIUS = 10
BEST_CREATURES_COUNT = 5  # Number of best creatures to keep for next generation

# Genes for the stack-based brain
GENES = ["PUSH", "POP", "DUP", "INC", "DEC", "SWAP", "ADD", "SUB", "RAND"]

class GenomeInterpreter:
    def __init__(self, genome):
        self.genome = genome

    def interpret(self, stack=None):
        if stack is None:
            stack = []
        for token in self.genome:
            try:
                if token == 'PUSH':
                    stack.append(random.randint(0, 10))
                elif token == 'POP' and stack:
                    stack.pop()
                elif token == 'DUP' and stack:
                    stack.append(stack[-1])
                elif token == 'INC' and stack:
                    stack[-1] += 1
                elif token == 'DEC' and stack:
                    stack[-1] -= 1
                elif token == 'SWAP' and len(stack) >= 2:
                    stack[-1], stack[-2] = stack[-2], stack[-1]
                elif token == 'ADD' and len(stack) >= 2:
                    b = stack.pop()
                    a = stack.pop()
                    stack.append(a + b)
                elif token == 'SUB' and len(stack) >= 2:
                    b = stack.pop()
                    a = stack.pop()
                    stack.append(a - b)
                elif token == 'RAND':
                    stack.append(random.randint(-5, 5))
            except Exception as e:
                pass
        return stack

class Creature:
    def __init__(self, x, y, genome, memory=None):
        self.x = x
        self.y = y
        self.energy = MAX_ENERGY
        self.fitness = 0
        self.genome = genome
        self.memory = memory if memory else []
        self.direction = random.uniform(0, 2 * math.pi)
        self.alive = True
        self.interpreter = GenomeInterpreter(genome)

    def update(self, food_list):
        if not self.alive:
            return

        # Execute genome
        self.memory = self.interpreter.interpret(self.memory.copy())

        # Movement based on genome results
        speed = 2
        if self.memory:
            direction_change = self.memory[-1] * 0.05 if len(self.memory) > 0 else 0
            self.direction += direction_change

        self.x += math.cos(self.direction) * speed
        self.y += math.sin(self.direction) * speed
        self.x %= 800
        self.y %= 600

        # Energy management
        self.energy -= 1
        if self.energy <= 0:
            self.alive = False

        # Check for food
        for food in food_list[:]:
            if math.hypot(self.x - food[0], self.y - food[1]) < FOOD_RADIUS:
                self.energy = min(self.energy + 30, MAX_ENERGY)
                self.fitness += 10
                food_list.remove(food)
                break

    def reproduce(self):
        if self.fitness >= REPRODUCTION_THRESHOLD and self.energy > MAX_ENERGY / 2:
            self.energy = int(self.energy * 0.5)
            new_genome = self.mutate(self.genome.copy())
            return Creature(self.x, self.y, new_genome, self.memory.copy())
        return None

    def mutate(self, genome):
        if random.random() < MUTATION_RATE:
            idx = random.randint(0, len(genome)-1)
            genome[idx] = random.choice(GENES)
        return genome

def random_genome():
    return [random.choice(GENES) for _ in range(GENOME_LENGTH)]

class SimulationState:
    def __init__(self):
        self.creatures = []
        self.food = []
        self.mutation_rate = MUTATION_RATE
        self.generation = 0

    def save(self, filename):
        state = {
            'creatures': [(c.x, c.y, c.genome, c.memory, c.energy, c.fitness, c.direction)
                          for c in self.creatures if c.alive],
            'food': self.food,
            'mutation_rate': self.mutation_rate,
            'generation': self.generation
        }
        with open(filename, 'wb') as f:
            pickle.dump(state, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        
        self.creatures = [
            Creature(x, y, genome, memory) for x, y, genome, memory, _, _, _ in state['creatures']
        ]
        for c, (_, _, _, _, energy, fitness, direction) in zip(self.creatures, state['creatures']):
            c.energy = energy
            c.fitness = fitness
            c.direction = direction
        
        self.food = state['food']
        self.mutation_rate = state['mutation_rate']
        self.generation = state['generation']

    def get_best_creatures(self):
        """Get the top N best creatures by fitness."""
        sorted_creatures = sorted(self.creatures, key=lambda c: c.fitness, reverse=True)
        return sorted_creatures[:BEST_CREATURES_COUNT]

def spawn_food(foods, existing_food, creature_positions):
    """Ensure food doesn't overlap with existing food or creatures."""
    while len(foods) < FOOD_COUNT:
        x, y = random.randint(0, 800), random.randint(0, 600)
        if (x, y) not in existing_food and not any(math.hypot(x - cx, y - cy) < FOOD_RADIUS for cx, cy in creature_positions):
            foods.append((x, y))
            existing_food.add((x, y))

def main():
    sim_state = SimulationState()
    sim_state.creatures = [Creature(random.randint(0, 800), random.randint(0, 600), random_genome())
                           for _ in range(CREATURE_COUNT)]
    sim_state.food = [(random.randint(0, 800), random.randint(0, 600)) for _ in range(FOOD_COUNT)]
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    sim_state.save('simulation.save')
                elif event.key == pygame.K_l:
                    sim_state.load('simulation.save')

        # Update creatures
        new_creatures = []
        for creature in sim_state.creatures:
            creature.update(sim_state.food)
            if new_child := creature.reproduce():
                new_creatures.append(new_child)
        
        # Add new creatures and remove dead ones
        sim_state.creatures = [c for c in sim_state.creatures if c.alive] + new_creatures

        # If all creatures are dead, restart the simulation with the best creatures
        if len(sim_state.creatures) == 0:
            best_creatures = sim_state.get_best_creatures()
            if best_creatures:
                sim_state.creatures = [Creature(random.randint(0, 800), random.randint(0, 600), c.genome)
                                       for c in best_creatures]  # Restart with best creatures
                print("New generation started with the best creatures.")
            else:
                # If no best creatures, restart the simulation with random creatures
                sim_state.creatures = [Creature(random.randint(0, 800), random.randint(0, 600), random_genome())
                                       for _ in range(CREATURE_COUNT)]
                print("No best creatures found, starting new simulation.")

        # Maintain food supply with non-overlapping food
        creature_positions = [(c.x, c.y) for c in sim_state.creatures]
        existing_food = set(sim_state.food)
        spawn_food(sim_state.food, existing_food, creature_positions)

        # Adjust mutation rate based on diversity
        unique_genomes = len(set(tuple(c.genome) for c in sim_state.creatures))
        if unique_genomes < len(sim_state.creatures) * 0.3:
            sim_state.mutation_rate = min(sim_state.mutation_rate + MUTATION_INCREASE_RATE, 0.5)
        else:
            sim_state.mutation_rate = max(sim_state.mutation_rate - MUTATION_INCREASE_RATE / 2, MUTATION_RATE)

        # Draw everything
        screen.fill((30, 30, 30))
        for food in sim_state.food:
            pygame.draw.circle(screen, (0, 255, 0), (int(food[0]), int(food[1])), FOOD_RADIUS)
        for creature in sim_state.creatures:
            pygame.draw.circle(screen, (0, 100, 255), (int(creature.x), int(creature.y)), 5)
        
        # Display stats
        stats = [
            f"Creatures: {len(sim_state.creatures)}",
            f"Mutation Rate: {sim_state.mutation_rate:.2f}",
            f"Unique Genomes: {unique_genomes}",
            f"Generation: {sim_state.generation}"
        ]
        for i, text in enumerate(stats):
            screen.blit(font.render(text, True, (255, 255, 255)), (10, 10 + i*20))
        
        pygame.display.flip()
        clock.tick(30)
        sim_state.generation += 1

    pygame.quit()

if __name__ == "__main__":
    main()