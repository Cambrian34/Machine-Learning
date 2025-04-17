import pygame
import threading
import numpy as np
import random
import time

# === Config ===
SCREEN_SIZE = 600
NUM_CREATURES = 25
TICK_TIME = 0.1  # seconds between logic updates

# === Creature Class ===
class Creature(threading.Thread):
    def __init__(self, world, x, y, genome=None):
        threading.Thread.__init__(self)
        self.world = world
        self.x, self.y = x, y
        self.genome = genome if genome else self.random_genome()
        self.fitness = 0
        self.color = (random.randint(50, 255), 200, 200)
        self.running = True
        self.lock = threading.Lock()

    def random_genome(self):
        # Placeholder: could be a list of instructions or weights
        return [random.random() for _ in range(10)]

    def run(self):
        while self.running:
            self.step()
            time.sleep(TICK_TIME)

    def step(self):
        # Simple logic: random walk
        dx = random.choice([-1, 0, 1])
        dy = random.choice([-1, 0, 1])
        self.x = max(0, min(SCREEN_SIZE, self.x + dx * 5))
        self.y = max(0, min(SCREEN_SIZE, self.y + dy * 5))
        self.fitness += 1  # Or reward for reaching food etc.

    def stop(self):
        self.running = False

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), 5)

# === Main World Logic ===
class World:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        pygame.display.set_caption("Threaded Creatures")
        self.clock = pygame.time.Clock()
        self.creatures = []

    def spawn_creatures(self):
        self.creatures = []
        for _ in range(NUM_CREATURES):
            c = Creature(self, random.randint(0, SCREEN_SIZE), random.randint(0, SCREEN_SIZE))
            self.creatures.append(c)
            c.start()

    def run(self):
        self.spawn_creatures()
        running = True
        while running:
            self.screen.fill((20, 20, 20))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Draw all creatures
            for c in self.creatures:
                c.draw(self.screen)

            pygame.display.flip()
            self.clock.tick(60)

        # Stop all threads cleanly
        for c in self.creatures:
            c.stop()
        for c in self.creatures:
            c.join()

        pygame.quit()

# === Launch the world ===
if __name__ == "__main__":
    world = World()
    world.run()