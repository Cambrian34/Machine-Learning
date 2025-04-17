import pygame
import json
import os
import time
from datetime import datetime

WIDTH, HEIGHT = 640, 400
FONT_SIZE = 20

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Training Monitor")
font = pygame.font.SysFont("Consolas", FONT_SIZE)
clock = pygame.time.Clock()

def draw_graph(loss_history):
    max_loss = max(loss_history) if loss_history else 1
    if len(loss_history) < 2:
        return

    for i in range(len(loss_history)-1):
        x1 = i * 6
        x2 = (i + 1) * 6
        y1 = HEIGHT - int((loss_history[i] / max_loss) * (HEIGHT - 100))
        y2 = HEIGHT - int((loss_history[i + 1] / max_loss) * (HEIGHT - 100))
        pygame.draw.line(screen, (0, 255, 0), (x1, y1), (x2, y2), 2)

def draw_text(text, x, y, color=(255, 255, 255)):
    screen.blit(font.render(text, True, color), (x, y))

def load_log():
    if os.path.exists("training_log.json"):
        with open("training_log.json", "r") as f:
            return json.load(f)
    return {}

def main():
    while True:
        screen.fill((20, 20, 20))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        data = load_log()
        epoch = data.get("epoch", 0)
        loss = data.get("loss", 0)
        sample = data.get("sample", "")
        loss_history = data.get("loss_history", [])

        draw_text(f"Epoch: {epoch}", 10, 10)
        draw_text(f"Loss: {loss:.4f}", 10, 40)
        draw_text("Sample:", 10, 70)
        draw_text(sample[:50], 10, 100)  # limit length
        draw_graph(loss_history)

        pygame.display.flip()
        clock.tick(1)  # Update 1x per second

if __name__ == "__main__":
    main()