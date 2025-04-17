import pygame
import sounddevice as sd
import soundfile as sf
import torch
import numpy as np
import librosa
import os
from speechrecog import SpeechRecognizer, extract_features, idx_to_char, SAMPLE_RATE

# Load your model
model = SpeechRecognizer()
model.load_state_dict(torch.load("speech_recognizer.pth", map_location=torch.device("cpu")))
model.eval()

# Greedy decode
def greedy_decode(output):
    pred = torch.argmax(output, dim=2).transpose(0, 1)
    transcripts = []
    for p in pred:
        decoded = []
        prev = -1
        for idx in p:
            idx = idx.item()
            if idx != prev and idx != 0:
                decoded.append(idx_to_char.get(idx, ""))
            prev = idx
        transcripts.append("".join(decoded))
    return transcripts

# Record and transcribe
def record_and_transcribe():
    try:
        global transcript_text, is_recording
        is_recording = True
        recording = sd.rec(int(5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        sf.write("temp.wav", recording, SAMPLE_RATE)
        waveform, sr = sf.read("temp.wav")
        waveform = librosa.resample(waveform.T, orig_sr=sr, target_sr=SAMPLE_RATE) if sr != SAMPLE_RATE else waveform.T
        features = extract_features(waveform).unsqueeze(0)
        with torch.no_grad():
            logits = model(features).log_softmax(2)
            prediction = greedy_decode(logits)
        transcript_text = prediction[0]
    except Exception as e:
        transcript_text = f"Error: {str(e)}"
    finally:
        is_recording = False

# Pygame init
pygame.init()
WIDTH, HEIGHT = 600, 300
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("🎤 Pygame Voice AI")
font = pygame.font.SysFont("arial", 24)
clock = pygame.time.Clock()

# Colors
WHITE, BLACK, GREEN, RED = (255, 255, 255), (0, 0, 0), (50, 200, 50), (200, 50, 50)

button_rect = pygame.Rect(WIDTH//2 - 100, HEIGHT//2 - 25, 200, 50)

transcript_text = ""
is_recording = False

# Main loop
running = True
while running:
    screen.fill(WHITE)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if button_rect.collidepoint(event.pos) and not is_recording:
                transcript_text = "Recording..."
                pygame.display.update()
                pygame.time.set_timer(pygame.USEREVENT, 100)
                record_and_transcribe()

    # Draw record button
    pygame.draw.rect(screen, GREEN if not is_recording else RED, button_rect)
    btn_label = "Start Recording" if not is_recording else "Recording..."
    text_surface = font.render(btn_label, True, BLACK)
    screen.blit(text_surface, (button_rect.centerx - text_surface.get_width()//2,
                               button_rect.centery - text_surface.get_height()//2))

    # Draw transcript
    lines = transcript_text.split('\n')
    for i, line in enumerate(lines):
        t = font.render(line, True, BLACK)
        screen.blit(t, (20, HEIGHT - 100 + 25*i))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()