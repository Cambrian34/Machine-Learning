import torch
import torch.nn as nn
import random
import numpy as np

# Create input and target sequences
seq_length = 100  # Length of each sequence

data = torch.load("combined_text_tensor.pt")

def create_sequences(data, seq_length):
    inputs = []
    targets = []
    for i in range(0, len(data) - seq_length, seq_length):
        inputs.append(data[i:i+seq_length])
        targets.append(data[i+1:i+seq_length+1])  # Targets are just the next character
    return torch.stack(inputs), torch.stack(targets)

# Split the data into input/target pairs
inputs, targets = create_sequences(data, seq_length)

# Check the shape of the data
print(f"Inputs shape: {inputs.shape}")
print(f"Targets shape: {targets.shape}")