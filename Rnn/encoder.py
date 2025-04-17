from pathlib import Path
import torch
from combinator import *

# Load the text file in the current directory
file_path = Path(__file__).parent / 'combined_text.txt'
with open(file_path, 'r', encoding='utf-8') as f:
    combined_text = f.read()

#put data in a list
combined_text = combined_text.split('\n')
# Remove empty strings from the list
combined_text = [line for line in combined_text if line.strip()]


chars = sorted(list(set(combined_text)))
vocab_size = len(chars)

# Char ↔ index mappings
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

# Encoding and decoding functions
def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

# Convert the combined text into a tensor
data = torch.tensor(encode(combined_text), dtype=torch.long)

# Saving the tensor to a file
torch.save(data, "combined_text_tensor.pt")