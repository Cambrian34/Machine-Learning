# rnn_textgen.py

import torch
import torch.nn as nn
from pathlib import Path


# Load and prepare text data
file_path = Path(__file__).parent / 'combined_text.txt'
with open(file_path, 'r', encoding='utf-8') as f:
    combined_text = f.read()

# Preprocess
lines = [line for line in combined_text.split('\n') if line.strip()]
full_text = '\n'.join(lines)

# Create vocabulary
chars = sorted(list(set(full_text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Encoding / Decoding
def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

# Tensor data
data = torch.tensor(encode(full_text), dtype=torch.long)
torch.save({
    'data': data,
    'stoi': stoi,
    'itos': itos
}, "combined_text_bundle.pt")

# -------------------------------
# RNN Model
# -------------------------------
class TinyRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=32, hidden_size=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        #self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
        #self.dropout = nn.Dropout(p=0.5)  # Apply dropout with a 50% chance

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.rnn(x, hidden)
        logits = self.fc(out)
        return logits, hidden


# version 2
class BetterRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=512, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size, 
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2  # Dropout between layers
        )
        self.dropout = nn.Dropout(0.3)  # Final dropout
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits, hidden
# -------------------------------
# Hyperparameters
# -------------------------------

eval_interval = 100
# Update hyperparameters
block_size = 128    # Longer sequence length
batch_size = 64      # Larger batches
max_iters = 15000    # Train longer
learning_rate = 0.001 # Higher learning rate


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Prepare data for batching
def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)



# -------------------------------
# Train
# -------------------------------
#model = TinyRNN(vocab_size).to(device)
model = BetterRNN(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
loss_fn = nn.CrossEntropyLoss()

for iter in range(max_iters):
    xb, yb = get_batch(data, block_size, batch_size)
    logits, _ = model(xb)

    loss = loss_fn(logits.view(-1, vocab_size), yb.view(-1))

    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % eval_interval == 0:
        print(f"Step {iter}: loss = {loss.item():.4f}")

# -------------------------------
# Text Generation
# -------------------------------
def generate(model, start_text, length=100, temperature=0.8):
    model.eval()
    context = torch.tensor([stoi[c] for c in start_text], dtype=torch.long).unsqueeze(0).to(device)
    hidden = None
    output = start_text

    for _ in range(length):
        logits, hidden = model(context, hidden)
        logits = logits[:, -1, :] / temperature
        #probs = torch.softmax(logits, dim=-1)
        probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        output += itos[next_id]
        context = torch.tensor([[next_id]], dtype=torch.long).to(device)

    return output

# Try generating some text
print("\n--- Sample ---")
start_text = "p"#testing words to see what words are are in the dataset
#lowercase
start_text = start_text.lower()
print(generate(model, start_text, length=100))


torch.save(model.state_dict(), "tiny_rnn_model.pth")