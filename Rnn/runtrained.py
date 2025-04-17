import torch
import torch.nn as nn

# --- 1. Load Vocabulary Mappings ---
bundle = torch.load("combined_text_bundle.pt")
stoi = bundle['stoi']  # char-to-index mapping
itos = bundle['itos']  # index-to-char mapping
vocab_size = len(stoi)

# --- 2. Define the Model (Same as Before) ---
class TinyRNN(nn.Module):
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

# --- 3. Load Model Weights ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TinyRNN(vocab_size).to(device)
model.load_state_dict(torch.load("tiny_rnn_model.pth", map_location=device))
model.eval() 

# --- 4. Text Generation Function  ---
def generate(model, start_text, length=150, temperature=0.8):
    model.eval()
    context = torch.tensor([stoi[c] for c in start_text], dtype=torch.long).unsqueeze(0).to(device)
    hidden = None
    output = start_text

    for _ in range(length):
        logits, hidden = model(context, hidden)
        logits = logits[:, -1, :] / temperature
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        output += itos[next_id]
        context = torch.tensor([[next_id]], dtype=torch.long).to(device)

    return output

# --- 5. Generate Text! ---
print("\n--- Generated Text ---")
start_text = "HE WENT".lower()
print(generate(model, start_text, length=200))  # Adjust length as needed