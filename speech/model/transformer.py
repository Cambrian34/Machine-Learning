# model/transformer.py
import torch.nn as nn
import torch

class TinySpeechTransformer(nn.Module):
    def __init__(self, input_dim=80, model_dim=128, num_heads=2, num_layers=2, vocab_size=30):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(model_dim, vocab_size)

    def forward(self, x):  # x: (batch, time, features)
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)  # (time, batch, features)
        x = self.encoder(x)
        x = self.classifier(x.permute(1, 0, 2))  # (batch, time, vocab)
        return x