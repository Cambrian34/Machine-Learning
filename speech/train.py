# train.py
import torch
from torch.utils.data import DataLoader
from dataset import LibriSpeechDataset
from model.transformer import TinySpeechTransformer
from torch.nn import CTCLoss
import torch.optim as optim
from utils import load_vocab

vocab, _ = load_vocab()
device = "cuda" if torch.cuda.is_available() else "cpu"

ds = LibriSpeechDataset()
loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=lambda b: b)  # simple collate for now

model = TinySpeechTransformer(vocab_size=len(vocab)).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = CTCLoss(blank=vocab["|"])

for epoch in range(10):
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        inputs, targets = zip(*batch)

        input_lengths = [x.shape[0] for x in inputs]
        target_lengths = [len(t) for t in targets]

        inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True).to(device)
        targets = torch.cat(targets).to(device)

        outputs = model(inputs)
        log_probs = torch.nn.functional.log_softmax(outputs, dim=-1)

        loss = criterion(
            log_probs.permute(1, 0, 2),  # (time, batch, vocab)
            targets,
            input_lengths=torch.tensor(input_lengths),
            target_lengths=torch.tensor(target_lengths)
        )
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} Loss: {loss.item():.4f}")