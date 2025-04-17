import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import librosa
from torch.utils.data import Dataset, DataLoader
from torch.nn import CTCLoss

# ----------- CONFIG -----------

DATA_PATH = "/Users/alistairchambers/Machine Learning/LibriSpeech/test-clean"  # <-- Change if needed
SAMPLE_RATE = 16000
NUM_EPOCHS = 5
BATCH_SIZE = 4

VOCAB = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ "
char_to_idx = {c: i for i, c in enumerate(VOCAB)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

# ----------- DATASET -----------

class LibriSpeechDataset(Dataset):
    def __init__(self, root_dir):
        self.sample_rate = SAMPLE_RATE
        self.data = []

        for speaker in os.listdir(root_dir):
            speaker_path = os.path.join(root_dir, speaker)
            if not os.path.isdir(speaker_path): continue

            for chapter in os.listdir(speaker_path):
                chapter_path = os.path.join(speaker_path, chapter)
                trans_path = os.path.join(chapter_path, f"{speaker}-{chapter}.trans.txt")

                if not os.path.isfile(trans_path): continue

                with open(trans_path, "r") as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split(" ", 1)
                    if len(parts) != 2: continue
                    file_id, transcript = parts
                    audio_path = os.path.join(chapter_path, f"{file_id}.flac")
                    if os.path.isfile(audio_path):
                        self.data.append((audio_path, transcript))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, transcript = self.data[idx]
        try:
            waveform, sr = sf.read(audio_path)

            if sr != self.sample_rate:
                waveform = librosa.resample(waveform.T, orig_sr=sr, target_sr=self.sample_rate)
            else:
                waveform = waveform.T

            return waveform, transcript
        except Exception as e:
            print(f"Error processing file {audio_path}: {e}")
            return None, None

# ----------- UTILS -----------

mel_transform = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=80, n_fft=400, hop_length=160)
db_transform = T.AmplitudeToDB()

def extract_features(waveform):
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=0)
    tensor = torch.tensor(waveform, dtype=torch.float32)
    mel_spec = mel_transform(tensor)
    return db_transform(mel_spec)

def text_to_indices(text):
    return [char_to_idx[c] for c in text.upper() if c in char_to_idx]

def collate_fn(batch):
    features = []
    labels = []
    input_lengths = []
    label_lengths = []

    for waveform, transcript in batch:
        mel = extract_features(waveform)
        features.append(mel.transpose(0, 1))  # Make time the first dimension
        label = torch.tensor(text_to_indices(transcript), dtype=torch.long)
        labels.append(label)
        input_lengths.append(mel.shape[1])
        label_lengths.append(len(label))

    features = nn.utils.rnn.pad_sequence(features, batch_first=True).permute(0, 2, 1)
    labels = torch.cat(labels)
    return features, labels, torch.tensor(input_lengths), torch.tensor(label_lengths)

# ----------- MODEL -----------

class SpeechRecognizer(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=256, output_dim=len(VOCAB)):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.rnn = nn.LSTM(input_size=32 * 40, hidden_size=hidden_dim, num_layers=2,
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, Mels, Time)
        x = self.cnn(x)     # (B, C, M', T')
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, T', C, M')
        B, T, C, M = x.shape
        x = x.view(B, T, C * M)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

# ----------- MAIN -----------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LibriSpeechDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = SpeechRecognizer().to(device)
    loss_fn = CTCLoss(blank=0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            inputs, targets, input_lengths, target_lengths = batch
            input_lengths = input_lengths // 4
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs).log_softmax(2)

            

            loss = loss_fn(logits.permute(1, 0, 2), targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}], Loss: {running_loss/10:.4f}")
                running_loss = 0.0

    torch.save(model.state_dict(), "speech_recognizer.pth")
    print("Training complete. Model saved as speech_recognizer.pth")

if __name__ == "__main__":
    main()