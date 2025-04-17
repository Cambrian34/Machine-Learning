# dataset.py
import os
import torch
import torchaudio
from torch.utils.data import Dataset
from utils import text_to_indices, load_vocab

class LibriSpeechDataset(Dataset):
    def __init__(self, root_dir="/Users/alistairchambers/Machine Learning/LibriSpeech/test-clean", sample_rate=16000):
        self.root = root_dir
        self.vocab, _ = load_vocab()
        self.items = []

        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".trans.txt"):
                    with open(os.path.join(root, file)) as f:
                        for line in f:
                            parts = line.strip().split(" ", 1)
                            if len(parts) == 2:
                                file_id, transcript = parts
                                audio_path = os.path.join(root, file_id + ".flac")
                                if os.path.exists(audio_path):
                                    self.items.append((audio_path, transcript))

        self.resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=sample_rate)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=80)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, text = self.items[idx]
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            waveform = self.resampler(waveform)
        mel = self.mel(waveform).squeeze(0).transpose(0, 1)  # (Time, 80)
        target = text_to_indices(text, self.vocab)
        return mel, torch.tensor(target)