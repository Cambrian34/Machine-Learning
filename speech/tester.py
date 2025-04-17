import torchaudio
from torchaudio.transforms import MelSpectrogram, Resample
TEST_AUDIO = "/Users/alistairchambers/Machine Learning/LibriSpeech/test-clean/61/70968/61-70968-0009.flac"

# Example for 16kHz audio input
waveform, sample_rate = torchaudio.load(TEST_AUDIO)
resampler = Resample(orig_freq=sample_rate, new_freq=16000)
waveform = resampler(waveform)

mel_spec = MelSpectrogram(
    sample_rate=16000,
    n_mels=80,
    n_fft=400,
    hop_length=160
)(waveform)  # (1, 80, Time)