# utils.py
import json

def load_vocab(path="/Users/alistairchambers/Machine Learning/speech/data/vocab.json"):
    with open(path, 'r') as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}
    return vocab, inv_vocab

def text_to_indices(text, vocab):
    return [vocab.get(c, vocab["<unk>"]) for c in text.lower()]