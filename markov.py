import random
from collections import defaultdict

def train_markov_model(text):
    words = text.split()
    model = defaultdict(list)
    for i in range(len(words)-1):
        model[words[i]].append(words[i+1])
    return model

def generate_text(model, start_word, length=10):
    word = start_word
    result = [word]
    for _ in range(length-1):
        word = random.choice(model[word]) if model[word] else random.choice(list(model.keys()))
        result.append(word)
    return ' '.join(result)

text = "I like apples. You like bananas. We all like fruits."
model = train_markov_model(text)
print(generate_text(model, "we"))