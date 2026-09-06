import torch
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt  # for making figures

random.seed(42)

words = open("names.txt", "r").read().splitlines()
random.shuffle(words)

# build the vocabulary of characters and mappings to/from integers
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)

# build the dataset
block_size = 3  # context length: how many char(s) do we take to predict the next one?


def build_dataset(words):
    X, Y = [], []

    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # crop and append

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(f"{X=}")
    print(f"{Y=}")
    return X, Y


n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])  # 80%
Xdev, Ydev = build_dataset(words[n2:n2])  # 10%
Xte, Yte = build_dataset(words[n2:])  # 10%

for x, y in zip(Xtr, Ytr):
    print("".join(itos[ix.item()] for ix in x), "-->", itos[y.item()])
