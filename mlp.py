from itertools import count
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open("names.txt", "r").read().splitlines()

# Vocabulary of characters
chars = sorted(list(set("".join(words))))
stoi = {c: i + 1 for i, c in enumerate(chars)}
stoi["."] = 0
itos = {i: c for c, i in stoi.items()}

# build the dataset

block_size = (
    3  # context length: how many characters do we take to predict the next one?
)
X, Y = [], []
for w in words[:5]:
    print(w)
    context = [0] * block_size
    for ch in w + ".":
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        print("".join(itos[i] for i in context), "--->", itos[ix])
        context = context[1:] + [ix]  # crop and append

X = torch.tensor(X)
Y = torch.tensor(Y)
print(f"{X=}")
print(f"{Y=}")

C = torch.randn((27, 2))
# F.one_hot(torch.tensor(5), num_classes=27).float()

W1 = torch.randn((6, 100))
b1 = torch.randn(100)
emb = C[X]

# emb @ W + b1 (incompatible)

# torch.cat(torch.unbind(emb, 1), 1)
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)

W2 = torch.randn((100, 27))
b2 = torch.randn(27)

logits = h @ W2 + b2
counts = logits.exp()
prob = counts / counts.sum(1, keepdim=True)

# Finally Testing the probabities with Y
print(prob[torch.arange(32), Y])
