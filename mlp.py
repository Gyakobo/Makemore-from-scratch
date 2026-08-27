from itertools import count
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

words = open("names.txt", "r").read().splitlines()

# Vocabulary of characters
chars = sorted(list(set("".join(words))))
stoi = {c: i + 1 for i, c in enumerate(chars)}
stoi["."] = 0
itos = {i: c for c, i in stoi.items()}


# build the dataset
def build_dataset(words):
    block_size = (
        3  # context length: how many characters do we take to predict the next one?
    )
    X, Y = [], []
    for w in words:
        print(w)
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            # print("".join(itos[i] for i in context), "--->", itos[ix])
            context = context[1:] + [ix]  # crop and append
    X = torch.tensor(X)  # (32, 3)
    Y = torch.tensor(Y)  # (32, )
    print(X.shape, Y.shape)
    return X, Y


random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])  # Training split
Xdev, Ydev = build_dataset(words[n1:n2])  # Dev/Validation split
Xte, Yte = build_dataset(words[n2:])  # Testing split

g = torch.Generator().manual_seed(2147483647)  # for reproducibility
C = torch.randn((27, 10), generator=g)  # A 2D vector for each letter
W1 = torch.randn((30, 200), generator=g)
b1 = torch.randn(200, generator=g)
W2 = torch.randn((200, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]

lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre

for p in parameters:
    p.requires_grad = True

lri = []
lossi = []
stepi = []

for i in range(50000):
    """
    Minibatch construct
    """
    ix = torch.randint(0, Xtr.shape[0], (32,))

    """
    Forward Pass
    """
    emb = C[Xtr[ix]]  # (32, 3, 2)
    # emb @ W + b1 (incompatible) => torch.cat(torch.unbind(emb, 1), 1)
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)  # (32, 100)
    logits = h @ W2 + b2  # (32, 27)
    # counts = logits.exp()
    # prob = counts / counts.sum(1, keepdim=True)
    # loss = -prob[torch.arange(32), Y].log().mean()
    loss = F.cross_entropy(logits, Ytr[ix])
    # print(loss.item())

    """
    Backward Pass
    """
    for p in parameters:
        p.grad = None
    loss.backward()

    """
    Update
    """
    # lr = lrs[i]
    lr = 0.1
    for p in parameters:
        p.data += -lr * p.grad

    """
    Track stats
    """
    # lri.append(lre[i])
    # lossi.append(loss.item())

# plt.plot(lri, lossi)
# plt.show()

"""
Visualize results
"""
emb = C[Xtr]  # (32, 3, 2)
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)  # (32, 100)
logits = h @ W2 + b2  # (32, 27)
loss = F.cross_entropy(logits, Ytr)
print(f"{loss=}")

emb = C[Xdev]  # (32, 3, 2)
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)  # (32, 100)
logits = h @ W2 + b2  # (32, 27)
loss = F.cross_entropy(logits, Ydev)
print(f"{loss=}")
