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
vocab_size = len(
    stoi
)  # Size of 'vocabulary'/quantity of all the possible/usable characters

block_size = (
    3  # context length: how many characters do we take to predict the next one?
)


# build the dataset
def build_dataset(words):
    X, Y = [], []
    for w in words:
        # print(w)
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
Xtr, Ytr = build_dataset(words[:n1])  # Training split (80%)
Xdev, Ydev = build_dataset(words[n1:n2])  # Dev/Validation split (10%)
Xte, Yte = build_dataset(words[n2:])  # Testing split (10%)

"""
MLP revised
"""
n_embd = 10  # the dimensionality of the character embedding vectors. Basically the dimension per character
n_hidden = 200  # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647)  # for reproducibility
C = torch.randn(
    (vocab_size, n_embd), generator=g
)  # A 2D vector for each letter/character
W1 = (
    torch.randn((n_embd * block_size, n_hidden), generator=g)
    * (5 / 3)
    / (n_embd * block_size) ** 0.5  # W1 * (5/3) / (gain/sqrt(fan_in))
)
b1 = torch.randn(n_hidden, generator=g) * 0.01
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
b2 = torch.randn(vocab_size, generator=g) * 0
parameters = [C, W1, b1, W2, b2]

# lre = torch.linspace(-3, 0, 1000)
# lrs = 10**lre

for p in parameters:
    p.requires_grad = True


max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    """
    Minibatch construct
    """
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]  # batch X, Y - just simpler notation

    """
    Forward Pass
    """
    emb = C[Xb]  # (32, 3, 2) embed the characters into vectors
    """
    # emb @ W + b1 (incompatible) => torch.cat(torch.unbind(emb, 1), 1)
    h = torch.tanh(emb.view(-1, n_embd * block_size) @ W1 + b1)  # (32, 100)
    logits = h @ W2 + b2  # (32, 27)
    # counts = logits.exp()
    # prob = counts / counts.sum(1, keepdim=True)
    # loss = -prob[torch.arange(32), Y].log().mean()
    loss = F.cross_entropy(logits, Ytr[ix])
    """
    embcat = emb.view(emb.shape[0], -1)  # concatenate the vectors
    hpreact = embcat @ W1 + b1  # hidden layer pre-activation
    h = torch.tanh(hpreact)  # hidden layer
    logits = h @ W2 + b2  # output layer
    loss = F.cross_entropy(logits, Yb)  # loss function

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
    lr = 0.1 if i < 100000 else 0.01  # step learning rate decay
    for p in parameters:
        p.data += -lr * p.grad

    """
    Track stats
    """
    if i % 10000 == 0:  # print every once in a while
        print(f"{i:7d}/{max_steps:7d}: {loss.item():.4f}")
    lossi.append(loss.log10().item())


"""
Visualize results
"""
# plt.plot(lossi)
# plt.show()
plt.figure(figsize=(20, 10))
plt.imshow(h.abs() > 0.99, cmap="gray", interpolation="nearest")
plt.show()

"""
Test results
"""


@torch.no_grad()  # this decorator disables gradient tracking
def split_loss(split: str):
    x, y = {
        "train": (Xtr, Ytr),
        "val": (Xdev, Ydev),
        "test": (Xte, Yte),
    }[split]
    emb = C[x]  # (N, block_size, n_embd)
    embcat = emb.view(emb.shape[0], -1)  # concat into (N, block_size * n_embd)
    h = torch.tanh(embcat @ W1 + b1)  # (N, n_hidden)
    logits = h @ W2 + b2  # (N, vocab_size)
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())


# split_loss("train")
# split_loss("val")

"""
Final step: Sampling for the model
"""
g = torch.Generator().manual_seed(2147483647 + 10)
for _ in range(20):
    out = []
    context = [0] * block_size  # initialize with all ...
    while True:
        emb = C[torch.tensor([context])]  # (1, block_size, d)
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break

    print("".join(itos[i] for i in out))
