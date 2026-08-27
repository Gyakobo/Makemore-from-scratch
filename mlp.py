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
# print(f"{X=}")
# print(f"{Y=}")

g = torch.Generator().manual_seed(2147483647)  # for reproducibility
C = torch.randn((27, 2), generator=g)  # A 2D vector for each letter
W1 = torch.randn((6, 100), generator=g)
b1 = torch.randn(100, generator=g)
W2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True

for _ in range(100):
    """
    Minibatch construct
    """
    ix = torch.randint(0, X.shape[0], (32,))

    """
    Forward Pass
    """
    emb = C[X[ix]]  # (32, 3, 2)
    # emb @ W + b1 (incompatible) => torch.cat(torch.unbind(emb, 1), 1)
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)  # (32, 100)
    logits = h @ W2 + b2  # (32, 27)
    # counts = logits.exp()
    # prob = counts / counts.sum(1, keepdim=True)
    # loss = -prob[torch.arange(32), Y].log().mean()
    loss = F.cross_entropy(logits, Y[ix])

    print(loss.item())
    """
    Backward Pass
    """
    for p in parameters:
        p.grad = None
    loss.backward()

    """
    Update
    """
    for p in parameters:
        p.data += -0.1 * p.grad
