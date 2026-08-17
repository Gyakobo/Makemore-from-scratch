import torch
import matplotlib.pyplot as plt

words = open("names.txt", "r").read().splitlines()

# Rudimentary approach
"""
b = {}
for word in words[:]:
    chs = ["<S>"] + list(word) + ["<E>"]

    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = (
            b.get(bigram, 0) + 1
        )  # Technical b.get(bigram, 0) <=> b[bigram] without the safe guard 0
"""


# Another better approach
N = torch.zeros((27, 27), dtype=torch.int32)

# All the available letters from the text file dataset
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

for word in words:
    chs = ["."] + list(word) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

plt.figure(figsize=(16, 16))
plt.imshow(N, cmap="Blues")
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
# plt.axis("off")
# plt.show()

# p = N[0].float()

# p = torch.rand(3, generator=g)
# p = p / p.sum()

P = N.float()
P = P / P.sum(1, keepdim=True)  # Broadcast all operation
# Basically divide [27, 27] array by a [27, 1] array

g = torch.Generator().manual_seed(2147483647)

for i in range(5):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print("".join(out))
