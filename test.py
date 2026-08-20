import torch
import matplotlib.pyplot as plt

words = open("names.txt", "r").read().splitlines()

"""
Here basically you're storing the information in a dictionary,
however, it's a bit inconvenient as it's way better to store
such information in a `tensor` 2D array specifically
"""
b = {}
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1

# sorted(b.items(), key=lambda kv: -kv[1])
# print(f"{b=}")


"""
Here is a rendition of the same approach only in a 2D tensor array
"""
N = torch.zeros((27, 27), dtype=torch.int32)

chars = sorted(
    list(set("".join(words)))
)  # Would give us the exact alphabet used in the training dataset

stoi = {c: i + 1 for (i, c) in enumerate(chars)}
stoi["."] = 0
itos = {i: c for (c, i) in stoi.items()}

for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

g = torch.Generator().manual_seed(2147483647)

# p = N[0].float()
# p = p / p.sum()
# ix = torch.multinomial(
#     p, num_samples=1, replacement=True, generator=g
# ).item()  # Get value from the tensor, tensor([3]).item() -> 3

ix = 0
P = N.float()  # Adjusts the null values to avoid .inf
P /= torch.sum(P, dim=1, keepdim=True)  # Probability of each element by row

while True:
    p = P[ix]
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    # print(itos[ix])
    if ix == 0:
        break

"""
# Plotting function
plt.figure(figsize=(16, 16))
plt.imshow(N, cmap="Blues")
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
plt.axis("off")
plt.show()
"""


"""
Now let's add loss function now 
"""
# Let's first look at the probability of each binomial character
for w in words[:3]:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch1]
        prob = P[ix1, ix2]
        print(f"{ch1}{ch2}: {prob:.4f}")
