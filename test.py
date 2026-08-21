import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

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

P = (N + 1).float()  # Adjusts the null values to avoid .inf - model smoothing
P /= P.sum(dim=1, keepdim=True)  # Probability of each element by row

ix = 0
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
log_likelihood = 0.0
n = 0
for w in words[:3]:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        # print(f"{ch1}{ch2}: {prob:.4f} {logprob:.4f}")
# print(f"{log_likelihood=}")
nll = -log_likelihood
# print(f"{nll=}")
# print(f"normalized nnl: {nll/n}")


"""
Let's now plug in a neural network to this 
"""

# Create a training set of all the bigrams(x, y)
xs, ys = [], []

for w in words[:1]:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        print(ch1, ch2)
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
print(f"{xs=}")
print(f"{ys=}")

xenc = F.one_hot(xs, num_classes=27).float()

W = torch.randn((27, 27), generator=g)
logits = xenc @ W  # predictz log-counts

# These last two lines are called the SOFT MAX
counts = logits.exp()  # counts, equivalent to the N matrix
probs = counts / counts.sum(1, keepdim=True)  # probabilities for next character

"""
Helper function to understand this new framework
"""
nlls = torch.zeros(5)
for i in range(5):
    # i-th bigram:
    x = xs[i].item()  # input character index
    y = ys[i].item()  # label character index

    print("-------")
    print(f"bigram example {i + 1}: {itos[x]}{itos[y]} (indexes {x}, {y})")
    print("input to the neural net:", x)
    print("output probabilities from the neural net:", probs[i])
    print("label (actual next character):", y)
    y = probs[i, y]
    print("probablity assigned by the net to the correct character:", p.item())
    logp = torch.log(p)
    print("log likelihood:", logp.item())
    nll = -logp
    print("negative log likelihood:", nll.item())
    nlls[i] = nll
print("========")
print("average negative log likelihood, i.e. loss =", nlls.mean().item())
