# How the Makemore MLP Works: From Letters to Loss

A walkthrough of `mlp.py` from [Makemore-from-scratch](https://github.com/Gyakobo/Makemore-from-scratch), covering the math notation, the code, and a plain-English explanation for every stage — from turning letters into numbers, through the embedding lookup table, to computing the training loss.

---

## Stage 0: Turning letters into numbers

Neural nets only understand numbers, not letters, so the first thing the code does is build a translation table.

**Code:**
```python
chars = sorted(list(set("".join(words))))
stoi = {c: i + 1 for i, c in enumerate(chars)}
stoi["."] = 0
itos = {i: c for c, i in stoi.items()}
```

**Math notation:** this defines a function `stoi: character → integer`, e.g. `stoi('a') = 1`, `stoi('b') = 2`, ..., `stoi('.') = 0`. The `.` is a special "start/end of word" marker. With 26 letters + 1 dot, there are **27 possible characters**, indexed `0` through `26`.

**Plain words:** a computer can't multiply "the letter q" by a weight. So every letter (and the boundary marker) gets a unique ID number. That's it — no meaning yet, just a name tag.

---

## Stage 1: Building X and Y — the training examples

**Code:**
```python
block_size = 3
X, Y = [], []
for w in words[:5]:
    context = [0] * block_size
    for ch in w + ".":
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]
```

**Plain words:** the whole task the network is being trained to do is "given the last 3 letters, guess the next one." For the name `emma`, the sliding window of examples looks like:

| context (X) | next letter (Y) |
|---|---|
| `. . .` | `e` |
| `. . e` | `m` |
| `. e m` | `m` |
| `e m m` | `a` |
| `m m a` | `.` |

Every row is one training example. `X` is a list of 3-letter contexts (as integer IDs), and `Y` is the correct answer for each (also an integer ID). `context = context[1:] + [ix]` means "drop the oldest letter, append the new one" — the sliding window.

**Math notation:** each row of X is a vector $x^{(i)} = (c_1, c_2, c_3)$ where each $c_j \in \{0, ..., 26\}$, and the label is $y^{(i)} \in \{0, ..., 26\}$. After the loop, `X` has shape `(N, 3)` and `Y` has shape `(N,)`, where N is the total number of examples across all words used (32, for the first 5 words — worth using `X.shape[0]` instead of hardcoding `32` later, so it doesn't break when more words are used).

---

## Stage 2: The lookup table — turning IDs into meaningful vectors

**Code:**
```python
C = torch.randn((27, 2))
emb = C[X]
```

**Plain words:** an integer ID like "5" carries no information about *how similar* letters are to each other — 5 and 6 are just as "different" as 5 and 20 to a computer, even though as letters they might behave similarly in words. So instead of feeding raw IDs into the network, every one of the 27 characters gets **its own small vector of learnable numbers** — here, 2 numbers each. Think of it like giving every letter a coordinate on a 2D map. Two letters that tend to behave similarly in words (e.g., vowels) can end up near each other on that map *after training*, purely because training nudges them there — nobody hand-designs it.

`C[X]` is literally "look up the row of C for each ID in X." Since X is shape `(32, 3)` (32 examples, 3 characters each) and each lookup returns a 2-number vector, `emb` ends up shape `(32, 3, 2)`.

**Math notation:** $C \in \mathbb{R}^{27 \times 2}$ is the embedding matrix — row $C_k$ is character $k$'s vector. For an example with context $(c_1, c_2, c_3)$, the embedding is the concatenation $e = (C_{c_1}, C_{c_2}, C_{c_3}) \in \mathbb{R}^6$ (3 characters × 2 numbers = 6 numbers total).

**Important:** `C` starts as `torch.randn(...)` — pure random noise. At this point, the "map" is meaningless. It only becomes meaningful once gradient descent starts adjusting it (see the last section) — the script as written computes the loss once but never calls `.backward()` or updates `C`, `W1`, or `W2`. So as-is, it computes what the *initial, untrained* loss looks like — it doesn't train yet.

---

## Stage 3: The hidden layer — mixing the letters together

**Code:**
```python
W1 = torch.randn((6, 100))
b1 = torch.randn(100)
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
```

**Plain words:** at this point there are, for each example, 3 separate letter-vectors sitting side by side (shape `(3,2)`). But the network needs to reason about *all three letters together* — "given these three specific letters in this order, what tends to come next?" `emb.view(-1, 6)` flattens each example's `(3,2)` block into one `6`-number row (equivalent to concatenating the three letter-vectors). That's why `W1` has 6 rows — it has to accept exactly 6 numbers in.

`@ W1` is a matrix multiply: each of the 6 input numbers gets multiplied against 100 different weighted combinations, producing 100 output numbers. Each of those 100 outputs is a *different* learned "question" being asked of the input — e.g. (very loosely) "does this look like a context that ends in a vowel?" — except the network invents what those 100 questions are itself, through training. `+ b1` adds a per-feature learnable offset (a bias), and `tanh` squashes every output into the range (-1, 1), which keeps values from blowing up and lets the network represent non-linear patterns (without it, stacking two linear layers would collapse into being equivalent to one linear layer — no extra expressive power).

**Math notation:** for a batch of examples, $H = \tanh(E W_1 + b_1)$, where $E \in \mathbb{R}^{32 \times 6}$ (flattened embeddings), $W_1 \in \mathbb{R}^{6 \times 100}$, $b_1 \in \mathbb{R}^{100}$, giving $H \in \mathbb{R}^{32 \times 100}$ — 100 "hidden features" per example.

---

## Stage 4: The output layer — scoring every possible next letter

**Code:**
```python
W2 = torch.randn((100, 27))
b2 = torch.randn(27)
logits = h @ W2 + b2
```

**Plain words:** the 100 hidden features get boiled down into **one score per possible next character** (27 of them). A high score means "the network currently thinks this character is likely to come next"; a low/negative score means unlikely. These raw scores are called **logits** — not probabilities yet (they can be negative, and don't sum to 1).

**Math notation:** $\text{logits} = H W_2 + b_2$, $W_2 \in \mathbb{R}^{100 \times 27}$, giving a `(32, 27)` matrix — one row of 27 scores per example.

---

## Stage 5: Turning scores into probabilities

**Code:**
```python
counts = logits.exp()
prob = counts / counts.sum(1, keepdim=True)
```

**Plain words:** to compare "how likely is each letter," the scores need to be turned into numbers that are all positive and sum to 1 (a valid probability distribution). `exp()` makes everything positive (and amplifies the gap between high and low scores), and dividing each row by its own sum forces each row to add up to exactly 1. This two-line operation is the **softmax** function — standard enough that PyTorch has `F.softmax` built in, but writing it by hand (as here) builds intuition for what's underneath the abstraction.

**Math notation:** for row $i$, character $k$:
$$p_{i,k} = \frac{e^{\text{logit}_{i,k}}}{\sum_{j=0}^{26} e^{\text{logit}_{i,j}}}$$

---

## Stage 6: Measuring how wrong the model is — the loss

**Code:**
```python
loss = prob[torch.arange(32), Y].log().mean()
```

**Plain words:** for each of the 32 training examples, the correct next letter is already known — that's `Y`. `prob[torch.arange(32), Y]` pulls out, for each example, the probability the model *currently* assigned to the correct answer (ignoring what it thinks about the other 26 wrong characters). If the model were perfect, every one of those 32 numbers would be 1.0 (100% confidence in the right answer). Since everything starts random, they'll be close to `1/27 ≈ 0.037` on average at first.

`.log()` then `.mean()` averages the log of those correctness-probabilities across all 32 examples. This is **negative log-likelihood** — one bug to flag: it should be `-prob[...].log().mean()` (with a minus sign). Log of a number between 0 and 1 is negative, so without the minus sign, "loss" gets *more negative* as the model gets *better*, and gradient descent (which tries to *decrease* the loss) would end up doing the opposite of what's intended. Karpathy's original code includes the minus sign — worth double-checking against the video here.

**Math notation:**
$$L = -\frac{1}{N}\sum_{i=1}^{N} \log p_{i, y_i}$$

Why log? Two reasons: (1) it turns "multiply 32 tiny probabilities together" (which underflows to zero numerically) into "add 32 log-probabilities" — much more numerically stable — and (2) it penalizes confident wrong answers *much* more harshly than mildly wrong ones, which turns out to be a better training signal than raw probability.

---

## What's missing to actually *train* — how the weights get closer to the answer

The script as written stops here — it computes the loss for one random initialization but never updates the weights. The next piece is:

```python
for p in [C, W1, b1, W2, b2]:
    p.requires_grad = True

# inside the training loop:
loss.backward()               # compute how much each weight contributed to the loss
for p in [C, W1, b1, W2, b2]:
    p.data += -0.1 * p.grad   # nudge each weight slightly against its gradient
    p.grad = None
```

**Plain words on why this works:** every number in `C`, `W1`, `b1`, `W2`, `b2` starts as random noise, so the model's first guesses are garbage — expected. `loss.backward()` uses calculus (the chain rule, applied automatically) to compute, for *every single weight*, "if this one number were nudged up slightly, would the loss go up or down, and by how much?" That's the **gradient**. It's like being blindfolded on a hill and having someone tell you the slope under your feet in every direction — the bottom isn't known, but which way is downhill *right where you're standing* is.

Then a small step is taken in the downhill direction: `p.data += -learning_rate * p.grad`. Repeated thousands of times over the dataset, every weight — including the embedding table `C` — gradually shifts in whatever direction makes the model assign higher probability to the letters that actually came next in real names. Over many iterations, `C` stops being random noise and starts positioning similar-behaving letters near each other, `W1`/`W2` start encoding "which letter combinations predict which next letters," and the loss goes down because the model's predicted probabilities increasingly match reality.

**The full loop:** random weights → forward pass computes a probability and a loss → backward pass computes each weight's blame for that loss → weights step slightly downhill → repeat. Nothing in the network is hand-designed to know English name patterns — it's discovered entirely through this repeated nudging process.