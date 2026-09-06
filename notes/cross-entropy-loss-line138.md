# Understanding the loss line

```python
loss = -logprobs[range(n), Yb].mean()
```

This is line 138 of `backpropagation.py` — the manual, unpacked version of
`F.cross_entropy(logits, Yb)`. It computes the **average negative log-likelihood
(cross-entropy) loss** over the batch.

## The setup

By this line, `logprobs` is a **matrix of shape `(n, 27)`** = `(32, 27)`:

- **32 rows** — one per example in the batch (`n = batch_size = 32`).
- **27 columns** — one per possible character (a–z plus the `.` token).

Each entry `logprobs[i, j]` is the **log-probability** the model assigns to
character `j` being the answer for example `i`. Since probabilities are between
0 and 1, their logs are always **negative**:

- `log(1.0)  = 0`
- `log(0.5) ≈ -0.69`
- `log(0.01) ≈ -4.6`

`Yb` is a vector of **32 integers** — the *correct* character index for each
example. E.g. `Yb = [5, 13, 0, ...]` means example 0's true next char is index 5,
example 1's is 13, etc.

## Breaking the line into 4 pieces

### 1. `range(n)` → the row selector

`range(32)` produces `[0, 1, 2, ..., 31]` — literally "every row."

### 2. `logprobs[range(n), Yb]` → fancy indexing (the key part)

This pairs up the two lists **element-by-element** (not as a grid). It picks:

```
logprobs[0,  Yb[0]]
logprobs[1,  Yb[1]]
logprobs[2,  Yb[2]]
...
logprobs[31, Yb[31]]
```

So for **each row, it plucks out the single column corresponding to the correct
answer.** Result: a vector of 32 numbers — the log-prob the model gave to the
*right* character in each example.

**Tiny example** — a batch of just 3 examples and 4 characters:

```python
logprobs = [[-2.3, -0.1, -3.0, -1.5],   # example 0
            [-0.7, -1.2, -0.9, -2.1],   # example 1
            [-1.1, -0.4, -2.2, -0.3]]   # example 2

Yb = [1, 0, 3]   # correct chars: 1, then 0, then 3
```

`logprobs[range(3), Yb]` = `logprobs[[0,1,2], [1,0,3]]` picks:

- row 0, col 1 → `-0.1`
- row 1, col 0 → `-0.7`
- row 2, col 3 → `-0.3`

Result: `[-0.1, -0.7, -0.3]`

### 3. `.mean()` → average over the batch

`(-0.1 + -0.7 + -0.3) / 3 = -0.3667`

We average so the loss doesn't depend on batch size.

### 4. `-` → flip the sign

`-(-0.3667) = 0.3667`

## Why this *is* the loss (the intuition)

This is the **negative log-likelihood** (cross-entropy). The logic:

- We want the model to assign **high probability** to the correct character.
- High prob → log near `0` (e.g. `log(0.9) = -0.1`) → small loss. ✅
- Low prob → very negative log (e.g. `log(0.01) = -4.6`) → big loss. ❌

The negative sign turns "log-probs are negative" into "loss is positive," and
minimizing it pushes the model to make the correct character's probability as
close to 1 as possible.

**In one sentence:** grab the model's log-probability for the correct answer in
each of the 32 examples, average them, and negate — that's the average
per-example penalty for being wrong.
