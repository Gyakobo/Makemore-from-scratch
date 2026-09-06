# Why the Linear Bias `b1` Is Redundant Before Batch Norm

## TL;DR

Any bias in a `Linear` layer that is **immediately followed by a Batch Norm layer**
is a dead parameter. Batch norm subtracts the batch mean as its first step, which
cancels the bias exactly. The shift that the bias was meant to provide is instead
supplied by batch norm's own `bnbias` parameter.

**Rule of thumb:** a `Linear` layer followed by `BatchNorm` should use `bias=False`.

---

## The setup

In `mlp.py` the layer looks like this:

```
hprebn  = embcat @ W1 + b1                              # linear layer
                    ↓
hpreact = bngain * (hprebn - mean) / std + bnbias       # batch norm
```

The question: if this is all just math, how do we *know* `b1` isn't needed?

---

## The math

Split the linear layer into the useful part and the bias:

```
z       = embcat @ W1        # the part that depends on the input
hprebn  = z + b1             # add the constant bias
```

Batch norm's first operation is to subtract the **batch mean**. Because `b1` is a
constant (the same for every example in the batch), it passes through the mean
untouched:

```
mean(hprebn) = mean(z + b1) = mean(z) + b1
```

Now look at the centered numerator inside batch norm:

```
hprebn - mean(hprebn) = (z + b1) - (mean(z) + b1)
                      = z - mean(z)
```

**The `b1` cancels exactly.** It is added and then subtracted right back out.
Whatever value `b1` takes — `0`, `5`, `-100` — the output is identical.

---

## Where the bias actually lives

The network still *has* a bias — it just moved. After centering, batch norm
re-introduces a learnable shift with `bnbias`:

```
hpreact = bngain * (z - mean(z)) / std + bnbias
```

`bnbias` sits **after** the normalization, where a constant offset actually
survives. So it, not `b1`, is the parameter that does the shifting.

Keeping both is like writing `y = a + b` when you can only ever observe the sum
`a + b`: you cannot recover two independent numbers, so one of them is wasted.

---

## Why this matters in practice

`b1` lives *before* a step that deletes constant offsets, so:

- It has **no effect** on the output for any value.
- It receives **no useful gradient** — nothing to learn.
- It just **consumes memory and compute** for nothing.

Removing it is purely about correctness of intent and efficiency, not about
changing the model's behavior.

---

## Consequences in the code

Because `b1` no longer exists:

1. Drop it from the parameter list:

   ```python
   parameters = [C, W1, W2, b2, bngain, bnbias]   # no b1
   ```

2. Drop the `+ b1` term from the forward pass:

   ```python
   hprebn = embcat @ W1            # no "+ b1"
   ```

If either still references `b1`, you get `NameError: name 'b1' is not defined`.

---

## The one-line takeaway

> A bias added right before batch norm is subtracted right back out by the mean
> centering. The real, effective bias is batch norm's `bnbias`.
