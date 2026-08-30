import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

g = torch.Generator().manual_seed(2147483647)  # for reproducibility


# Let's train a deeper network
class Linear:
    def __init__(
        self, fan_in, fan_out, bias=True
    ):  # n_inputs, n_outputs, bias(True/False)
        self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True

        # parameters (trained with backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
