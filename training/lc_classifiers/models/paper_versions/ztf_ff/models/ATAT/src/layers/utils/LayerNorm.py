import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    "Construct a layernorm module"

    def __init__(self, embedding_size, eps=1e-6, **kwargs):
        super(LayerNorm, self).__init__()

        self.a_2 = nn.parameter.Parameter(torch.ones(embedding_size))
        self.b_2 = nn.parameter.Parameter(torch.zeros(embedding_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
