import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class LinearProj(nn.Module):
    """implementation of https://arxiv.org/pdf/2003.09291.pdf"""

    def __init__(self, n_harmonics=7, embedding_size=64, T_max=1000.0, input_size=1):
        super(LinearProj, self).__init__()
        self.embedding_size = embedding_size
        self.t_max = T_max
        self.linear_proj_ = nn.Linear(
            in_features=1, out_features=embedding_size, bias=False
        )

        self.even = nn.parameter.Parameter(
            torch.arange(self.embedding_size // 2) * 2, requires_grad=False
        )
        self.odd = nn.parameter.Parameter(
            torch.arange(self.embedding_size // 2) * 2 + 1, requires_grad=False
        )

    def time_proj(self, t):
        # multiple-dimensions
        t = t.repeat(1, 1, self.embedding_size)
        # indices
        # sin-cos transformations over time
        t[:, :, self.even] = torch.sin(
            t[:, :, self.even]
            / (self.t_max ** (2 * self.even / self.embedding_size)[None, None, :])
        )
        t[:, :, self.even] = torch.cos(
            t[:, :, self.even]
            / (self.t_max ** (2 * self.odd / self.embedding_size)[None, None, :])
        )
        return t

    def forward(self, x, t):
        # self.linear_proj_(self.linear_proj(x[:, :, None])*torch.tanh(torch.squeeze(gama_)) + torch.squeeze(beta_))
        return self.linear_proj_(x)  # + self.time_proj(t)
