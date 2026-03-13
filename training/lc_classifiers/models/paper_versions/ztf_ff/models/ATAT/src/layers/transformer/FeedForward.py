import functools
import math
import torch
import torch.nn as nn
from torch.nn import init


class FeedForward(nn.Module):
    def __init__(self, embedding_size, embedding_size_sub, dropout=0.1, **kwargs):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_size, embedding_size_sub),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size_sub, embedding_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
