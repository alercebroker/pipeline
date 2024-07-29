import functools
import math
import torch
import torch.nn as nn
from torch.nn import init


class CnnForward(nn.Module):
    def __init__(
        self,
        embedding_size,
        embedding_size_sub,
        cnn_kernel,
        max_pool_kernel,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=embedding_size,
                out_channels=embedding_size,
                kernel_size=cnn_kernel,
                padding=cnn_kernel // 2,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(
                kernel_size=max_pool_kernel,
                padding=max_pool_kernel // 2,
                stride=1,
                dilation=1,
            ),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = x.permute(0, 2, 1)

        return x
