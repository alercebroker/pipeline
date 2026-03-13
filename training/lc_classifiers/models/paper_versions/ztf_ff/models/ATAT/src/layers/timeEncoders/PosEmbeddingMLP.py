import torch
import torch.nn as nn
import math


class PosEmbeddingMLP(nn.Module):
    def __init__(self, m_layers=32, embedding_size=192, input_size=1, **kwargs):
        super(PosEmbeddingMLP, self).__init__()
        """
        Re-implementation of Learnable Fourier Features from https://arxiv.org/abs/2106.02795
        """

        assert embedding_size % 2 == 0

        self.w = nn.Linear(input_size, embedding_size // 2, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, m_layers),
            nn.GELU(),
            nn.Linear(m_layers, embedding_size),
        )

        self.linear_proj = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=embedding_size, bias=True),
        )

        self.scale = embedding_size ** -0.5

    def forward(self, x, t):
        batch_size, seq_len, _ = t.shape
        t = t / seq_len

        pe = self.w(t)
        pe = self.scale * torch.cat((torch.cos(pe), torch.sin(pe)), dim=-1)
        pe = self.mlp(pe)

        return self.linear_proj(x) + pe
