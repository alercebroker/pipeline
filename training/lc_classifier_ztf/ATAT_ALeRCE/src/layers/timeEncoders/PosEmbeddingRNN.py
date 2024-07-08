import torch
import torch.nn as nn
import math


class PosEmbeddingRNN(nn.Module):
    def __init__(self, embedding_size=192, input_size=1, **kwargs):
        super(PosEmbeddingRNN, self).__init__()

        assert embedding_size % 2 == 0

        self.w = nn.Linear(input_size, embedding_size // 2, bias=False)
        self.rnn = nn.GRU(embedding_size, embedding_size, batch_first=True)
        self.mlp = nn.Linear(embedding_size, embedding_size)

        self.scale = embedding_size ** -0.5

        self.linear_proj = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=embedding_size, bias=True),
        )

    def forward(self, x, t):
        batch_size, seq_len, _ = t.shape
        t = t / seq_len

        pe = self.w(t)
        pe = self.scale * torch.cat((torch.cos(pe), torch.sin(pe)), dim=-1)
        pe, _ = self.rnn(pe)
        pe = self.mlp(pe)

        return self.linear_proj(x) + pe
