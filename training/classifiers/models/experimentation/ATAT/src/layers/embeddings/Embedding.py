import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, length_size, embedding_size, **kwargs):
        super(Embedding, self).__init__()

        self.tab_W_feat = nn.Parameter(torch.randn(1, length_size, embedding_size))
        self.tab_b_feat = nn.Parameter(torch.randn(1, length_size, embedding_size))

    def forward(self, f):
        return self.tab_W_feat * f + self.tab_b_feat
