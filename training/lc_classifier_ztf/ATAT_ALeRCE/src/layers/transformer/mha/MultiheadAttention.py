import torch
import math
import copy
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"

    d_k = query.size(-1)
    nbatch = query.size(0)
    nseq = query.size(2)
    hds = query.size(1)
    # 1) dot-product
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 2) mask if exist
    if mask is not None:
        # repeat for each head apply the same mask
        mask = mask.repeat(1, hds, 1, 1)
        scores = scores.masked_fill(mask < 1, -1e9)
        # what is this ?
        scores = scores - scores.amax(dim=-1, keepdim=True).detach()
    # 3) normalize along each row
    p_attn = F.softmax(scores, dim=-1)

    """ dropout"""
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn, scores


class MultiheadAttention(nn.Module):
    def __init__(self, num_heads=None, embedding_size=None, dropout=0.1, **kwargs):
        super(MultiheadAttention, self).__init__()

        self.d_k = embedding_size // num_heads  # size of each head
        self.h = num_heads  # number of heads
        self.linears = clones(
            nn.Linear(embedding_size, embedding_size, bias=True), 3
        )  # Q, K and V models
        self.last = nn.Linear(
            embedding_size, embedding_size, bias=True
        )  # Final proyection
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, **kwargs):
        nbatches = query.size(0)

        # 0) mask attn if there is mask
        if mask is not None:
            mask = mask.unsqueeze(1).permute(0, 1, 3, 2)

        # 1) Parallel projections (query, key , value)
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn, sc = attention(
            query, key, value, mask=mask, dropout=self.dropout, **kwargs
        )

        # 3) "Concat" and apply a final linear.
        output = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        output = self.last(output)

        return output
