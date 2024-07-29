import torch
import torch.nn as nn
from ..utils import LayerNorm, clones
from .FeedForward import FeedForward
from .mha import MultiheadAttention


class TransformerEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.layer_norm = clones(LayerNorm(**kwargs), 2)
        self.attn_forward = MultiheadAttention(**kwargs)
        self.feed_forward = FeedForward(**kwargs)

    def forward(self, x, mask, **kwargs):
        x = self.attn_forward(**{"query": x, "key": x, "value": x, "mask": mask}) + x
        x = self.layer_norm[0](x)
        x = self.feed_forward(x) + x
        x = self.layer_norm[1](x)

        return x
