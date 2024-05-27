import torch
import torch.nn as nn
from ..utils import LayerNorm, clones
from .FeedForward import FeedForward
from .CnnForward import CnnForward
from .mha import MultiheadAttention


class TransformerEncoderCnn(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.layer_norm = clones(LayerNorm(**kwargs), 2)
        self.attn_forward = MultiheadAttention(**kwargs)
        self.feed_forward = FeedForward(**kwargs)
        self.cnn_forward = CnnForward(**kwargs)

    def forward(self, x, mask, **kwargs):
        x = self.attn_forward(**{"query": x, "key": x, "value": x, "mask": mask}) + x
        x = self.layer_norm[0](x)
        x = self.feed_forward(x) + x
        x = self.layer_norm[1](x)

        """ the nex input mus be convolutional """
        token, attn = x[:, 0:1, :], x[:, 1:, :]
        attn = self.cnn_forward(attn)
        out = torch.cat([token, attn], axis=1)

        return out
