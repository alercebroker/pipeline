import torch
import torch.nn as nn
from ..utils import clones
from .TransformerEncoder import TransformerEncoder
from .TransformerEncoderCnn import TransformerEncoderCnn


class Transformer(nn.Module):
    def __init__(self, encoder_type="Linear", **kwargs):
        super().__init__()

        self.num_encoders = kwargs["num_encoders"]

        self.stacked_transformers = clones(
            TransformerEncoder(**kwargs)
            if encoder_type == "Linear"
            else TransformerEncoderCnn(**kwargs),
            self.num_encoders,
        )

    def forward(self, x, mask, **kwargs):
        for l in self.stacked_transformers:
            x = l(x, mask)

        return x
