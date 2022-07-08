"""
MLP resnet layers
"""
from . import general_layers as glayers
from . import time_modulator as tmod

import torch
import torch.nn as nn


class DBlock(nn.Module):
    def __init__(self, in_dim, out_dim, which_linear=nn.Linear, activation=None):
        super(DBlock, self).__init__()

        self.in_dim, self.out_dim = in_dim, out_dim
        self.activation = activation
        # Conv layers
        self.linear1 = which_linear(self.in_dim, self.out_dim)
        self.linear2 = which_linear(self.out_dim, self.out_dim)

    def forward(self, x):
        h = self.activation(x)
        h = self.linear1(h)
        h = self.activation(h)
        h = self.linear2(h)
        return h + x


class MlpMod(torch.nn.Module):
    def __init__(self, **kwargs):
        super(MlpMod, self).__init__()
        self.create_layers(**kwargs)

    def create_layers(
        self,
        embed_dim_mlp,
        num_mlp_blocks,
        which_post_decoder="",
        which_linear=nn.Linear,
        **kwargs
    ):
        self.embed_dim = embed_dim_mlp
        self.num_mlp_blocks = num_mlp_blocks
        self.act = torch.nn.LeakyReLU(inplace=False)
        self.time_modulator = tmod.DecTimeModulatorHandler(
            **{
                **kwargs,
                "embed_dim": self.get_input_dim(),
                "which_linear": which_linear,
            }
        )

        self.blocks = []
        for index in range(self.num_mlp_blocks):
            self.blocks += [
                DBlock(
                    in_dim=self.embed_dim,
                    out_dim=self.embed_dim,
                    which_linear=which_linear,
                    activation=self.act,
                )
            ]
        self.blocks = nn.ModuleList(self.blocks)

    def requires_time(self):
        return True

    def get_input_dim(self):
        return self.embed_dim

    def get_last_dim(self):
        return self.embed_dim

    def forward(
        self, z, mask=None, return_attention=False, t_used=None, band_index=None
    ):
        emb = self.time_modulator(z, t_used, band_index)
        for layer in self.blocks:
            emb = layer(emb)
        return emb


# Arguments for parser
def add_sample_parser(parser):
    parser.add_argument(
        "--embed_dim_mlp",
        type=int,
        default=32,
        help="Number of hidden units of MLP" "(default: %(default)s)",
    )
    parser.add_argument(
        "--num_mlp_blocks",
        type=int,
        default=4,
        help="Number of hidden units of MLP" "(default: %(default)s)",
    )
    return parser


def add_name_config(config):
    name = []
    if config["which_decoder"] == "mlp":
        name += [
            "MLP",
            "Hdim%d" % config["embed_dim_mlp"],
            "num%d" % config["num_mlp_blocks"],
        ]
    return name
