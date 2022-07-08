"""
Time modulator
"""
import math
import torch
import torch.nn as nn


class FeatModulator(nn.Module):
    def __init__(self, Feat_M, embed_dim, F_max, using_norm_feat=False, **kwargs):
        super().__init__()
        self.F_len = len(F_max)
        self.alpha_sin = nn.Parameter(torch.randn(self.F_len, Feat_M, embed_dim))
        self.alpha_cos = nn.Parameter(torch.randn(self.F_len, Feat_M, embed_dim))
        self.beta_sin = nn.Parameter(torch.randn(self.F_len, Feat_M, embed_dim))
        self.beta_cos = nn.Parameter(torch.randn(self.F_len, Feat_M, embed_dim))
        self.bias = nn.Parameter(torch.randn(1, self.F_len, embed_dim))
        # self.F_max     = F_max
        self.Feat_M = Feat_M
        self.register_buffer("ar", torch.arange(Feat_M).unsqueeze(0).unsqueeze(0))
        if not using_norm_feat:
            self.register_buffer(
                "F_max", torch.tensor(F_max).unsqueeze(0).unsqueeze(2).unsqueeze(3)
            )
        else:
            self.F_max = 10.0

    def get_sin(self, x):
        # x: Batch size x Feat_len x 1:
        x = x.unsqueeze(2)
        # x: Batch size x Feat_len x 1 x 1:
        return torch.sin(
            (2 * math.pi * self.ar * x.repeat(1, 1, 1, self.Feat_M)) / self.F_max
        )

    def get_cos(self, x):
        # x: Batch size x Feat_len x 1:
        x = x.unsqueeze(2)
        return torch.cos(
            (2 * math.pi * self.ar * x.repeat(1, 1, 1, self.Feat_M)) / self.F_max
        )

    def get_sin_cos(self, x):
        return self.get_sin(x), self.get_cos(x)

    def forward(self, x):
        # x: Batch size x Feat_len x 1:
        sin_emb, cos_emb = self.get_sin_cos(x)
        # sin_emb: Batch size x Feat_len x 1 x Feat_M:
        # alpha_sin: Feat_len x Feat_M, embed_dim:
        return (
            torch.matmul(sin_emb, self.alpha_sin).squeeze(2)
            + torch.matmul(cos_emb, self.alpha_cos).squeeze(2)
            + self.bias
        )


class TimeModulator(nn.Module):
    def __init__(self, M, embed_dim, T_max):
        super().__init__()
        self.alpha_sin = nn.Parameter(torch.randn(M, embed_dim))
        self.alpha_cos = nn.Parameter(torch.randn(M, embed_dim))
        self.beta_sin = nn.Parameter(torch.randn(M, embed_dim))
        self.beta_cos = nn.Parameter(torch.randn(M, embed_dim))
        self.T_max = T_max
        self.M = M
        self.register_buffer("ar", torch.arange(M).unsqueeze(0).unsqueeze(0))

    def get_sin(self, t):
        # t: Batch size x time x dim, dim = 1:
        return torch.sin((2 * math.pi * self.ar * t.repeat(1, 1, self.M)) / self.T_max)

    def get_cos(self, t):
        # t: Batch size x time x dim, dim = 1:
        return torch.cos((2 * math.pi * self.ar * t.repeat(1, 1, self.M)) / self.T_max)

    def get_sin_cos(self, t):
        return self.get_sin(t), self.get_cos(t)

    def forward(self, x, sin_emb, cos_emb):
        alpha = torch.matmul(sin_emb, self.alpha_sin) + torch.matmul(
            cos_emb, self.alpha_cos
        )
        beta = torch.matmul(sin_emb, self.beta_sin) + torch.matmul(
            cos_emb, self.beta_cos
        )
        return x * alpha + beta


class TimeModulatorHandler(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        M,
        T_max,
        dataset_channel,
        which_linear=nn.Linear,
        which_pre_encoder="",
        **kwargs
    ):
        super().__init__()
        self.to_emb = which_linear(input_dim, embed_dim)
        self.dataset_channel = dataset_channel
        self.time_mod = []
        for i in range(dataset_channel):
            self.time_mod += [TimeModulator(M, embed_dim, T_max)]
        self.time_mod = nn.ModuleList(self.time_mod)


class EncTimeModulatorHandler(TimeModulatorHandler):
    def __init__(self, cat_noise_to_E=False, **kwargs):
        super().__init__(**kwargs)
        self.cat_noise_to_E = cat_noise_to_E

    def forward(self, x, t, mask, var=None):
        # [Batch x seq len x features]
        all_mod_emb_x = []
        if self.cat_noise_to_E and var is not None:
            all_x = [
                torch.stack([x[:, :, i], var[:, :, i]], -1)
                for i in range(self.dataset_channel)
            ]
        else:
            all_x = [x[:, :, i].unsqueeze(2) for i in range(self.dataset_channel)]
        all_time = [t[:, :, i].unsqueeze(2) for i in range(self.dataset_channel)]
        all_mask = [mask[:, :, i].unsqueeze(2) for i in range(self.dataset_channel)]
        for i in range(self.dataset_channel):
            ### Modulate input ###
            x, time, mask = all_x[i], all_time[i], all_mask[i]
            emb_x = self.to_emb(x)
            time_emb_sin, time_emb_cos = self.time_mod[i].get_sin_cos(time)
            all_mod_emb_x += [self.time_mod[i](emb_x, time_emb_sin, time_emb_cos)]
        mod_emb_x, time, mask = (
            torch.cat(all_mod_emb_x, 1),
            torch.cat(all_time, 1),
            torch.cat(all_mask, 1),
        )
        a_time = time.argsort(1)
        return (
            mod_emb_x.gather(1, a_time.repeat(1, 1, mod_emb_x.shape[-1])),
            time.gather(1, a_time),
            mask.gather(1, a_time),
        )


class EncJustTimeModulatorHandler(TimeModulatorHandler):
    def __init__(self, dataset_channel=1, **kwargs):
        super().__init__(**{"dataset_channel": 1, **kwargs})

    def forward(self, x, t):
        # [Batch x seq len x features]
        emb_x = self.to_emb(x)
        time_emb_sin, time_emb_cos = self.time_mod[0].get_sin_cos(t)
        mod_emb_x = self.time_mod[0](emb_x, time_emb_sin, time_emb_cos)
        return mod_emb_x


class DecTimeModulatorHandler(TimeModulatorHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, z, time, index):
        emb_z = self.to_emb(z).unsqueeze(1).repeat(1, time.shape[1], 1)
        time_emb_sin, time_emb_cos = self.time_mod[index].get_sin_cos(time)
        emb_z = self.time_mod[index](emb_z, time_emb_sin, time_emb_cos)
        return emb_z


# Arguments for parser
def add_sample_parser(parser):
    ### Attention stuff ###
    parser.add_argument(
        "--M",
        type=int,
        default=16,
        help="Number of component of fourier modulator (time)?(default: %(default)s)",
    )
    parser.add_argument(
        "--Feat_M",
        type=int,
        default=16,
        help="Number of component of fourier modulator (feat)?(default: %(default)s)",
    )
    return parser


def add_name_config(config):
    name = []
    if config["which_encoder"] == "mha" or config["which_decoder"] == "mha":
        name += ["Tmod", "M%d" % config["M"]]
    return name
