"""
Encoder model
"""
from . import general_layers as glayers
from . import init_model as itmodel
from . import mha
from . import optimizers
from . import time_modulator as tmod
from torch.nn import init
from torch.nn import Parameter as P

import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class TokenClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.output_layer = nn.Linear(input_dim, n_classes)

    def forward(self, x):
        return self.output_layer(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim, embed_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ClassifierFeedForward(nn.Module):
    def __init__(self, input_dim, embed_dim, n_classes, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_classes),
        )

    def forward(self, x):
        return self.net(self.norm(x))


class Transformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.cfg_general(**kwargs)
        self.create_layers(**kwargs)

    def cfg_general(self, head_dim, num_heads, attn_layers=1, dropout=0.0, **kwargs):
        self.input_dim = head_dim * num_heads
        self.attn_layers = attn_layers
        self.dropout = dropout

    def create_layers(self, **kwargs):
        self.layers = nn.ModuleList([])
        for _ in range(self.attn_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            self.input_dim,
                            mha.MultiheadAttentionHandler(
                                **{**kwargs, "input_dim": self.input_dim}
                            ),
                        ),
                        PreNorm(
                            self.input_dim,
                            FeedForward(self.input_dim, 2 * self.input_dim),
                        ),
                    ]
                )
            )

    def get_input_dim(self):
        return self.layers[0][0].fn.get_input_dim()

    def forward(self, x, mask, causal_mask=False):
        for idx, (attn, ff) in enumerate(self.layers):
            x = attn(**{"x": x, "mask": mask, "causal_mask": causal_mask}) + x
            x = ff(x) + x
        return x


class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()
        self.cfg_general(**kwargs)
        self.cfg_layers(**kwargs)
        self.cfg_bn(**kwargs)
        self.cfg_optimizers(**kwargs)
        self.create_layers(**kwargs)
        self.cfg_init(**kwargs)

    def cfg_general(
        self,
        dim_z=2,
        dataset_channel=3,
        which_encoder="vanilla",
        cat_noise_to_E=False,
        which_train_fn="VAE",
        n_classes=10,
        emb_norm_cte=0.0,
        dropout_first_mha=0.0,
        dropout_second_mha=0.0,
        drop_mask_second_mha=False,
        **kwargs
    ):
        # Data/Latent dimension
        self.dataset_channel = dataset_channel
        self.dim_z = dim_z
        self.which_encoder = which_encoder
        self.cat_noise_to_E = cat_noise_to_E
        self.which_train_fn = which_train_fn
        self.input_dim = 1 if not self.cat_noise_to_E else 2
        self.n_classes = n_classes
        self.emb_norm_cte = emb_norm_cte
        self.dropout_first_mha = dropout_first_mha
        self.dropout_second_mha = dropout_second_mha
        self.drop_mask_second_mha = drop_mask_second_mha

    def cfg_init(self, E_init="ortho", skip_init=False, **kwargs):
        self.init = E_init
        # Initialize weights
        if not skip_init:
            itmodel.init_weights(self)

    def cfg_layers(self, E_nl="relu", num_linear=0, **kwargs):
        self.nl = E_nl
        self.num_linear = num_linear

    def cfg_bn(self, BN_eps=1e-5, norm_style="in", **kwargs):
        self.BN_eps, self.norm_style = BN_eps, norm_style

    def cfg_optimizers(
        self,
        optimizer_type="adam",
        E_lr=5e-5,
        E_B1=0.0,
        E_B2=0.999,
        adam_eps=1e-8,
        weight_decay=5e-4,
        **kwargs
    ):
        self.lr, self.B1, self.B2, self.adam_eps, self.weight_decay = (
            E_lr,
            E_B1,
            E_B2,
            adam_eps,
            weight_decay,
        )
        self.optimizer_type = optimizer_type

    def create_layers(
        self,
        using_metadata=False,
        using_features=False,
        emb_to_classifier="token",
        F_max=[],
        tab_detach=False,
        tab_num_heads=4,
        tab_head_dim=32,
        tab_output_dim=0,
        combine_lc_tab=False,
        use_detection_token=False,
        **kwargs
    ):

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right nows
        self.activation = glayers.activation_dict[self.nl]
        self.which_linear = nn.Linear
        self.which_embedding = nn.Embedding
        self.which_bn = nn.BatchNorm2d
        self.using_metadata = using_metadata
        self.using_features = using_features
        self.emb_to_classifier = emb_to_classifier

        nkwargs = kwargs.copy()
        nkwargs.update(
            {
                "input_dim": self.input_dim,
                "which_linear": self.which_linear,
                "which_bn": self.which_bn,
                "which_embedding": self.which_embedding,
            }
        )
        # Using MultiheadAttention
        self.transformer = Transformer(**{**nkwargs, "dropout": self.dropout_first_mha})
        self.input_dim_mha = self.transformer.get_input_dim()
        self.time_modulator = tmod.EncTimeModulatorHandler(
            **{**nkwargs, "embed_dim": self.input_dim_mha}
        )

        self.tab_detach = tab_detach
        self.tab_num_heads = tab_num_heads
        self.tab_head_dim = tab_head_dim
        self.tab_output_dim = tab_output_dim
        self.use_detection_token = use_detection_token

        self.combine_lc_tab = combine_lc_tab

        self.F_len = len(F_max)
        self.dim_tab = len(F_max)

        if self.combine_lc_tab:
            self.tab_input_dim_mha = self.input_dim_mha
            self.tab_W_feat = nn.Parameter(
                torch.randn(1, self.dim_tab, self.tab_input_dim_mha)
            )
            self.tab_b_feat = nn.Parameter(
                torch.randn(1, self.dim_tab, self.tab_input_dim_mha)
            )

        if (self.using_metadata or self.using_features) and not self.combine_lc_tab:
            self.tab_transformer = Transformer(
                **{
                    **nkwargs,
                    "head_dim": self.tab_head_dim,
                    "num_heads": self.tab_num_heads,
                    "dropout": self.dropout_first_mha,
                }
            )
            self.tab_input_dim_mha = self.tab_transformer.get_input_dim()
            self.tab_classifier = TokenClassifier(
                self.tab_input_dim_mha, self.n_classes
            )
            self.F_len = (
                self.tab_output_dim
                if self.tab_output_dim > 0.0
                else self.tab_input_dim_mha
            )
            self.tab_token = nn.Parameter(torch.randn(1, 1, self.tab_input_dim_mha))
            self.tab_W_feat = nn.Parameter(
                torch.randn(1, self.dim_tab, self.tab_input_dim_mha)
            )
            self.tab_b_feat = nn.Parameter(
                torch.randn(1, self.dim_tab, self.tab_input_dim_mha)
            )
            self.mix_classifier = ClassifierFeedForward(
                self.F_len + self.input_dim_mha,
                self.F_len + self.input_dim_mha,
                self.n_classes,
                self.dropout_second_mha,
            )
        self.token = nn.Parameter(torch.randn(1, 1, self.input_dim_mha))
        self.lc_classifier = TokenClassifier(self.input_dim_mha, self.n_classes)
        self.log_softmax = torch.nn.LogSoftmax()
        if self.use_detection_token:
            self.detection_token = nn.Parameter(torch.randn(1, 1, self.input_dim_mha))
            self.non_detection_token = nn.Parameter(
                torch.randn(1, 1, self.input_dim_mha)
            )

    def obtain_emb_to_classify(
        self,
        emb_x,
        mask,
        time=None,
        causal_mask=False,
        obtain_multiple_times=False,
        **kwargs
    ):
        if obtain_multiple_times:
            return emb_x
        elif causal_mask:
            return self.obtain_last_emb(emb_x, mask, time)
        elif self.emb_to_classifier == "avg":
            return (emb_x * mask).sum(1) / mask.sum(1)
        elif self.emb_to_classifier == "token":
            return emb_x[:, 0, :]

    def obtain_argsort(self, time, mask):
        return (time * mask + (1 - mask) * 9999999).argsort(1)

    def obtain_mask_used(self, mask, time):
        bs = mask.shape[0]
        mask_r = mask.reshape(bs, -1)
        time_r = time.reshape(bs, -1)
        a_time = self.obtain_argsort(time_r, mask_r)
        return mask_r.gather(1, a_time)

    def obtain_last_emb(self, emb_x, mask, time):
        bs = emb_x.shape[0]
        time_r = time.reshape(bs, -1)
        mask_r = mask.reshape(bs, -1)
        a_time = self.obtain_argsort(time_r, mask_r)
        time_sorted = time_r.gather(1, a_time)
        mask_sorted = mask_r.gather(1, a_time)
        idx = (time_sorted * mask_sorted).argmax(1)
        return emb_x[torch.arange(bs), idx, :]

    def obtain_data_used(self, data, data_var, mask, mask_base, mask_pred, time):
        bs = data.shape[0]
        data_r = data.reshape(bs, -1)
        data_var_r = data_var_used.reshape(bs, -1) if data_var is not None else None
        mask_r = mask.reshape(bs, -1)
        mask_base_r = mask_base.reshape(bs, -1)
        mask_pred_r = mask_pred.reshape(bs, -1)
        time_r = time.reshape(bs, -1)
        a_time = self.obtain_argsort(time_r, mask_r)
        return (
            data_r.gather(1, a_time),
            data_r.gather(1, a_time) if data_var is not None else None,
            mask_base_r.gather(1, a_time),
            mask_pred_r.gather(1, a_time),
        )

    def mask_to_modify(self, mask_base, mask_pred):
        return mask_pred * mask_base

    def obtain_all_lc_emb(
        self,
        data,
        data_var=None,
        time=None,
        mask=None,
        tabular_feat=None,
        mask_pred=None,
        causal_mask=False,
        mask_detection=False,
        **kwargs
    ):
        emb_x, mask = self.time_modulator(
            data,
            time,
            mask,
            var=data_var,
        )
        if self.use_detection_token:
            emb_x += (
                self.detection_token * mask_detection
                + self.non_detection_token * (1 - mask_detection)
            )
        if self.emb_to_classifier == "token":
            token_repeated = self.token.repeat(emb_x.shape[0], 1, 1)
            mask_token = torch.ones(emb_x.shape[0], 1, 1).float().to(emb_x.device)
            mask = torch.cat([mask_token, mask], axis=1)
            emb_x = torch.cat([token_repeated, emb_x], axis=1)
        if self.combine_lc_tab:
            tab_emb = self.tab_W_feat * tabular_feat + self.tab_b_feat
            tab_mask = torch.ones(tabular_feat.shape).float().to(emb_x.device)
            emb_x = torch.cat([tab_emb, emb_x], axis=1)
            mask = torch.cat([tab_mask, mask], axis=1)
        emb_x = self.transformer(emb_x, mask, causal_mask=causal_mask)
        return emb_x

    def obtain_lc_emb(self, **kwargs):
        emb_x = self.obtain_all_lc_emb(**kwargs)
        return self.obtain_emb_to_classify(emb_x, **kwargs)

    def obtain_tab_emb(self, tabular_feat=None, **kwargs):
        tab_emb = self.tab_W_feat * tabular_feat + self.tab_b_feat
        tab_token_repeated = self.tab_token.repeat(tab_emb.shape[0], 1, 1)
        tab_emb = torch.cat([tab_token_repeated, tab_emb], axis=1)
        tab_emb = self.tab_transformer(tab_emb, None)
        return tab_emb[:, 0, :]

    def predict_lc(self, **kwargs):
        z_rep = self.obtain_lc_emb(**kwargs)
        return {"MLP": self.log_softmax(self.lc_classifier(z_rep))}

    def predict_tab(self, **kwargs):
        emb_x = self.obtain_tab_emb(**kwargs)
        return {"MLPTab": self.log_softmax(self.tab_classifier(emb_x))}

    def predict_mix(self, **kwargs):
        emb_y = self(**kwargs)
        return {"MLPMix": self.log_softmax(emb_y["MLPMix"])}

    def predict_all(self, **kwargs):
        emb_y = self(**kwargs)
        return {key: self.log_softmax(emb_y[key]) for key in emb_y.keys()}

    def combine_lc_tab_emb(
        self, emb_lc, emb_tab, obtain_multiple_times=False, **kwargs
    ):
        emb_lc = emb_lc if not self.tab_detach else emb_lc.detach()
        emb_tab = emb_tab if not self.tab_detach else emb_tab.detach()
        if not obtain_multiple_times:
            return torch.cat([emb_tab, emb_lc], axis=1)
        else:
            return torch.cat(
                [emb_tab.unsqueeze(1).repeat(emb_lc.shape[1]), emb_lc], axis=2
            )

    def forward(self, global_step=0, **kwargs):
        output = {}
        # Obtain lc embedding
        emb_lc = self.obtain_lc_emb(**kwargs)
        output.update({"MLP": self.lc_classifier(emb_lc)})
        if (self.using_metadata or self.using_features) and not self.combine_lc_tab:
            # Obtain tabular embedding
            emb_tab = self.obtain_tab_emb(**kwargs)
            output.update({"MLPTab": self.tab_classifier(emb_tab)})
            # Combine both embedding and we classified them with a MLP
            emb_mix = self.combine_lc_tab_emb(emb_lc, emb_tab, **kwargs)
            mix_output = self.mix_classifier(emb_mix)
            # output.update({'MLPMix': mix_output if global_step > 20000 else mix_output.detach()})
            output.update({"MLPMix": mix_output})

        if not ("MLPMix" in output.keys()):
            output["MLPMix"] = output["MLP"]
        return output


# Arguments for parser
def add_sample_parser(parser):
    parser.add_argument(
        "--attn_layers",
        type=int,
        default=1,
        help="Number of attentions layers" "(default: %(default)s)",
    )
    parser.add_argument(
        "--emb_to_classifier",
        type=str,
        default="avg",
        help="what embedding to use" "(default: %(default)s)",
    )
    parser.add_argument(
        "--using_tabular_feat",
        action="store_true",
        default=False,
        help="using tabular features?" "(default: %(default)s)",
    )
    return parser


def add_name_config(config):
    name = []
    if config["which_encoder"] == "mha" or config["which_decoder"] == "mha":
        name += ["MHA", "HD%d" % config["head_dim"], "NHead%d" % config["num_heads"]]
    return name
