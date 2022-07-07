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
    def __init__(self, dim, fn, double_input=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        self.double_input = double_input
        if self.double_input:
            self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, x_target=None, **kwargs):
        if not self.double_input:
            return self.fn(self.norm(x), **kwargs)
        else:
            return self.fn(self.norm(x), self.norm2(x_target), **kwargs)


class TokenClassifier(nn.Module):
    def __init__(self, input_dim, n_classes, embed_dim=0):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.embed_dim = embed_dim
        if embed_dim == 0:
            self.output_layer = nn.Linear(input_dim, n_classes)
        else:
            self.embed_layer = nn.Linear(input_dim, embed_dim)
            self.output_layer = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        if self.embed_dim == 0:
            return self.output_layer(self.norm(x)), x
        else:
            emb = self.embed_layer(self.norm(x))
            return self.output_layer(emb), emb


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


class DoubleTransformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.cfg_general(**kwargs)
        self.create_layers(**kwargs)

    def cfg_general(self, head_dim, num_heads, attn_layers=1, dropout=0.0, **kwargs):
        self.input_dim = head_dim * num_heads
        self.attn_layers = attn_layers
        self.dropout = dropout

    def create_layers(self, **kwargs):
        self.target_layers = nn.ModuleList([])
        self.source_layers = nn.ModuleList([])
        for _ in range(self.attn_layers):
            self.target_layers.append(
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
                            mha.DoubleMultiheadAttentionHandler(
                                **{**kwargs, "input_dim": self.input_dim}
                            ),
                            double_input=True,
                        ),
                        PreNorm(
                            self.input_dim,
                            FeedForward(self.input_dim, 2 * self.input_dim),
                        ),
                    ]
                )
            )
        for _ in range(self.attn_layers):
            self.source_layers.append(
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
        return self.source_layers[0][0].fn.get_input_dim()

    def forward(self, x, x_source, mask, mask_source):
        for attn, ff in self.source_layers:
            x_source = attn(**{"x": x_source, "mask": mask_source}) + x_source
            x_source = ff(x_source) + x_source
        for attn, attn_double, ff in self.target_layers:
            x = attn(**{"x": x, "mask": mask}) + x
            x = attn_double(**{"x": x_source, "x_target": x, "mask": mask_source}) + x
            x = ff(x) + x
        return x, x_source


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

    def forward(self, x, mask):
        for attn, ff in self.layers:
            x = attn(**{"x": x, "mask": mask}) + x
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
        which_encoder="mha",
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
        using_tabular_feat=False,
        emb_to_classifier="avg",
        F_max=[],
        tab_classifier_type="MLP",
        tab_num_heads=2,
        tab_head_dim=32,
        tab_detach=False,
        using_extra_transformer=False,
        classify_source=False,
        preprocess_tab=False,
        pretab_num_heads=2,
        pretab_head_dim=16,
        pretab_output_dim=0,
        combine_lc_tab=False,
        use_feat_modulator=False,
        **kwargs
    ):

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right nows
        self.activation = glayers.activation_dict[self.nl]
        self.which_linear = nn.Linear
        self.which_embedding = nn.Embedding
        self.which_bn = nn.BatchNorm2d
        self.using_tabular_feat = using_tabular_feat
        self.emb_to_classifier = emb_to_classifier
        self.using_extra_transformer = using_extra_transformer
        self.classify_source = classify_source

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
        # self.mha_layers  = []
        # for i in range(self.attn_layers):
        #     self.mha_layers += [mha.MultiheadAttentionHandler(**nkwargs)]
        # self.mha_layers       =  nn.ModuleList(self.mha_layers)

        # Time Modulator
        # self.input_dim_mha = self.mha_layers[-1].get_last_dim()
        self.time_modulator = tmod.EncTimeModulatorHandler(
            **{**nkwargs, "embed_dim": self.input_dim_mha}
        )

        self.tab_classifier_type = tab_classifier_type
        self.tab_num_heads = tab_num_heads
        self.tab_head_dim = tab_head_dim
        self.tab_detach = tab_detach

        self.preprocess_tab = preprocess_tab
        self.pretab_num_heads = pretab_num_heads
        self.pretab_head_dim = pretab_head_dim
        self.pretab_output_dim = pretab_output_dim

        self.combine_lc_tab = combine_lc_tab
        self.use_feat_modulator = use_feat_modulator

        self.F_len = len(F_max)
        self.dim_tab = len(F_max)

        if self.preprocess_tab or self.combine_lc_tab:
            if not self.combine_lc_tab:
                self.pretab_transformer = Transformer(
                    **{
                        **nkwargs,
                        "head_dim": self.pretab_head_dim,
                        "num_heads": self.pretab_num_heads,
                        "dropout": self.dropout_first_mha,
                    }
                )

                self.pretab_input_dim_mha = self.pretab_transformer.get_input_dim()
                self.pretab_mlp_head = TokenClassifier(
                    self.pretab_input_dim_mha,
                    self.n_classes,
                    embed_dim=self.pretab_output_dim,
                )
                self.F_len = (
                    self.pretab_output_dim
                    if self.pretab_output_dim > 0.0
                    else self.pretab_input_dim_mha
                )
                self.pretab_token = nn.Parameter(
                    torch.randn(1, 1, self.pretab_input_dim_mha)
                )
            else:
                self.pretab_input_dim_mha = self.input_dim_mha

            if self.use_feat_modulator:
                self.feat_modulator = tmod.FeatModulator(
                    **{
                        **nkwargs,
                        "F_max": F_max,
                        "embed_dim": self.pretab_input_dim_mha,
                    }
                )
            else:
                self.pretab_W_feat = nn.Parameter(
                    torch.randn(1, self.dim_tab, self.pretab_input_dim_mha)
                )
                self.pretab_b_feat = nn.Parameter(
                    torch.randn(1, self.dim_tab, self.pretab_input_dim_mha)
                )

        if self.using_tabular_feat and self.tab_classifier_type == "MLP":
            self.tab_classifier = ClassifierFeedForward(
                self.F_len + self.input_dim_mha,
                self.F_len + self.input_dim_mha,
                self.n_classes,
                self.dropout_second_mha,
            )

        if (
            self.using_tabular_feat
            and (
                self.tab_classifier_type == "Transformer"
                or self.tab_classifier_type == "TransformerLC"
                or self.tab_classifier_type == "TransformerTab"
            )
        ) or self.using_extra_transformer:
            if self.using_extra_transformer:
                self.F_len = 0

            if (
                self.tab_classifier_type == "Transformer"
                or self.using_extra_transformer
            ):
                self.tab_transformer = Transformer(
                    **{
                        **nkwargs,
                        "head_dim": self.tab_head_dim,
                        "num_heads": self.tab_num_heads,
                        "dropout": self.dropout_second_mha,
                    }
                )
            if (
                self.tab_classifier_type == "TransformerLC"
                or self.tab_classifier_type == "TransformerTab"
            ):
                self.tab_transformer = DoubleTransformer(
                    **{
                        **nkwargs,
                        "head_dim": self.tab_head_dim,
                        "num_heads": self.tab_num_heads,
                        "dropout": self.dropout_second_mha,
                    }
                )
                self.tab_input_dim_mha = self.tab_transformer.get_input_dim()
                if self.classify_source:
                    self.source_mlp_head = nn.Sequential(
                        nn.LayerNorm(self.tab_input_dim_mha),
                        nn.Linear(self.tab_input_dim_mha, self.n_classes),
                    )
            self.tab_input_dim_mha = self.tab_transformer.get_input_dim()
            self.W_feat = nn.Parameter(
                torch.randn(1, self.F_len + self.input_dim_mha, self.tab_input_dim_mha)
            )
            self.b_feat = nn.Parameter(
                torch.randn(1, self.F_len + self.input_dim_mha, self.tab_input_dim_mha)
            )

            self.tab_mlp_head = nn.Sequential(
                nn.LayerNorm(self.tab_input_dim_mha),
                nn.Linear(self.tab_input_dim_mha, self.n_classes),
            )
            self.tab_token = nn.Parameter(torch.randn(1, 1, self.tab_input_dim_mha))

            # self.feat_modulator  = tmod.FeatModulator(**{**nkwargs, 'embed_dim': self.input_dim_mha})

        if self.emb_to_classifier == "token":
            self.token = nn.Parameter(torch.randn(1, 1, self.input_dim_mha))

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.input_dim_mha),
            nn.Linear(self.input_dim_mha, self.n_classes),
        )
        self.log_softmax = torch.nn.LogSoftmax()

    def predict(
        self,
        data,
        data_var=None,
        time=None,
        mask=None,
        mask_drop=None,
        tabular_feat=None,
        **kwargs
    ):

        emb_x, t, mask = self.time_modulator(data, time, mask, var=data_var)
        if self.emb_to_classifier == "token":
            token_repeated = self.token.repeat(emb_x.shape[0], 1, 1)
            mask_token = torch.ones(emb_x.shape[0], 1, 1).float().to(emb_x.device)
            mask = torch.cat([mask_token, mask], axis=1)
            emb_x = torch.cat([token_repeated, emb_x], axis=1)

        emb_x = self.transformer(emb_x, mask)
        if self.emb_to_classifier == "avg":
            z_rep = (emb_x * mask).sum(1) / mask.sum(1)
        elif self.emb_to_classifier == "token":
            z_rep = emb_x[:, 0, :]
        return self.log_softmax(self.mlp_head(z_rep)).exp().detach()

    def predict_mix(
        self,
        data,
        data_var=None,
        time=None,
        mask=None,
        mask_drop=None,
        tabular_feat=None,
        **kwargs
    ):
        emb_y, add_loss = self(
            data,
            data_var=data_var,
            time=time,
            mask=mask,
            mask_drop=mask_drop,
            tabular_feat=tabular_feat,
            **kwargs
        )
        output = self.log_softmax(emb_y["MLPMix"]).exp().detach()
        return output

    def log_latent_classifier(
        self,
        data,
        data_var=None,
        time=None,
        mask=None,
        mask_drop=None,
        tabular_feat=None,
        **kwargs
    ):
        emb_y, add_loss = self(
            data,
            data_var=data_var,
            time=time,
            mask=mask,
            mask_drop=mask_drop,
            tabular_feat=tabular_feat,
            **kwargs
        )
        output = {key: self.log_softmax(emb_y[key]) for key in emb_y.keys()}
        return output, add_loss

    def forward(
        self,
        data,
        data_var=None,
        time=None,
        mask=None,
        mask_drop=None,
        tabular_feat=None,
        global_step=0,
        **kwargs
    ):
        output = {}
        tabular_used = tabular_feat
        # Modulate time
        emb_x, t, mask = self.time_modulator(data, time, mask, var=data_var)

        # Preprocess tabular data
        if self.preprocess_tab or self.combine_lc_tab:
            if not self.use_feat_modulator:
                pretab_emb = self.pretab_W_feat * tabular_used + self.pretab_b_feat
            else:
                pretab_emb = self.feat_modulator(tabular_used)
            if not self.combine_lc_tab:
                pretab_token_repeated = self.pretab_token.repeat(
                    pretab_emb.shape[0], 1, 1
                )
                pretab_emb = torch.cat([pretab_token_repeated, pretab_emb], axis=1)
                pretab_emb = self.pretab_transformer(pretab_emb, None)
                pretab_emb_y_pred, pretab_z_rep = self.pretab_mlp_head(
                    pretab_emb[:, 0, :]
                )
                output.update({"MLPTab": pretab_emb_y_pred})
                tabular_used = pretab_z_rep.unsqueeze(2)
            else:
                pretab_mask = torch.ones(tabular_used.shape).float().to(emb_x.device)
                emb_x = torch.cat([pretab_emb, emb_x], axis=1)
                mask = torch.cat([pretab_mask, mask], axis=1)

        # if self.using_tabular_feat and tabular_feat is not None:
        #     emb_feat = self.W_feat * tabular_feat + self.b_feat
        #     #emb_feat  = self.feat_modulator(tabular_feat)
        #     mask_feat = torch.ones(tabular_feat.shape).float().to(emb_x.device)
        #     emb_x     = torch.cat([emb_feat, emb_x], axis = 1)
        #     mask      = torch.cat([mask_feat, mask], axis = 1)
        if self.emb_to_classifier == "token":
            token_repeated = self.token.repeat(emb_x.shape[0], 1, 1)
            mask_token = torch.ones(emb_x.shape[0], 1, 1).float().to(emb_x.device)
            mask = torch.cat([mask_token, mask], axis=1)
            emb_x = torch.cat([token_repeated, emb_x], axis=1)
        emb_x = self.transformer(emb_x, mask)
        if self.emb_to_classifier == "avg":
            z_rep = (emb_x * mask).sum(1) / mask.sum(1)
        elif self.emb_to_classifier == "token":
            z_rep = emb_x[:, 0, :]
        output.update({"MLP": self.mlp_head(z_rep)})

        if (
            self.using_tabular_feat or self.using_extra_transformer
        ) and not self.combine_lc_tab:
            z_rep_aux = z_rep if not self.tab_detach else z_rep.detach()
            tabular_used = (
                tabular_used.detach()
                if self.tab_detach and self.preprocess_tab
                else tabular_used
            )

            tab_emb = (
                torch.cat([tabular_used, z_rep_aux.unsqueeze(2)], axis=1)
                if not self.using_extra_transformer
                else z_rep_aux.unsqueeze(2)
            )
            if (
                self.tab_classifier_type == "Transformer"
                or self.tab_classifier_type == "TransformerLC"
                or self.tab_classifier_type == "TransformerTab"
                or self.using_extra_transformer
            ):
                tab_emb = self.W_feat * tab_emb + self.b_feat
                if (
                    self.tab_classifier_type == "Transformer"
                    or self.using_extra_transformer
                ):
                    tab_emb = torch.cat(
                        [self.tab_token.repeat(tab_emb.shape[0], 1, 1), tab_emb], axis=1
                    )
                    tab_mask = torch.ones(tab_emb.shape[0], tab_emb.shape[1], 1).to(
                        tab_emb.device
                    )
                    tab_emb = self.tab_transformer(tab_emb, tab_mask)
                if (
                    self.tab_classifier_type == "TransformerLC"
                    or self.tab_classifier_type == "TransformerTab"
                ):
                    if self.tab_classifier_type == "TransformerTab":
                        tab_emb_target, tab_emb_source = (
                            tab_emb[:, : self.F_len, :],
                            tab_emb[:, self.F_len :, :],
                        )
                    if self.tab_classifier_type == "TransformerLC":
                        tab_emb_target, tab_emb_source = (
                            tab_emb[:, self.F_len :, :],
                            tab_emb[:, : self.F_len, :],
                        )
                    tab_emb_target = torch.cat(
                        [
                            self.tab_token.repeat(tab_emb_target.shape[0], 1, 1),
                            tab_emb_target,
                        ],
                        axis=1,
                    )
                    tab_mask_target = torch.ones(
                        tab_emb_target.shape[0], tab_emb_target.shape[1], 1
                    ).to(tab_emb.device)
                    tab_mask_source = torch.ones(
                        tab_emb_source.shape[0], tab_emb_source.shape[1], 1
                    ).to(tab_emb.device)
                    tab_emb, tab_source = self.tab_transformer(
                        tab_emb_target, tab_emb_source, tab_mask_target, tab_mask_source
                    )
                    if self.classify_source:
                        output.update(
                            {"MLPSource": self.source_mlp_head(tab_source.mean(1))}
                        )
                tab_output = self.tab_mlp_head(tab_emb[:, 0, :])
            if self.tab_classifier_type == "MLP":
                tab_output = self.tab_classifier(tab_emb.squeeze(2))
            output.update(
                {"MLPMix": tab_output if global_step > 20000 else tab_output.detach()}
            )
        add_loss = None
        if self.emb_norm_cte > 0:
            add_loss = self.emb_norm_cte * (z_rep**2).sum(-1).mean()
            if self.preprocess_tab:
                add_loss += self.emb_norm_cte * (pretab_z_rep**2).sum(-1).mean()
        if not ("MLPMix" in output.keys()):
            output["MLPMix"] = output["MLP"]
        return output, add_loss


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
