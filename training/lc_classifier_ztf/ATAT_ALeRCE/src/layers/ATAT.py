import torch
import torch.nn as nn


from .timeEncoders import TimeHandler
from .embeddings import Embedding
from .transformer import Transformer
from .classifiers import TokenClassifier, MixedClassifier
from .tokenEmbeddings import Token

import copy


class ATAT(nn.Module):
    def __init__(self, **kwargs):
        super(ATAT, self).__init__()

        self.general_ = kwargs["general"]
        self.lightcv_ = kwargs["lc"]
        self.feature_ = kwargs["ft"]

        # Lightcurve Transformer
        if self.general_["use_lightcurves"]:
            self.time_encoder = TimeHandler(**kwargs["lc"])
            self.transformer_lc = Transformer(**kwargs["lc"])
            self.classifier_lc = TokenClassifier(
                num_classes=self.general_["num_classes"], **kwargs["lc"]
            )
            self.token_lc = Token(**kwargs["lc"])

        # Tabular Transformer
        if self.general_["use_metadata"] or self.general_["use_features"]:
            self.embedding_ft = Embedding(**kwargs["ft"])
            self.transformer_ft = Transformer(**kwargs["ft"])
            self.classifier_ft = TokenClassifier(
                num_classes=self.general_["num_classes"], **kwargs["ft"]
            )
            self.token_ft = Token(**kwargs["ft"])

        # Mixed Classifier (Lightcurve and tabular)
        if self.general_["use_lightcurves"] and any(
            [self.general_["use_metadata"], self.general_["use_features"]]
        ):

            input_dim = kwargs["lc"]["embedding_size"] + kwargs["ft"]["embedding_size"]
            self.classifier_mix = MixedClassifier(
                input_dim=input_dim, **kwargs["general"]
            )

        # init model params
        self.init_model()

    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, 0, 0.02)

    def embedding_feats(self, f):
        batch_size = f.size(0)
        f_mod = self.embedding_ft(**{"f": f})
        token_ft = self.token_ft(batch_size)
        return torch.cat([token_ft, f_mod], dim=1)

    def embedding_light_curve(self, x, t, mask=None, **kwargs):
        batch_size = x.size(0)
        m_token = torch.ones(batch_size, 1, 1).float().to(x.device)
        x_mod, m_mod, t_mod = self.time_encoder(**{"x": x, "t": t, "mask": mask})
        x_mod = torch.cat([self.token_lc(batch_size), x_mod], axis=1)
        m_mod = torch.cat([m_token, m_mod], axis=1)
        return x_mod, m_mod, t_mod

    def forward(
        self,
        data=None,
        data_err=None,
        time=None,
        tabular_feat=None,
        mask=None,
        **kwargs
    ):
        x_cls, f_cls, m_cls = None, None, None

        if self.general_["use_lightcurves"]:
            if self.general_["use_lightcurves_err"]:
                data = torch.stack((data, data_err), dim=data.dim() - 1)

            x_mod, m_mod, _ = self.embedding_light_curve(
                **{"x": data, "t": time, "mask": mask}
            )
            x_emb = self.transformer_lc(**{"x": x_mod, "mask": m_mod})
            x_cls = self.classifier_lc(x_emb[:, 0, :])

        if self.general_["use_metadata"] or self.general_["use_features"]:
            f_mod = self.embedding_feats(**{"f": tabular_feat})
            f_emb = self.transformer_ft(**{"x": f_mod, "mask": None})
            f_cls = self.classifier_ft(f_emb[:, 0, :])

        if self.general_["use_lightcurves"] and (
            self.general_["use_metadata"] or self.general_["use_features"]
        ):
            m_cls = self.classifier_mix(
                torch.cat([f_emb[:, 0, :], x_emb[:, 0, :]], axis=1)
            )

        return x_cls, f_cls, m_cls

    def predict_mix(self, data, time, tabular_feat, mask, **kwargs):
        x_mod, m_mod, _ = self.embedding_light_curve(
            **{"x": data, "t": time, "mask": mask}
        )
        x_emb = self.transformer_lc(**{"x": x_mod, "mask": m_mod})

        f_mod = self.embedding_feats(**{"f": tabular_feat})
        f_emb = self.transformer_ft(**{"x": f_mod, "mask": None})

        m_cls = self.classifier_mix(torch.cat([f_emb[:, 0, :], x_emb[:, 0, :]], axis=1))
        m_cls = torch.softmax(m_cls, dim=1)
        return m_cls

    def predict_lc(self, data, time, mask, **kwargs):
        x_mod, m_mod, _ = self.embedding_light_curve(
            **{"x": data, "t": time, "mask": mask}
        )
        x_emb = self.transformer_lc(**{"x": x_mod, "mask": m_mod})
        x_cls = self.classifier_lc(x_emb[:, 0, :])
        x_cls = torch.softmax(x_cls, dim=1)
        return x_cls

    def predict_tab(self, tabular_feat, **kwargs):
        f_mod = self.embedding_feats(**{"f": tabular_feat})
        f_emb = self.transformer_ft(**{"x": f_mod, "mask": None})
        f_cls = self.classifier_ft(f_emb[:, 0, :])
        f_cls = torch.softmax(f_cls, dim=1)
        return f_cls

    def change_clf(self, num_nuevas_clases=22):
        # Reemplazar el clasificador light curve
        embedding_size_lc = self.classifier_lc.output_layer.in_features
        self.classifier_lc = TokenClassifier(embedding_size_lc, num_nuevas_clases)
