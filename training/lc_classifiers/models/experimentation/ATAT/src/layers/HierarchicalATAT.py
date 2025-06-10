import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from .timeEncoders import TimeHandler
from .embeddings import Embedding
from .transformer import Transformer
from .classifiers import TokenClassifier, MixedClassifier
from .tokenEmbeddings import Token


class HierarchicalATAT(nn.Module):
    def __init__(self, **kwargs):
        super(HierarchicalATAT, self).__init__()

        self.general_ = kwargs["general"]
        self.lightcv_ = kwargs["lc"]
        self.feature_ = kwargs["ft"]

        # ---------- parámetros jerárquicos ----------
        self.chunk_size  = self.lightcv_.get("chunk_size", 100)   # ★ NUEVO
        self.use_hier    = self.lightcv_.get("hierarchical", True)

        # ---------- BLOQUE Light-curves ----------
        if self.general_["use_lightcurves"]:
            self.time_encoder = TimeHandler(**kwargs["lc"])
            self.transformer_lc = Transformer(**kwargs["lc"]) 

            d_model = kwargs["lc"]["embedding_size"]

            if self.use_hier:             
                self.gru_global = nn.GRU(
                    input_size = d_model,
                    hidden_size = d_model,
                    num_layers = 2,
                    dropout     = 0.2,
                    batch_first = True,
                ) 
            #else:
            #    # Modo tradicional: un solo Transformer
            #    self.transformer_lc = Transformer(**kwargs["lc"])


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

    def embedding_feats(self, f, mask_tab):
        batch_size = f.size(0)
        f_mod = self.embedding_ft(**{"f": f})
        token_ft = self.token_ft(batch_size)
        f_mod = torch.cat([token_ft, f_mod], dim=1)
        if mask_tab is not None:
            m_token = torch.ones(batch_size, 1, 1).float().to(f.device)
            mask_tab = torch.cat([m_token, mask_tab], dim=1)        
        return f_mod, mask_tab

    def embedding_light_curve(self, x, t, mask=None, **kwargs):
        batch_size = x.size(0)
        m_token = torch.ones(batch_size, 1, 1).float().to(x.device)
        x_mod, m_mod, t_mod = self.time_encoder(**{"x": x, "t": t, "mask": mask})
        x_mod = torch.cat([self.token_lc(batch_size), x_mod], axis=1)
        m_mod = torch.cat([m_token, m_mod], axis=1)
        return x_mod, m_mod, t_mod

    def lc_hierarchical(self, x, t, mask):
        """
        x, t, mask : (B, S, n_bands)   con mask=True para valores reales
        Devuelve un embedding global por curva (B, d_model)
        """
        B, S, n_bands = x.shape
        C             = self.chunk_size
        device        = x.device
        d_model       = self.lightcv_["embedding_size"]

        # ---- 1) PADDING para múltiplo de chunk_size ----
        n_chunks = math.ceil(S / C)
        S_pad    = n_chunks * C
        pad_len  = S_pad - S
        if pad_len > 0:
            x     = F.pad(x,     (0,0, 0,pad_len))
            t     = F.pad(t,     (0,0, 0,pad_len))
            mask  = F.pad(mask,  (0,0, 0,pad_len))

        # ---- 2) RE-SHAPE a (B*n_chunks, C, n_bands) ----
        x4d    = x.view(B, n_chunks, C, n_bands)
        t4d    = t.view(B, n_chunks, C, n_bands)
        m4d    = mask.view(B, n_chunks, C, n_bands)

        BNC    = B * n_chunks
        x_flat = x4d.reshape(BNC, C, n_bands)
        t_flat = t4d.reshape(BNC, C, n_bands)
        m_flat = m4d.reshape(BNC, C, n_bands)

        # ---- 3) TimeHandler + Transformer LOCAL ----
        x_mod, m_mod, _ = self.time_encoder(x_flat, t_flat, m_flat)
        #  (BNC, L', d), (BNC, L')

        token_local  = self.token_lc(BNC)                      # (BNC,1,d)
        x_cat        = torch.cat([token_local, x_mod], dim=1)  # (BNC, L'+1, d)
        m_token      = torch.ones(BNC, 1, 1, dtype=torch.bool, device=device)
        m_cat        = torch.cat([m_token, m_mod], dim=1)      # (BNC, L'+1)

        out_local    = self.transformer_lc(x_cat, m_cat) # (BNC, L'+1, d)
        cls_local    = out_local[:, 0, :]                      # (BNC, d)

        # ---- 4) Volver a (B, n_chunks, d) ----
        cls_local    = cls_local.view(B, n_chunks, d_model)    # (B, n_chunks, d)

        # ---- 5) chunk_mask: True si chunk tiene ≥1 dato real ----
        chunk_mask   = m4d.any(dim=(2,3))                      # (B, n_chunks)
        lengths      = chunk_mask.sum(dim=1).cpu()             # longitudes reales

        # ---- 6) GRU agregador (packed) ----
        packed       = nn.utils.rnn.pack_padded_sequence(
                           cls_local, lengths,
                           batch_first=True, enforce_sorted=False)
        _, h_n       = self.gru_global(packed)                 # (1,B,d)
        emb_global   = h_n[-1]                          # (B,d)

        return emb_global
    
    def lc_flat(self, x, t, mask):
        """
        Versión original (un solo Transformer global).
        """
        B = x.size(0)
        m_token = torch.ones(B, 1, 1, device=x.device)
        x_mod, m_mod, _ = self.time_encoder(x=x, t=t, mask=mask)
        x_mod = torch.cat([self.token_lc(B), x_mod], dim=1)
        m_mod = torch.cat([m_token, m_mod],  dim=1)

        out   = self.transformer_lc(x_mod, m_mod)      # (B, L'+1, d)
        emb   = out[:, 0, :]                           # (B, d)
        return emb

    def forward(
        self,
        data=None,
        data_err=None,
        time=None,
        tabular_feat=None,
        mask=None,
        mask_tabular=None,
        **kwargs
    ):
        x_cls, f_cls, m_cls = None, None, None

        if self.general_["use_lightcurves"]:
            if self.general_["use_lightcurves_err"]:
                data = torch.stack((data, data_err), dim=data.dim() - 1)

            if self.use_hier:
                x_emb = self.lc_hierarchical(data, time, mask)      # ★ NUEVO
            else:
                x_emb = self.lc_flat(data, time, mask)

            x_cls = self.classifier_lc(x_emb)

        if self.general_["use_metadata"] or self.general_["use_features"]:
            f_mod, m_tab = self.embedding_feats(**{"f": tabular_feat, "mask_tab": None}) # None
            f_emb = self.transformer_ft(**{"x": f_mod, "mask": m_tab}) # None
            f_cls = self.classifier_ft(f_emb[:, 0, :])        

        if self.general_["use_lightcurves"] and (
            self.general_["use_metadata"] or self.general_["use_features"]
        ):
            m_cls = self.classifier_mix(
                torch.cat([f_emb[:, 0, :], x_emb], axis=1)
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
