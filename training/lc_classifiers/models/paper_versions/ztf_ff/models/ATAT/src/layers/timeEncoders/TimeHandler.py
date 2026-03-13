import torch
import torch.nn as nn
import torch.nn.functional as F
from .TimeFilm import TimeFilm
from .TimeFilmModified import TimeFilmModified
from .PosEmbedding import PosEmbedding
from .PosEmbeddingMLP import PosEmbeddingMLP
from .PosEmbeddingRNN import PosEmbeddingRNN
from .PosConcatEmbedding import PosConcatEmbedding
from .PosEmbeddingCadence import PosEmbeddingCadence
from .tAPE import tAPE

from .LinearProj import LinearProj
from ..utils import clones


class TimeHandler(nn.Module):
    def __init__(
        self,
        num_bands=2,
        input_size=1,
        embedding_size=64,
        Tmax=1500.0,
        pe_type="tm",
        use_emb_x_band=False,
        **kwargs
    ):
        super(TimeHandler, self).__init__()
        # general params
        self.num_bands = num_bands
        self.ebedding_size = embedding_size
        self.T_max = Tmax
        self.use_emb_x_band = use_emb_x_band

        dict_PEs = {
            "tm": TimeFilmModified,
            "pe": PosEmbedding,
            "pe_cad": PosEmbeddingCadence,
            "mlp": PosEmbeddingMLP,
            "rnn": PosEmbeddingRNN,
            "pe_concat": PosConcatEmbedding,
            "tAPE": tAPE,
        }

        # tume_encoders
        self.time_encoders = clones(
            dict_PEs[pe_type](
                embedding_size=embedding_size, input_size=input_size, Tmax=Tmax
            ),
            num_bands,
        )

        if not self.use_emb_x_band:
            self.linear_proj = nn.Sequential(
                nn.Linear(in_features=input_size, out_features=embedding_size, bias=True),
            )

    def forward(self, x, t, mask, **kwargs):
        x_mod = []
        t_mod = []
        m_mod = []

        for i in range(x.shape[-1]):
            slices_x = [slice(None)] * (x.dim() - 1) + [slice(i, i + 1)]
            slices_t = [slice(None)] * (t.dim() - 1) + [slice(i, i + 1)]

            t_band = t[slices_t]
            m_band = mask[slices_t]

            if x.dim() != t.dim():
                x_proj = x[slices_x].squeeze(-1)
            else:
                x_proj = x[slices_x]

            if not self.use_emb_x_band:
                x_proj = self.linear_proj(x_proj)
    
            x_band = self.time_encoders[i](x_proj, t_band)

            x_mod.append(x_band)
            t_mod.append(t_band)
            m_mod.append(m_band)

        x_mod = torch.cat(x_mod, axis=1)
        t_mod = torch.cat(t_mod, axis=1)
        m_mod = torch.cat(m_mod, axis=1)

        # sorted indexes along time, trwoh to the end  new samples
        indexes = (t_mod * m_mod + (~m_mod) * 9999999).argsort(axis=1)

        return (
            x_mod.gather(1, indexes.repeat(1, 1, x_mod.shape[-1])),
            m_mod.gather(1, indexes),
            t_mod.gather(1, indexes),
        )
